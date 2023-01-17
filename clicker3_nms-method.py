!wget --no-check-certificate -O data.zip "https://onedrive.live.com/download?cid=0A908748F2E95D62&resid=A908748F2E95D62%2186888&authkey=AD9ZXBTJuvwP1bQ"
!unzip data.zip
!pip install detecto
import os
from PIL import Image
import torch
from detecto import visualize
from torchvision import transforms
import xml.etree.ElementTree as ET
import numpy as np
import tqdm
from terminaltables import AsciiTable
class MarsDataset():
    def __init__(self, label_path, image_path, transform, classes):
        self.label = list(sorted(os.listdir(label_path)))
        self.image = list(sorted(os.listdir(image_path)))
        self.label_path = label_path
        self.image_path = image_path
        if transform:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self._classes = ['__background__'] + classes
        self._int_mapping = {label: index for index, label in enumerate(self._classes)}
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.image[idx])
        label_path = os.path.join(self.label_path, self.label[idx])
        
        img = Image.open(img_path)
        
        root = ET.parse(label_path).getroot()
        
        filename = root.find('filename').text
        assert filename == self.image[idx]
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        labels = []
        boxes = []
        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text
            if label == "butte":
                continue
            xmin = int(box[0].text)
            ymin = int(box[1].text)
            xmax = int(box[2].text)
            ymax = int(box[3].text)
            labels.append(self._int_mapping[label])
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        image = self.transforms(img)
        
        return image, target
    
    def __len__(self):
        return len(self.image)
classes = ['crater','cone']
test = MarsDataset("label/test/", "data/test/", None, classes)
idx2class = {k+1:v for k,v in enumerate(classes)}
image, targets = test[0]
boxes = targets["boxes"]
labels = [idx2class[l.item()] for l in targets["labels"]]
visualize.show_labeled_image(image, boxes, labels)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
import torchvision
from torchvision.models.detection import FasterRCNN

num_classes = 3
backbone = torchvision.models.vgg16().features
backbone.out_channels = 512
fasterRCNN = FasterRCNN(backbone, num_classes=num_classes)

fasterRCNN.to(device)

fasterRCNN.load_state_dict(torch.load("./faster-rcnn.pth",map_location=device))
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def get_batch_statistics(target, pred_boxes, pred_labels, pred_scores, iou_threshold):
    batch_metrics = []
    detected_boxes = []
    
    true_boxes = target["boxes"][0]
    
    true_positives = np.zeros(pred_boxes.shape[0])
    
    for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        
        iou, box_index = bbox_iou(pred_box.unsqueeze(0), true_boxes).max(0)
        
        if iou >= iou_threshold and box_index not in detected_boxes:
            true_positives[pred_i] = 1
            detected_boxes += [box_index]
            
    batch_metrics.append([true_positives, pred_scores.cpu(), pred_labels.cpu()])
    
    return batch_metrics
    
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]
    
    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()
    return keep
    


def eval(model, test_loader, nms_threshold=0.3, iou_threshold=0.4, conf_threshold=0.01, nms_method="tradition"):
    sample_metrics = []
    labels = []
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    model.to(device)
    
    
    for image,target in test_loader:
        with torch.no_grad():
            image = image.to(device)
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
            labels += targets["labels"].tolist()

            pred = model(image)[0]

        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]
        
        true_boxes = target["boxes"]
        true_labels = target["labels"]

        # nms
        if nms_method == "tradition":
            keep = torchvision.ops.nms(pred_boxes,pred_scores,nms_threshold)
            pred_labels = pred_labels[keep]
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            
            # 去除分数低于阈值的框
            pred_boxes = pred_boxes[pred_scores>conf_threshold]
            pred_labels = pred_labels[pred_scores>conf_threshold]
            pred_scores = pred_scores[pred_scores>conf_threshold]
        elif nms_method == "soft":
            keep = soft_nms_pytorch(pred_boxes,pred_scores, sigma=0.5, thresh=conf_threshold, cuda=True)
            keep = torch.as_tensor(keep, dtype=torch.long)
            pred_labels = pred_labels[keep]
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            
        
        

        sample_metrics += get_batch_statistics(target, pred_boxes, pred_labels, pred_scores, iou_threshold)
        
    
    
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    
    ap_table = [["Index", "Class name", "Precision", "Recall", "AP"]]
    class_names = ['__background__', 'crater', 'cone'] 
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % precision[i], "%.5f" % recall[i], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")
test_loader = torch.utils.data.DataLoader(
    test, batch_size=1, shuffle=False, num_workers=4)


eval(fasterRCNN, test_loader, nms_threshold=0.5, iou_threshold=0.4, conf_threshold=0.001, nms_method="tradition")
eval(fasterRCNN, test_loader, nms_threshold=0.5, iou_threshold=0.4, conf_threshold=0.001, nms_method="soft")
idx2class = {k+1:v for k,v in enumerate(classes)}
# 经过nms之后展示处理结果
keep = torchvision.ops.nms(boxes,scores,0.1)
labels = labels[keep]
boxes = boxes[keep]
scores = scores[keep]
labels = [idx2class[val.item()] for val in labels]
visualize.show_labeled_image(image, boxes, labels)