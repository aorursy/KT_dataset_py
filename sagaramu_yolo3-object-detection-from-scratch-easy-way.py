# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from __future__ import division


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from tqdm import tqdm_notebook 

from IPython.display import clear_output

import matplotlib.pyplot as plt
import matplotlib.patches as patches


import math
import time
import tqdm

import torchvision.transforms as transforms

from PIL import Image

import cv2

import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


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


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


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


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
# Building Block Layers
class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()
class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss
def pad_to_square(img, pad_value):
    c,h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
img = transforms.ToTensor()(Image.open('../input/Image/1/0.jpg').convert('RGB'))
print(img.size())
img,pad = pad_to_square(img,0)
img= resize(img,416)
print(img.shape)
image = np.transpose(img, (1,2,0))
plt.imshow(image)
cla = []
dic_class = {}
for i in range(4703):
    filename = '../input/Annotation/1/' + str(i) + '.txt'
#     print(filename)
    file = open(filename, 'r',encoding="utf8")
    lines = file.read().split('\n')
    lines = [line for line in lines if len(line)>0]
#     print(lines)
    for line in lines:
      cla.append(line.split()[-1])
    
cla = list(set(cla))   
for j , val in enumerate(cla):
    if val not in list(dic_class.keys()):
        dic_class[val] = j
print(len(dic_class))

def labelsprocess(path):
    file = open(path, 'r',encoding="utf8")
    lines = file.read().split('\n')
    lines = [line for line in lines if len(line)>0]
    
    labels = []
    for i in lines:
        data = i.split()
        cl = dic_class[data[-1]]
        data = data[:-1]
        data  = list(map(float,data))
        x1 = min(data[0:4])
        y1 = min(data[4:8])
        w = max(data[0:4])-min(data[0:4])
        h = max(data[4:8])-min(data[4:8])
        labels.append([cl,x1,y1,w,h])
        
    return labels
tar = labelsprocess('../input/Annotation/1/0.txt')
fr = cv2.imread("../input/Image/1/0.jpg")
frame_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
for i in tar:
    frame_rgb = cv2.rectangle(frame_rgb, (int(i[1]), int(i[2])), (int(i[1])+int(i[3]), int(i[2])+int(i[4])), (255, 0, 0), 5)
plt.imshow(frame_rgb)
def dataprocess(image_path,targets_path):
    
    # Image processing
    
    img = transforms.ToTensor()(Image.open(image_path).convert('RGB'))
    _, h, w = img.shape
    
    img, pad = pad_to_square(img, 0)
    _, padded_h, padded_w = img.shape
        
    img = resize(img, 416)
    
    h_factor,w_factor = (1,1)
    # label processing
    targets = None
    
    labels = labelsprocess(targets_path)
    boxes = torch.tensor(labels)
    
    x1 = boxes[:,1]
    y1 = boxes[:,2]
    x2 = boxes[:,1] + boxes[:,3]
    y2 = boxes[:,2] + boxes[:,4]
    
    # Adjust for added padding
    x1 += pad[0]
    y1 += pad[2]
    x2 += pad[1]
    y2 += pad[3]

    # Returns (x, y, w, h)
    boxes[:, 1] = ((x1 + x2) / 2) / padded_w
    boxes[:, 2] = ((y1 + y2) / 2) / padded_h
    boxes[:, 3] *= w_factor / padded_w
    boxes[:, 4] *= h_factor / padded_h
    
    
    targets = torch.zeros((len(boxes), 6))
    targets[:, 1:] = boxes
    
    return img, targets
yololayer1 = YOLOLayer([(81, 82), (135, 169), (344, 319)],2299,416)
yololayer2 = YOLOLayer([(23, 27), (37, 58), (81, 82)],2299,416)
class yolomini(nn.Module):
    
    def __init__(self):
        
        super(yolomini,self).__init__()
        
        self.conv1 = nn.Sequential(
                        
                        nn.Conv2d(3,16,3,1,1,bias =False),  # 0
                        nn.BatchNorm2d(16),
                        nn.LeakyReLU(0.1 , inplace=True),
            
                        )
        
                        
        self.pool1  =   nn.MaxPool2d(2,2,0)                 # 1
        
        self.conv2  = nn.Sequential(
            
                        nn.Conv2d(16,32,3,1,1,bias=False),  # 2
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.1,inplace=True),
            
                        )
        
        self.pool2 =    nn.MaxPool2d(2,2,0)                #3
            
        self.conv3 =  nn.Sequential(
                        nn.Conv2d(32,64,3,1,1,bias=False), #4
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.1,inplace=True),
            
                        )
        
        self.pool3 =    nn.MaxPool2d(2,2,0)               #5
            
        self.conv4 =  nn.Sequential(
                        nn.Conv2d(64,128,3,1,1,bias=False), #6
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.1,inplace=True),
                        )
            
            
        self.pool4 =  nn.MaxPool2d(2,2,0)               #7
                        
        self.conv5 =  nn.Sequential(             
                        nn.Conv2d(128,256,3,1,1,bias=False),#8
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.1,inplace=True),
                        )
            
        self.pool5 =  nn.MaxPool2d(2,2,0)               #9
            
        self.conv6 =  nn.Sequential(
                        nn.Conv2d(256,512,3,1,1,bias=False),#10
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.1,inplace=True),
                        )
            
        self.pool6 = nn.Sequential(
                        nn.ZeroPad2d((0, 1, 0, 1)),
                        nn.MaxPool2d(2,1,0),               #11
                        )
                        
        self.conv7 = nn.Sequential(
                        nn.Conv2d(512,1024,3,1,1,bias=False),#12
                        nn.BatchNorm2d(1024),
                        nn.LeakyReLU(0.1,inplace=True),
                        )
                        
        self.conv8 = nn.Sequential(
                        nn.Conv2d(1024,256,1,1,bias=False),#13
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.1,inplace=True),
                        )
            
        self.conv9 = nn.Sequential(
                        nn.Conv2d(256,512,3,1,1,bias=False),#14
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.1,inplace=True),
                        )
        
        self.conv10 =   nn.Conv2d(512,6912,1,1)   # 15
                        
        
        self.yolo1 = yololayer1  #16
          
        self.route1 = EmptyLayer()  #17
        
        self.conv11 = nn.Sequential(
            
                        nn.Conv2d(256,128,1,1,bias=False),#18
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.1,inplace=True),
        
                        )
        
        self.upsample1 = nn.Upsample(scale_factor=2,mode='nearest') #19
        
        self.route2 = EmptyLayer() #20
        
        self.conv12 = nn.Sequential(
                        
                        nn.Conv2d(384,256,3,1,1), #21
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.1,inplace=True),
                        ) 
                        
        self.conv13 = nn.Conv2d(256,6912,1,1) #22
                        
        self.yolo2 = yololayer2 #23
        
        
    def forward(self,X,targets = None):
        img_dim = X.shape[2]
#         print(img_dim)
        loss = 0
        
        x1 = self.conv1(X)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)
#         print("x7",x7.shape)
        x8 = self.pool4(x7)
#         print("x8",x8.shape)
        x9 = self.conv5(x8)
#         print("x9",x9.shape)
        x10 = self.pool5(x9)
#         print("x10",x10.shape)
        x11 = self.conv6(x10)
#         print("x11",x11.shape)
        x12 = self.pool6(x11)
#         print("x12",x12.shape)
        x13 = self.conv7(x12)
#         print("x13",x13.shape)
        x14 = self.conv8(x13)
#         print("x14",x14.shape)
        x15 = self.conv9(x14)
        x16 = self.conv10(x15)
        x17,layer_loss = self.yolo1(x16,targets,img_dim)
        loss += layer_loss
        x18 = x14
        x19 = self.conv11(x18)
        x20 = self.upsample1(x19)
#         print(x20.shape,x9.shape,x19.shape)
        x21 = torch.cat([x20,x9],1)
        x22 = self.conv12(x21)
        x23 = self.conv13(x22)
#         print("x23",x23.shape)
        x24,layer_loss = self.yolo2(x23,targets,img_dim)
        loss += layer_loss
        
        yolo_outputs = to_cpu(torch.cat([x17,x24],1))
        
        return yolo_outputs if targets is None else (loss,yolo_outputs)
        
        
def createbatch(batch_size,batch_start):
    batch_end = batch_start + batch_size
    if batch_end <= 4703 :
        batch = list(range(batch_start,batch_end))
    else:
        temp = batch_size - (4703-batch_start)
        batch = list(range(batch_start,4703)) + list(range(0,temp))
    images = []
    boxes =  []
#     print(batch)
    for i in batch:
        image_path  = '../input/Image/1/' + str(i) + '.jpg'
        target_path = '../input/Annotation/1/' + str(i) + '.txt'
#     print(image_path,target_path)
        img,targets = dataprocess(image_path,target_path)
        images.append(img)
        boxes.append(targets)
#     print(type(boxes))
    for j, b in enumerate(boxes):
            b[:, 0] = j
    targets = torch.cat(boxes, 0)
    imgs = torch.stack(images)
    
    return imgs, targets
imgs,targets = createbatch(10,4703)
#Training Model
Epoch = 100
batch_size = 2
batch_start = np.random.randint(0,4703)
model = yolomini()
model.to(device)
optimizer = optim.SGD(model.parameters(),lr=0.0001)
loss_epoch = []
for num_epoch in tqdm_notebook(range(Epoch), total=Epoch, unit="Epoch"):
    Loss_batch = []
    N_batch = 1000//batch_size
    for num_batch in tqdm_notebook(range(N_batch),total=N_batch , unit ="Batch"):
        imgs,targets = createbatch(batch_size,batch_start)
        imgs = imgs.to(device)
        targets = targets.to(device)
        loss,output = model(imgs,targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        Loss_batch.append(loss)
        batch_start = batch_start + batch_size
        torch.cuda.empty_cache()
    clear_output(wait=True)
    plt.plot(Loss_batch)
    plt.title("Loss v/s batches")
    plt.xlabel("Bat")
    plt.ylabel("Loss")
    plt.show()
    break
    loss_epoch.append(loss)
plt.plot(loss_epoch)
plt.title("loss v/s epoch")
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()