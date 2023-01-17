!cp -r ../input/yolomodel/* .
from utils.datasets import *

from utils.utils import *

from models.experimental import *

import sys

sys.path.insert(0, "../input/weightedboxesfusion")

from ensemble_boxes import *

import torch

import glob

import pandas as pd
weights = ["../input/40efolds/best48fold0.pt", "../input/40efolds/best48fold1.pt", "../input/40efolds/best48fold2.pt", "../input/40efolds/best48fold3.pt", "../input/40efolds/best40fold4.pt"]

print(len(weights))

source = "../input/global-wheat-detection/test"

imgsz = 1024

conf_t = 0.5

iou_t = 0.8

built_in_tta = True



device = torch_utils.select_device("0" if torch.cuda.is_available() else "")

half = device.type != 'cpu'  # half precision only supported on CUDA
skip_box_thr = 0.43

iou_thr = 0.6

score_thr = 0.25



ws = [0.15, 0.15, 0.27, 0.15, 0.28]



def run_wbf(boxes,scores, image_size=1024, iou_thr=0.4, skip_box_thr=0.34, weights=None):

    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes, scores
def TTAImage(image, index):

    image1 = image.copy()

    if index==0: 

        rotated_image = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)

        return rotated_image

    elif index==1:

        rotated_image2 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)

        rotated_image2 = cv2.rotate(rotated_image2, cv2.ROTATE_90_CLOCKWISE)

        return rotated_image2

    elif index==2:

        rotated_image3 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)

        rotated_image3 = cv2.rotate(rotated_image3, cv2.ROTATE_90_CLOCKWISE)

        rotated_image3 = cv2.rotate(rotated_image3, cv2.ROTATE_90_CLOCKWISE)

        return rotated_image3

    elif index == 3:

        return image1

    

def rotBoxes90(boxes, im_w, im_h):

    ret_boxes =[]

    for box in boxes:

        x1, y1, x2, y2 = box

        x1, y1, x2, y2 = x1-im_w//2, im_h//2 - y1, x2-im_w//2, im_h//2 - y2

        x1, y1, x2, y2 = y1, -x1, y2, -x2

        x1, y1, x2, y2 = int(x1+im_w//2), int(im_h//2 - y1), int(x2+im_w//2), int(im_h//2 - y2)

        x1a, y1a, x2a, y2a = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

        ret_boxes.append([x1a, y1a, x2a, y2a])

    return np.array(ret_boxes)



def detect1Image(img, img0, model, device, aug):

    img = img.transpose(2,0,1)



    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32



    img /= 255.0

    if img.ndimension() == 3:

        img = img.unsqueeze(0)

    

    # Inference

    pred = model(img, augment=aug)[0]

    

    # Apply NMS

    pred = non_max_suppression(pred, conf_t, iou_t, merge=True, classes=None, agnostic=False)

    

    boxes = []

    scores = []

    for i, det in enumerate(pred):  # detections per image

        # save_path = 'draw/' + image_id + '.jpg'

        if det is not None and len(det):

            # Rescale boxes from img_size to img0 size

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()



            # Write results

            for *xyxy, conf, cls in det:

                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

                scores.append(conf)



    return np.array(boxes), np.array(scores) 
def detect():

    models = []



    for w in weights if isinstance(weights, list) else [weights]:

        models.append(torch.load(w, map_location=device)['model'].to(device).float().eval())

        

    dataset = LoadImages(source, img_size=imgsz)

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    for model in models:

        if half:

            model.half()  # to FP16

        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once



    all_paths, all_bboxes, all_confs = [], [], []

    for p, img, img0, _ in dataset:

        print()

        img = img.transpose(1,2,0) # [H, W, 3]



        all_paths.append(p)

        m_box, m_score = [], []

        for model in models:

            enboxes = []

            enscores = []

            for i in range(4):

                img1 = TTAImage(img, i)

                boxes, scores = detect1Image(img1, img0, model, device, aug=False)

                for _ in range(3-i):

                    boxes = rotBoxes90(boxes, *img.shape[:2])            

                enboxes.append(boxes)

                enscores.append(scores)



            boxes, scores = detect1Image(img, img0, model, device, aug=True)

            enboxes.append(boxes)

            enscores.append(scores) 

            

            boxes, scores = run_wbf(enboxes, enscores, image_size=1024, iou_thr=iou_thr, skip_box_thr=skip_box_thr)    

            boxes = boxes.astype(np.int32).clip(min=0, max=1023)

            indices = scores >= score_thr

            boxes = boxes[indices]

            scores = scores[indices]

            m_box.append(boxes)

            m_score.append(scores)

            

        all_bboxes.append(m_box)

        all_confs.append(m_score)

    return all_paths, all_bboxes, all_confs
with torch.no_grad():

    res = detect()

paths,all_boxes,all_confs = res

resultsYOLO =dict()

def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

bts = []

sts = []

for row in range(len(paths)):

    image_id = paths[row].split("/")[-1].split(".")[0]

    boxes, scores = run_wbf(all_boxes[row],all_confs[row], skip_box_thr = skip_box_thr, iou_thr = iou_thr, weights = None)

    boxes = (boxes*1024/1024).astype(np.int32).clip(min=0, max=1023)

    

    resultsYOLO[image_id] = [boxes, scores]

    

    

#     result = {'image_id': image_id,'PredictionString': format_prediction_string(boxes, scores)}

#     results.append(result)

#     bts.append(boxes)

#     sts.append(scores)

!rm -rf *

# test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

# test_df.to_csv('submission.csv', index=False)

# test_df.head()
resultsYOLO["2fd875eaa"]
import sys

sys.path.insert(0, "/kaggle/input/weightedboxesfusion")



import ensemble_boxes



import pandas as pd

import numpy as np

import cv2

import os, re

import gc

import random



import torch

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



from torch.utils.data import DataLoader, Dataset



from matplotlib import pyplot as plt
DATA_DIR = "/kaggle/input/global-wheat-detection"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

test_df.shape
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __len__(self) -> int:

        return len(self.image_ids)



    def __getitem__(self, idx: int):

        image_id = self.image_ids[idx]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        records = self.df[self.df['image_id'] == image_id]

    

        if self.transforms:

            sample = {"image": image}

            sample = self.transforms(**sample)

            image = sample['image']



        return image, image_id

def fasterrcnn_resnet50_fpn(path,pretrained_backbone=False):



    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)

    model = FasterRCNN(backbone, 2)

    model.load_state_dict(torch.load(path))

    model.to(DEVICE)

    model.eval()

    return model
def get_model_101( path,pretrained=False):

    backbone = resnet_fpn_backbone('resnet101', pretrained=pretrained)

    model = FasterRCNN(backbone, num_classes=2)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    model.load_state_dict(torch.load(path))

    model.to(DEVICE)

    model.eval()

    return model
model =get_model_101("../input/faster-cnn-final-model/fastercnn_resnet_101_145.pth")

model2 =get_model_101("../input/faster-cnn-final-model/fastercnn_resnet_101_145.pth")

model3=get_model_101("../input/faster-cnn-101-155-fix/fastercnn_resnet_101_fix_155.pth")

model4= get_model_101("../input/fastercnn-155/fastercnn_101_155.pth")

model5= get_model_101("../input/faster-cnn-125/fastercnn_resnet_101_125.pth")

faster_CNN_50_90=fasterrcnn_resnet50_fpn("../input/isfix-model/fastercnn_50_fix.pth")

faster_CNN_50_90_2=fasterrcnn_resnet50_fpn("../input/isfix-model/fastercnn_50_fix.pth")

faster_CNN_50_105=fasterrcnn_resnet50_fpn("../input/faster-50-fix-115/fastercnn_50_fix_115.pth")



models=[model,model3,model4,model5,faster_CNN_50_105,faster_CNN_50_90,faster_CNN_50_90_2

       ,model2

       ]
from ensemble_boxes import *



def make_ensemble_predictions(images):

    images = list(image.to(DEVICE) for image in images)    

    result = []

    for model in models:

        with torch.no_grad():

            outputs = model(images)

            result.append(outputs)

            del model

            gc.collect()

            torch.cuda.empty_cache()

    return result



def run_wbf_ensemble(predictions, image_index, image_size=1024, iou_thr=0.45, skip_box_thr=0.43, weights=None):

    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]

    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
def get_test_transforms():

    return A.Compose([

            ToTensorV2(p=1.0)

        ], p=1.0)
def collate_fn(batch):

    return tuple(zip(*batch))



test_dataset = WheatDataset(test_df, os.path.join(DATA_DIR, "test"), get_test_transforms())



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=1,

    drop_last=False,

    collate_fn=collate_fn

)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
class BaseWheatTTA:

    """ author: @shonenkov """

    image_size = 1024



    def augment(self, image):

        raise NotImplementedError

    

    def batch_augment(self, images):

        raise NotImplementedError

    

    def deaugment_boxes(self, boxes):

        raise NotImplementedError



class TTAHorizontalFlip(BaseWheatTTA):

    """ author: @shonenkov """



    def augment(self, image):

        return image.flip(1)

    

    def batch_augment(self, images):

        return images.flip(2)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]

        return boxes



class TTAVerticalFlip(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return image.flip(2)

    

    def batch_augment(self, images):

        return images.flip(3)

    

    def deaugment_boxes(self, boxes):

        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]

        return boxes

    

class TTARotate90(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return torch.rot90(image, 1, (1, 2))



    def batch_augment(self, images):

        return torch.rot90(images, 1, (2, 3))

    

    def deaugment_boxes(self, boxes):

        res_boxes = boxes.copy()

        res_boxes[:, [0,2]] = self.image_size - boxes[:, [3,1]] 

        res_boxes[:, [1,3]] = boxes[:, [0,2]]

        return res_boxes



class TTACompose(BaseWheatTTA):

    """ author: @shonenkov """

    def __init__(self, transforms):

        self.transforms = transforms

        

    def augment(self, image):

        for transform in self.transforms:

            image = transform.augment(image)

        return image

    

    def batch_augment(self, images):

        for transform in self.transforms:

            images = transform.batch_augment(images)

        return images

    

    def prepare_boxes(self, boxes):

        result_boxes = boxes.copy()

        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)

        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)

        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)

        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)

        return result_boxes

    

    def deaugment_boxes(self, boxes):

        for transform in self.transforms[::-1]:

            boxes = transform.deaugment_boxes(boxes)

        return self.prepare_boxes(boxes)
from itertools import product



tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], 

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
def make_tta_predictions(images, score_threshold=0.57):

    with torch.no_grad():

        images = torch.stack(images).float().to(DEVICE)

        predictions = []

        for tta_transform in tta_transforms:

            result = []

            #ensemble predict

            outputs = make_ensemble_predictions(tta_transform.batch_augment(images.clone()))

            #outputs = model(tta_transform.batch_augment(images.clone()))



            for i, image in enumerate(images):

                #chose the boxes and scores

                boxes, scores, labels = run_wbf_ensemble(outputs, image_index=i)

                #boxes = outputs[i]['boxes'].data.cpu().numpy()   

                #scores = outputs[i]['scores'].data.cpu().numpy()

                

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]

                boxes = tta_transform.deaugment_boxes(boxes.copy())

                result.append({

                    'boxes': boxes,

                    'scores': scores[indexes],

                })

            predictions.append(result)

    return predictions
def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.4, skip_box_thr=0.43, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist() for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
resultsFASTER = {}



for images, image_ids in test_data_loader:



    predictions = make_tta_predictions(images)

    for i, image in enumerate(images):

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        boxes = boxes.round().astype(np.int32).clip(min=0, max=1023)

        image_id = image_ids[i]

        

        resultsFASTER[image_id] = [boxes, scores]

        

        

        
def run_wbf(boxes,scores, image_size=1024, iou_thr=0.6, skip_box_thr=0.43, weights=None):

    labels0 = [np.ones(len(scores[idx])) for idx in range(len(scores))]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels0, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes, scores
pred_string = []

bts, sts, ids = [], [], []

results = dict()

for image_id, faster in resultsFASTER.items():

    yolo = resultsYOLO[image_id]

    scores_ = [faster[1], yolo[1],yolo[1]]

    boxes_ = [faster[0], yolo[0],yolo[0]]

    boxes, scores = run_wbf(boxes_, scores_, iou_thr = 0.45, skip_box_thr=0.01)

    boxes = boxes.round().astype(np.int32).clip(min=0, max=1023)

    

    indices = scores >= 0.3

    boxes = boxes[indices]

    scores = scores[indices]

    

    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    

    result = {'image_id': image_id, 'PredictionString': format_prediction_string(boxes, scores)}

    pred_string.append(result)

    ids.append(image_id)

    bts.append(boxes)

    sts.append(scores)



    

test_df = pd.DataFrame(pred_string, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)
test_df
PATH = "../input/global-wheat-detection/test/"

font = cv2.FONT_HERSHEY_SIMPLEX 

fig = plt.figure(figsize=[30,30])

bts =[b.round().astype(np.int32).clip(min=0, max=1023) for b in bts]

for i in range(len(image_id)):

    fig.add_subplot(4, 3, i+1)



    p = PATH + ids[i] + ".jpg"

    image = cv2.imread(p, cv2.IMREAD_COLOR)

    color = (255, 0, 0) 



    for b, s in zip(bts[i], sts[i]):

        if s> 0.2:

            b2 = int(b[0]+ b[2])

            b3 = int(b[1] + b[3])

            b[0] = int(b[0])

            b[1] = int(b[1])



            image = cv2.rectangle(image, (b[0],b[1]), (b2,b3), (255,0,0), 2) 

            image = cv2.putText(image, '{:.2}'.format(s), (b[0]+np.random.randint(3),b[1]), font, 1, color, 2, cv2.LINE_AA)

    plt.title(ids[i])

    plt.imshow(image)

fig.tight_layout()

plt.show()