# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import sys

sys.path.insert(0, "../input/weightedboxesfusion")



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torchvision import transforms

from matplotlib import pyplot as plt

from torchvision import transforms

import torchvision 

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torch.utils.data import Dataset,DataLoader

import glob

import albumentations as A

import cv2

from albumentations.pytorch.transforms import ToTensorV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

from ensemble_boxes import *

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def load_resnet101_model(checkpoint_path):

    num_classes = 2  # 1 class (wheat) + background

    backbone = resnet_fpn_backbone('resnet101', pretrained= False)

    model_faster = FasterRCNN(backbone, num_classes)

    # replace the pre-trained head with a new one

    in_features = model_faster.roi_heads.box_predictor.cls_score.in_features

    model_faster.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load(checkpoint_path)

    model_faster.load_state_dict(checkpoint['model_state_dict'])

    model_faster.to(device);

    model_faster.eval()

    return model_faster



def load_resnet50_model(checkpoint_path):

    num_classes = 2  # 1 class (wheat) + background

    backbone = resnet_fpn_backbone('resnet50', pretrained= False)

    model_faster = FasterRCNN(backbone, num_classes)

    # replace the pre-trained head with a new one

    in_features = model_faster.roi_heads.box_predictor.cls_score.in_features

    model_faster.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model_faster.to(device);

    checkpoint = torch.load(checkpoint_path)

    model_faster.load_state_dict(checkpoint['model_state_dict'])

    model_faster.to(device);

    model_faster.eval()

    return model_faster





def load_resnet152_model(checkpoint_path):

    num_classes = 2  # 1 class (wheat) + background

    backbone = resnet_fpn_backbone('resnet152', pretrained= False)

    model_faster = FasterRCNN(backbone, num_classes)

    # replace the pre-trained head with a new one

    in_features = model_faster.roi_heads.box_predictor.cls_score.in_features

    model_faster.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model_faster.to(device);

    checkpoint = torch.load(checkpoint_path)

    model_faster.load_state_dict(checkpoint['model_state_dict'])

    model_faster.to(device);

    model_faster.eval()

    return model_faster

resnet101_33e = "../input/resnet10133e-1024/cp33.pt"

resnet50 = "../input/resnet50-40e/resnet50_40e.pt"

resnet101_25e = "../input/resnet101mymodel/cp25.pt"

# resnet101_50e = "../input/resnet101-50e/cp49.pt"

resnet152_20e = "../input/resnet152-20e/cp20.pt"

resnet152_19e ="../input/resnet152-19e/cp19 (1).pt"
models = [

    load_resnet50_model(resnet50),

#     load_resnet101_model(resnet101_33e),

#     load_resnet101_model(resnet101_25e),

#     load_resnet101_model(resnet101_50e),

    load_resnet152_model(resnet152_20e),

    load_resnet152_model(resnet152_19e)

]
DATA_ROOT_PATH = '../input/global-wheat-detection/test/'



class DatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
def get_valid_transforms():

    return A.Compose([

            ToTensorV2(p=1.0),

        ], p=1.0)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
dataset = DatasetRetriever(np.array([path.split('/')[-1][:-4] for path in glob.glob(f'{DATA_ROOT_PATH}/*.jpg')]),get_valid_transforms())



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=1,

    shuffle=False,

    num_workers=2,

    drop_last=False,

    collate_fn=collate_fn

)
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

        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]

        res_boxes[:, [1,3]] = boxes[:, [2,0]]

        return res_boxes

    

class TTARotate180(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return torch.rot90(image, 2, (1, 2))



    def batch_augment(self, images):

        return torch.rot90(images, 2, (2, 3))

    

    def deaugment_boxes(self, boxes):

        boxes[:, [0,1,2,3]] = self.image_size - boxes[:, [2,3,0,1]]

        return boxes

    

class TTARotate270(BaseWheatTTA):

    """ author: @shonenkov """

    

    def augment(self, image):

        return torch.rot90(image, 3, (1, 2))



    def batch_augment(self, images):

        return torch.rot90(images, 3, (2, 3))

    

    def deaugment_boxes(self, boxes):

        res_boxes = boxes.copy()

        res_boxes[:, [0,2]] = boxes[:, [1,3]]

        res_boxes[:, [1,3]] = self.image_size - boxes[:, [2,0]]

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
def make_tta_predictions(model,images, score_threshold=0.35):

  model.eval()

  model.to(device)

  with torch.no_grad():

      images = torch.stack(images).float().cuda()

      predictions = []

      for tta_transform in tta_transforms:

          result = []

          det = model(tta_transform.batch_augment(images.clone()))



          for i in range(images.shape[0]):

              boxes = det[i]['boxes'].detach().cpu().numpy()[:,:4]    

              scores = det[i]['scores'].detach().cpu().numpy()[:]

              indexes = np.where(scores > score_threshold)[0]

              boxes = boxes[indexes]

            #   boxes[:, 2] = boxes[:, 2] + boxes[:, 0]

            #   boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

              boxes = tta_transform.deaugment_boxes(boxes.copy())

              result.append({

                  'boxes': boxes,

                  'scores': scores[indexes],

              })

          predictions.append(result)

  return predictions



def make_ensemble_predictions(images):

    images = list(image.to(device) for image in images)    

    result = []

    for model in models:

        predictions = make_tta_predictions(model, images)

        with torch.no_grad():

            for i in range(len(images)):

                boxes, scores, labels = run_wbf(predictions, image_index=i)

                outputs = {'boxes': boxes,'labels': labels, 'scores':scores }

                result.append([outputs])

                torch.cuda.empty_cache()



    return result





# def make_ensemble_predictions(images):

#     images = list(image.to(device) for image in images)    

#     result = []

#     for model in models:

#         with torch.no_grad():

#             outputs = model(images)

#             result.append(outputs)

#             torch.cuda.empty_cache()

#     return result





def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.45, skip_box_thr=0.35,score_threshold = 0.35, weights=None,conf_type='max'):

    boxes = [prediction[image_index]['boxes']/(image_size) for prediction in predictions]

    scores = [prediction[image_index]['scores'] for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)

    indexes = np.where(scores > score_threshold)[0]

    boxes = boxes[indexes]    

    scores = scores[indexes]

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
from itertools import product



tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], 

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), TTARotate180(), TTARotate270(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))

    
import matplotlib.pyplot as plt



for j, (images, image_ids) in enumerate(data_loader):

    break



predictions = make_ensemble_predictions(images)



i = 0

sample = images[i].permute(1,2,0).cpu().numpy()

boxes, scores, labels = run_wbf(predictions, image_index=i, iou_thr=0.45, skip_box_thr=0.35, score_threshold = 0.35, weights=None,conf_type='max')

# print(len(boxes),len(scores))

boxes = boxes.round().astype(np.int32).clip(min=0, max=1023)



fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for j,box in enumerate(boxes):

    cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 3)

    score=str(float("{:.3f}".format(scores[j])))

    

#     cv2.putText(

#               sample,

#               score,

#               org=(int(box[0]), int(box[1] )), # bottom left

#               fontFace=cv2.FONT_HERSHEY_PLAIN,

#               fontScale=3,

#               color=(1,1, 1),

#               thickness=2

#             )

ax.set_axis_off()

ax.imshow(sample);


    

validation_image_precisions = []

iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

# model.eval()

# model.to(device)



results = []

ts = 0.147



for images, image_ids in data_loader:    

    predictions = make_ensemble_predictions(images)

    for i, image in enumerate(images):

#         boxes, scores, labels = run_wbf(predictions, image_index=i, iou_thr=0.35, skip_box_thr=0.35, score_threshold = 0.147, weights=None,conf_type='avg')# res50, res152(20) 

        boxes, scores, labels = run_wbf(predictions, image_index=i, iou_thr=0.35, skip_box_thr=0.35, score_threshold = 0.35, weights=None,conf_type='avg')

        boxes = boxes.round().astype(np.int32).clip(min=0, max=1023)

        image_id = image_ids[i]



        

        

#         plot the images

#         sample = images[i].permute(1,2,0).cpu().numpy()

#         fig, ax = plt.subplots(1, 1, figsize=(15,15))

#         indexes = np.where(scores > ts)[0]



#         for j,box in enumerate(boxes):

#             if j not  in indexes:

#                 c = (1,0,0)

#             else:

#                 c = (0,0,0)

                

#             cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), c, 3)

#             score=str(float("{:.3f}".format(scores[j])))



#             cv2.putText(

#               sample,

#               score,

#               org=(int(box[0]), int(box[1] )), # bottom left

#               fontFace=cv2.FONT_HERSHEY_PLAIN,

#               fontScale=3,

#               color=(1,1, 1),

#               thickness=2

#             )

#         ax.set_axis_off()

#         ax.imshow(sample)

        





        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        

        result = {

            'image_id': image_id,

            'PredictionString': format_prediction_string(boxes, scores)

        }

        results.append(result)

    
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df.head(10)