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





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)

num_classes = 2 

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.__name__ = "fpn_resnet"

model.to(device)

# model = load_checkpoint(MODEL_PATH,model)

model.eval()
checkpoint = torch.load('../input/gwdmodels1/cp512.pt')

model.load_state_dict(checkpoint['model_state_dict'])

epoch = checkpoint['epoch']

loss_train = checkpoint['loss_train']

loss_val = checkpoint['loss_val']
def get_valid_transforms():

    return A.Compose([

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
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
dataset = DatasetRetriever(np.array([path.split('/')[-1][:-4] for path in glob.glob(f'{DATA_ROOT_PATH}/*.jpg')]),get_valid_transforms())



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=4,

    shuffle=False,

    num_workers=2,

    drop_last=False,

    collate_fn=collate_fn

)
batch = iter(data_loader)

imgs,img_id = next(batch)

images = list(img.to(device) for img in imgs)
results = []

model.eval()

model.to(device)

predictions = model(images)
def print_boxes(images,labels,threshold=None):

  plt.figure(figsize=[25, 20])

  for i, (img, label) in enumerate(zip(images, labels)):

    plt.subplot(4, 4, i+1)

    # boxes = label['boxes'].cpu().numpy().astype(np.int32)

    boxes = label['boxes']

    scores=None

    if 'scores' in label.keys():

      scores=label['scores']

    sample = img.permute(1,2,0).cpu().numpy()

    sample = sample*255.0

    for j in range(len(boxes)):

      box=boxes[j]

      active=True



      if scores is not None: 

        score=str(float("{:.3f}".format(scores[j])))

        if threshold is not None and float(score)<threshold:

          active=False

      else:

        score=''

      

      if active:

        cv2.rectangle(sample,

                    (box[0], box[1]),

                    (box[2], box[3]),

                    (255, 0, 0), 1)

        

        cv2.putText(

          sample,

          score,

          org=(int(box[0]), int(box[1] )), # bottom left

          fontFace=cv2.FONT_HERSHEY_PLAIN,

          fontScale=0.7,

          color=(255,0, 0),

          thickness=1

        )



      plt.imshow(sample.astype(np.uint8))    

    plt.axis('off')

  plt.show()

print_boxes(images,predictions)# Tests~!
class BaseWheatTTA:

    """ author: @shonenkov """

    image_size = 512



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
def make_tta_predictions(images, score_threshold=0.33):

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



def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size)).tolist() for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]

    boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size - 1)

    return boxes, scores, labels
from itertools import product



tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], 

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), TTARotate180(), TTARotate270(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))



validation_image_precisions = []

iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

model.eval()

model.to(device)



results = []

    

for images, image_ids in data_loader:

    predictions = make_tta_predictions(images)

    for i,img in enumerate(images):

        # sample = images[i].permute(1,2,0).cpu().numpy()

        boxes, scores, labels = run_wbf(predictions, image_index=i)

        boxes_s = (boxes*(2)).astype(np.int32).clip(min=0, max=1023)

        image_id = image_ids[i]

#         boxes = boxes.round().astype(np.int32).clip(min=0, max=IMAGE_SIZE-1)

        # images = list(img.to(device) for img in images)

        # outputs = model(images)

        # for output, target in zip(outputs, targets):

        #     preds = output['boxes'].cpu().detach().numpy()

        #     scores = output['scores'].cpu().detach().numpy()

#         gt_boxes = targets[i]['boxes'].detach().numpy()

#         preds_sorted_idx = np.argsort(scores)[::-1]

#         preds_sorted = boxes[preds_sorted_idx]



        boxes_s[:, 2] = boxes_s[:, 2] - boxes_s[:, 0]

        boxes_s[:, 3] = boxes_s[:, 3] - boxes_s[:, 1]

#         scores = img['scores']

        result = {

          'image_id': image_id,

          'PredictionString': format_prediction_string(boxes_s,scores)

        }

        results.append(result)  

        # # for idx, image in enumerate(images):

#         image_precision = calculate_image_precision(preds_sorted,

#                                                     gt_boxes,

#                                                     thresholds=iou_thresholds,

#                                                     form='pascal_voc')



#         validation_image_precisions.append(image_precision)



# print("Validation IOU: {0:.4f}".format(np.mean(validation_image_precisions)))
# results = []

# model.eval()

# model.to(device)

# for imgs, image_ids in data_loader:

#   images = list(img.to(device) for img in imgs)

#   predictions = model(images)

#   for i, pred in enumerate(predictions):

#       boxes = pred['boxes']

#       boxes=boxes.cpu().detach().numpy()

#       boxes = (boxes*(1024/224)).round().astype(np.int32).clip(min=0, max=1023)

#       image_id = image_ids[i]



#       boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

#       boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

#       scores = pred['scores']

#       result = {

#           'image_id': image_id,

#           'PredictionString': format_prediction_string(boxes,scores)

#       }

#       results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df.head(10)