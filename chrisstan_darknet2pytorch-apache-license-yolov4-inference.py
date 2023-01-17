!cp -r ../input/pytorchyolov4/tool .

#!cp ../input/yolov4weights/yolo-wheat_best.weights .
import sys

sys.path.insert(0, "../input/weightedboxesfusion")





from ensemble_boxes import *

import glob
import os

from ensemble_boxes import *

import torch

import random

import numpy as np

import pandas as pd

from glob import glob

from torch.utils.data import Dataset,DataLoader

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

import cv2

import gc

from tqdm import tqdm

from matplotlib import pyplot as plt



from sklearn.model_selection import StratifiedKFold

from skopt import gp_minimize, forest_minimize

from skopt.utils import use_named_args

from skopt.plots import plot_objective, plot_evaluations, plot_convergence, plot_regret

from skopt.space import Categorical, Integer, Real
from tool.utils import *

from tool.torch_utils import *

from tool.darknet2pytorch import Darknet

import argparse
cfgfile = '../input/yolov4weights/yolov4-obj.cfg'

weightfile = '../input/yolov4weights/yolo-wheat_best.weights'
import cv2

m = Darknet(cfgfile)



m.print_network()

m.load_weights(weightfile)

#print('Loading weights from %s... Done!' % (weightfile))
m.cuda()

num_classes = m.num_classes

class_names = load_class_names('../input/yolov4weights/wheat.names')
iou_threshold = 0.34

skip_threshold = 0.31
def get_valid_transforms():

    return A.Compose([

            A.Resize(height=704, width=704, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)
DATA_ROOT_PATH = '../input/global-wheat-detection/test'



class DatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg')

        image = cv2.resize(image, (704, 704))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
dataset = DatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),

    transforms=get_valid_transforms()

)



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=False,

    num_workers=1,

    drop_last=False,

    collate_fn=collate_fn

)
class BaseWheatTTA:

    """ author: @shonenkov """

    image_size = 704



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

    

    def augment(self, image):

        tmp = torch.rot90(image, 1, (1, 2))

        return torch.rot90(tmp, 1, (1, 2))



    def batch_augment(self, images):

        tmp = torch.rot90(images, 1, (2, 3))

        return torch.rot90(tmp, 1, (2, 3))

    

    def deaugment_boxes(self, boxes):

        tmp = TTARotate90().deaugment_boxes(boxes)

        return TTARotate90().deaugment_boxes(tmp)

    

class TTARotate270(BaseWheatTTA):

    

    def augment(self, image):

        tmp = TTARotate180().augment(image)

        return torch.rot90(tmp, 1, (1, 2))



    def batch_augment(self, images):

        tmp = TTARotate180().batch_augment(images)

        return torch.rot90(tmp, 1, (2, 3))

    

    def deaugment_boxes(self, boxes):

        tmp = TTARotate180().deaugment_boxes(boxes)

        return TTARotate90().deaugment_boxes(tmp)

    

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
def process_det(index, det, score_threshold=0.25):

    scores = det[index][:, 5].copy()

    det = det[index][:, :4].copy()

    bboxes = np.zeros((det.shape))

    bboxes[:, 0] = ((det[:, 0] - det[:, 2] / 2) * 704).astype(int)

    bboxes[:, 1] = ((det[:, 1] - det[:, 3] / 2) * 704).astype(int)

    bboxes[:, 2] = ((det[:, 0] + det[:, 2] / 2) * 704).astype(int)

    bboxes[:, 3] = ((det[:, 1] + det[:, 3] / 2) * 704).astype(int)

    bboxes = (bboxes).clip(min = 0, max = 703).astype(int)

    

    indexes = np.where(scores>score_threshold)

    bboxes = bboxes[indexes]

    scores = scores[indexes]

    return bboxes, scores
transform = TTACompose([

    TTARotate270(),

    #TTAVerticalFlip(),

])



fig, ax = plt.subplots(1, 3, figsize=(16, 6))



image, image_id = dataset[5]



numpy_image = image.permute(1,2,0).cpu().numpy().copy()



ax[0].imshow(numpy_image);

ax[0].set_title('original')



tta_image = transform.augment(image)

print(tta_image.shape)

tta_image_numpy = tta_image.permute(1,2,0).cpu().numpy().copy()



det = do_detect(m, tta_image.permute(1,2,0).cpu().numpy(), 0.001, 0.6)

det = np.array(det)

print(det.shape)

boxes, scores = process_det(0, det)





for box in boxes:

    cv2.rectangle(tta_image_numpy, (box[0], box[1]), (box[2],  box[3]), (0, 1, 0), 2)



ax[1].imshow(tta_image_numpy);

ax[1].set_title('tta')

    

boxes = transform.deaugment_boxes(boxes)

for box in boxes:

    cv2.rectangle(numpy_image, (box[0], box[1]), (box[2],  box[3]), (0, 1, 0), 2)

    

ax[2].imshow(numpy_image);

ax[2].set_title('deaugment predictions');
from itertools import product



tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], 

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), None],

                               [TTARotate180(), None],

                               [TTARotate270(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
def make_tta_predictions(images, score_threshold=0.25):

    with torch.no_grad():

        images = torch.stack(images).float().cuda()

        predictions = []

        #print('images.shape', images.shape)

        for tta_transform in tta_transforms:

            result = []

            input_img = tta_transform.batch_augment(images.clone()).permute(0,2,3,1).cpu().numpy()

            #print('input_img',input_img.shape)

            det = do_detect(m, input_img, 0.4, 0.6)

            #print([len(i) for i in det])

            det = [np.array(i)for i in det]

            #print(det)

            for i in range(images.shape[0]):

                boxes, scores = process_det(i, det)

                boxes = tta_transform.deaugment_boxes(boxes.copy())

                result.append({

                    'boxes': boxes,

                    'scores': scores,

                })

            predictions.append(result)

    return predictions



import ensemble_boxes 

def run_wbf(predictions, image_index, image_size=704, iou_thr=0.5, skip_box_thr=0.1, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist() for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
import matplotlib.pyplot as plt

for j, (images, image_ids) in enumerate(data_loader):

    break



predictions = make_tta_predictions(images)



i = 0

sample = images[i].permute(1,2,0).cpu().numpy()

boxes, scores, labels = run_wbf(predictions, image_index=i)

boxes = boxes.round().astype(np.int32).clip(min=0, max=703)



fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 1)



ax.set_axis_off()

ax.imshow(sample);
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
results = []



for images, image_ids in data_loader:

    try:

        predictions = make_tta_predictions(images)

    

        for i, image in enumerate(images):

            

            boxes, scores, labels = run_wbf(predictions, image_index=i)

            boxes = (boxes*(1024 / 704)).round().astype(np.int32).clip(min=0, max=1023)

            image_id = image_ids[i]



            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            if len(boxes) > 0:

                result = {

                    'image_id': image_id,

                    'PredictionString': format_prediction_string(boxes, scores)

                }

                results.append(result)

    except:

        for i, image in enumerate(images):

            image_id = image_ids[i]

            result = {

                'image_id': image_id,

                'PredictionString': ''

            }

            results.append(result)

        
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df