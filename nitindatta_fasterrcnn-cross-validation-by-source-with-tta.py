!cp -rf /kaggle/input/wheatdetection-resnest-develop-branch-july9/wheatdetection/wheatdetection /kaggle/working/

CODE_PATH = '/kaggle/working/wheatdetection/'

%cd {CODE_PATH}
import sys

import random

from albumentations.pytorch.transforms import ToTensorV2

sys.path.insert(0, "/kaggle/input/wheatdetection-resnest-develop-branch-july9/wheatdetection")
import torch



import albumentations as A



BEST_PATH = "/kaggle/input/5fold-68-clear/F1_68_nofinetune_clear_best.bin"

VALID_FOLD = 1 #Change your fold here



# print to see best weights information

ckp = torch.load(BEST_PATH)

print(ckp.keys())

print(ckp['epoch'], ckp['best_valid_loss'])
# Assign values to hyperparameters

USE_NMS = False

SCORE_THRESHOLD = 0.68

NMS_IOU_THRESHOLD = 0.5

IMG_SIZE = 1024



WBF_IOU, WBF_SKIP_BOX = 0.44, 0.38

import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from itertools import product



marking = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')





bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']):

    marking[column] = bboxs[:, i]

marking.drop(columns=['bbox'], inplace=True)

marking['area'] = marking['w'] * marking['h']



# Filtering boxes

marking = marking[marking['area'] < 154200.0]

error_bbox = [100648.0, 145360.0, 149744.0, 119790.0, 106743.0]

marking = marking[~marking['area'].isin(error_bbox)]

marking = marking[marking['h']>16.0]

marking = marking[marking['w']>16.0]



#Stratified 5 fold split

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



df_folds = marking[['image_id']].copy()

df_folds.loc[:, 'bbox_count'] = 1

df_folds = df_folds.groupby('image_id').count()

df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']

df_folds.loc[:, 'stratify_group'] = np.char.add(

    df_folds['source'].values.astype(str),

    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)

)

df_folds.loc[:, 'fold'] = 0



for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):

    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number



train_ids = df_folds[df_folds['fold'] != VALID_FOLD].index.values

valid_ids = df_folds[df_folds['fold'] == VALID_FOLD].index.values



val_df = marking[marking['image_id'].isin(valid_ids)]

train_df = marking[marking['image_id'].isin(train_ids)]
%%writefile ./modeling/wheat_detector.py



import torch

from torch import nn

from layers import FasterRCNN

from layers.backbone_utils import resnest_fpn_backbone



class WheatDetector(nn.Module):

    def __init__(self, cfg, **kwargs):

        super(WheatDetector, self).__init__()

        self.backbone = resnest_fpn_backbone(pretrained=False) #change here

        self.base = FasterRCNN(self.backbone, num_classes=cfg.MODEL.NUM_CLASSES, **kwargs)



    def forward(self, images, targets=None):

        return self.base(images, targets)
import matplotlib.pyplot as plt

import cv2



import os

import warnings

import torch

import numpy as np

from tqdm import tqdm

import pandas as pd

from itertools import product

import sys

sys.path.insert(0, "./external/wbf")

import ensemble_boxes

warnings.filterwarnings("ignore")

from torchvision.transforms import functional_tensor as TF



class BaseWheatTTA:

    """ author: @shonenkov """

    image_size = IMG_SIZE



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

        boxes[:, [1, 3]] = self.image_size - boxes[:, [3, 1]]

        return boxes



class TTAVerticalFlip(BaseWheatTTA):

    """ author: @shonenkov """



    def augment(self, image):

        return image.flip(2)



    def batch_augment(self, images):

        return images.flip(3)



    def deaugment_boxes(self, boxes):

        boxes[:, [0, 2]] = self.image_size - boxes[:, [2, 0]]

        return boxes





class TTARotate90(BaseWheatTTA):

    """ author: @shonenkov """



    def augment(self, image):

        return torch.rot90(image, 1, (1, 2))



    def batch_augment(self, images):

        return torch.rot90(images, 1, (2, 3))



    def deaugment_boxes(self, boxes):

        res_boxes = boxes.copy()

        res_boxes[:, [0, 2]] = self.image_size - boxes[:, [3, 1]]

        res_boxes[:, [1, 3]] = boxes[:, [0, 2]]

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

        result_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)

        result_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)

        result_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)

        result_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)

        return result_boxes



    def deaugment_boxes(self, boxes):

        for transform in self.transforms[::-1]:

            boxes = transform.deaugment_boxes(boxes)

        return self.prepare_boxes(boxes)

    

tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None],

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
from evaluate.evaluate import evaluate



class Tester:

    def __init__(self, model, device, cfg, test_loader):

        self.config = cfg

        self.test_loader = test_loader



        self.base_dir = f'{self.config.OUTPUT_DIR}'

        if not os.path.exists(self.base_dir):

            os.makedirs(self.base_dir)



        self.log_path = f'{self.base_dir}/log.txt'

        self.score_threshold = SCORE_THRESHOLD

        self.iou_threshold = NMS_IOU_THRESHOLD

        self.use_nms = USE_NMS

        

        self.model = model

        self.model.eval()



        self.device = device

        self.model.to(self.device)



    def make_tta_predictions(self,images, score_threshold=0.3):

        self.model.eval()

        torch.cuda.empty_cache()

        with torch.no_grad():

            images = torch.stack(images).float().cuda()

            predictions = []

            for tta_transform in tta_transforms:

                result = []

                outputs = self.model(tta_transform.batch_augment(images.clone()))

            for i in range(images.shape[0]):

                boxes = outputs[i]['boxes'].data.cpu().numpy()  

                scores = outputs[i]['scores'].data.cpu().numpy()

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]

                boxes = tta_transform.deaugment_boxes(boxes.copy())

                result.append({

                    'boxes': boxes,

                    'scores': scores[indexes],

                })

            predictions.append(result)

        return predictions



    def load(self, path):

        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])



    def log(self, message):

        if self.config.VERBOSE:

            print(message)

        with open(self.log_path, 'a+') as logger:

            logger.write(f'{message}\n')

from config import cfg



#IMPORTANT

cfg.DATASETS.VALID_FOLD = VALID_FOLD



cfg['OUTPUT_DIR'] = "/kaggle/working/"

cfg['DATASETS']['ROOT_DIR'] = "/kaggle/input/global-wheat-detection"

cfg['TEST']['WEIGHT'] = BEST_PATH

print(cfg)
from data import make_data_loader

train_loader, val_loader = make_data_loader(cfg, is_train=False)
import os

import sys



from os import mkdir

sys.path.append('.')

from config import cfg

from data import make_test_data_loader

from modeling import build_model

from utils.logger import setup_logger



# start here!!

if True:

    cfg.freeze()



    output_dir = cfg.OUTPUT_DIR

    if output_dir and not os.path.exists(output_dir):

        print('creating ',cfg.OUTPUT_DIR)

        mkdir(output_dir)



    model = build_model(cfg)

    device = cfg.MODEL.DEVICE

    checkpoint = torch.load(cfg.TEST.WEIGHT)



    tester = Tester(model=model, device=device, cfg=cfg, test_loader=val_loader)

    tester.load(cfg['TEST']['WEIGHT'])

    print('*** success in loading weights! ***')

    

#     predictions = tester.test()
def get_valid_transforms():

    return A.Compose(

        [

            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),

            

            ToTensorV2(p=1.0),

        ], 

        p=1.0, 

        bbox_params=A.BboxParams(

            format='pascal_voc',

            min_area=0, 

            min_visibility=0,

            label_fields=['labels']

        )

    )

def pascal2coco(boxes):

    boxes = boxes.reshape(-1, 4)

    boxes[:,2] = (boxes[:,2]-boxes[:,0])

    boxes[:,3] = (boxes[:,3]-boxes[:,1])

    return boxes



import pandas as pd

import numpy as np

import numba

import re

import cv2

import ast

import matplotlib.pyplot as plt



from numba import jit

from typing import List, Union, Tuple



from collections import namedtuple

from typing import List, Union



Box = namedtuple('Box', 'xmin ymin xmax ymax')





def calculate_iou(gt: List[Union[int, float]],

                  pred: List[Union[int, float]],

                  form: str = 'pascal_voc') -> float:

    """Calculates the IoU.

    

    Args:

        gt: List[Union[int, float]] coordinates of the ground-truth box

        pred: List[Union[int, float]] coordinates of the prdected box

        form: str gt/pred coordinates format

            - pascal_voc: [xmin, ymin, xmax, ymax]

            - coco: [xmin, ymin, w, h]

    Returns:

        IoU: float Intersection over union (0.0 <= iou <= 1.0)

    """

    if form == 'coco':

        bgt = Box(gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3])

        bpr = Box(pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3])

    else:

        bgt = Box(gt[0], gt[1], gt[2], gt[3])

        bpr = Box(pred[0], pred[1], pred[2], pred[3])

        



    overlap_area = 0.0

    union_area = 0.0



    # Calculate overlap area

    dx = min(bgt.xmax, bpr.xmax) - max(bgt.xmin, bpr.xmin)

    dy = min(bgt.ymax, bpr.ymax) - max(bgt.ymin, bpr.ymin)



    if (dx > 0) and (dy > 0):

        overlap_area = dx * dy



    # Calculate union area

    union_area = (

            (bgt.xmax - bgt.xmin) * (bgt.ymax - bgt.ymin) +

            (bpr.xmax - bpr.xmin) * (bpr.ymax - bpr.ymin) -

            overlap_area

    )



    return overlap_area / union_area



def collate_batch(batch):

    return tuple(zip(*batch))



def find_best_match(gts, predd, threshold=0.5, form='pascal_voc'):

    """Returns the index of the 'best match' between the

    ground-truth boxes and the prediction. The 'best match'

    is the highest IoU. (0.0 IoUs are ignored).

    

    Args:

        gts: Coordinates of the available ground-truth boxes

        pred: Coordinates of the predicted box

        threshold: Threshold

        form: Format of the coordinates

        

    Return:

        Index of the best match GT box (-1 if no match above threshold)

    """

    best_match_iou = -np.inf

    best_match_idx = -1

    

    for gt_idx, ggt in enumerate(gts):

        iou = calculate_iou(ggt, predd, form=form)

        

        if iou < threshold:

            continue

        

        if iou > best_match_iou:

            best_match_iou = iou

            best_match_idx = gt_idx



    return best_match_idx





def calculate_precision(preds_sorted, gt_boxes, threshold=0.5, form='coco'):

    """Calculates precision per at one threshold.

    

    Args:

        preds_sorted: 

    """

    tp = 0

    fp = 0

    fn = 0



    fn_boxes = []



    for pred_idx, pred in enumerate(preds_sorted):

        best_match_gt_idx = find_best_match(gt_boxes, pred, threshold=threshold, form='coco')



        if best_match_gt_idx >= 0:

            # True positive: The predicted box matches a gt box with an IoU above the threshold.

            tp += 1



            # Remove the matched GT box

            gt_boxes = np.delete(gt_boxes, best_match_gt_idx, axis=0)



        else:

            # No match

            # False positive: indicates a predicted box had no associated gt box.

            fn += 1

            fn_boxes.append(pred)



    # False negative: indicates a gt box had no associated predicted box.

    fp = len(gt_boxes)

    precision = tp / (tp + fp + fn)

    return precision, fn_boxes, gt_boxes





def calculate_image_precision(preds_sorted, gt_boxes, thresholds=(0.5), form='coco', debug=False):

    

    n_threshold = len(thresholds)

    image_precision = 0.0

    

    for threshold in thresholds:

        precision_at_threshold, _, _ = calculate_precision(preds_sorted,

                                                           gt_boxes,

                                                           threshold=threshold,

                                                           form=form

                                                          )

        if debug:

            print("@{0:.2f} = {1:.4f}".format(threshold, precision_at_threshold))



        image_precision += precision_at_threshold / n_threshold

    

    return image_precision



import sys

sys.path.insert(0, "/kaggle/input/weightedboxesfusion")

from ensemble_boxes import *

    

def run_wbf(predictions, image_index, image_size=IMG_SIZE, iou_thr=WBF_IOU, skip_box_thr=WBF_SKIP_BOX, weights=None):

    boxes = [(prediction[image_index]['boxes'] / (image_size - 1)).tolist() for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in

              predictions]

    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels,

                                                                                    weights=None, iou_thr=iou_thr,

                                                                                    skip_box_thr=skip_box_thr)

    boxes = boxes * (image_size - 1)

    return boxes, scores, labels

from torch.utils.data import Dataset

class train_wheat(Dataset):



    def __init__(self, root, marking, image_ids, transforms=None, test=False):

        super().__init__()

        self.root = root

        self.image_ids = image_ids

        self.marking = marking

        self.transforms = transforms

        self.test = test



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        p_ratio = random.random()

        if self.test or p_ratio > 0.7:

            image, boxes = self.load_image_and_boxes(index)

        else:

            if p_ratio < 0.4:

                image, boxes = self.load_mosaic_image_and_boxes(index)

            elif p_ratio < 0.55:

                image, boxes = self.load_image_and_bboxes_with_cutmix(index)

            else:

                image, boxes = self.load_mixup_image_and_boxes(index)



        # there is only one class

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)



        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['image_id'] = torch.tensor([index])



        if self.transforms:

            for i in range(10):

                sample = self.transforms(**{

                    'image': image,

                    'bboxes': target['boxes'],

                    'labels': labels

                })

                if len(sample['bboxes']) > 0:

                    image = sample['image']

                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

                    # target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning

                    break



        return image, target, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]



    def load_image_and_boxes(self, index):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{self.root}/train/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        records = self.marking[self.marking['image_id'] == image_id]

        boxes = records[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return image, boxes



    def load_mosaic_image_and_boxes(self, index, imsize=1024):

        """

        This implementation of mosaic author:  https://www.kaggle.com/nvnnghia

        Refactoring and adaptation: https://www.kaggle.com/shonenkov

        """

        w, h = imsize, imsize

        s = imsize // 2



        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y

        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]



        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)

        result_boxes = []



        for i, index in enumerate(indexes):

            image, boxes = self.load_image_and_boxes(index)

            if i == 0:

                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)

                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)

            elif i == 1:  # top right

                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc

                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h

            elif i == 2:  # bottom left

                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)

                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)

            elif i == 3:  # bottom right

                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)

                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            padw = x1a - x1b

            padh = y1a - y1b



            boxes[:, 0] += padw

            boxes[:, 1] += padh

            boxes[:, 2] += padw

            boxes[:, 3] += padh



            result_boxes.append(boxes)



        result_boxes = np.concatenate(result_boxes, 0)

        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])

        result_boxes = result_boxes.astype(np.int32)

        result_boxes = result_boxes[

            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]

        return result_image, result_boxes



    def load_image_and_bboxes_with_cutmix(self, index):

        image, bboxes = self.load_image_and_boxes(index)

        image_to_be_mixed, bboxes_to_be_mixed = self.load_image_and_boxes(

            random.randint(0, self.image_ids.shape[0] - 1))



        image_size = image.shape[0]

        cutoff_x1, cutoff_y1 = [int(random.uniform(image_size * 0.0, image_size * 0.49)) for _ in range(2)]

        cutoff_x2, cutoff_y2 = [int(random.uniform(image_size * 0.5, image_size * 1.0)) for _ in range(2)]



        image_cutmix = image.copy()

        image_cutmix[cutoff_y1:cutoff_y2, cutoff_x1:cutoff_x2] = image_to_be_mixed[cutoff_y1:cutoff_y2,

                                                                 cutoff_x1:cutoff_x2]



        # Begin preparing bboxes_cutmix.

        # Case 1. Bounding boxes not intersect with cut off patch.

        bboxes_not_intersect = bboxes[np.concatenate((np.where(bboxes[:, 0] > cutoff_x2),

                                                      np.where(bboxes[:, 2] < cutoff_x1),

                                                      np.where(bboxes[:, 1] > cutoff_y2),

                                                      np.where(bboxes[:, 3] < cutoff_y1)), axis=None)]



        # Case 2. Bounding boxes intersect with cut off patch.

        bboxes_intersect = bboxes.copy()



        top_intersect = np.where((bboxes[:, 0] < cutoff_x2) &

                                 (bboxes[:, 2] > cutoff_x1) &

                                 (bboxes[:, 1] < cutoff_y2) &

                                 (bboxes[:, 3] > cutoff_y2))

        right_intersect = np.where((bboxes[:, 0] < cutoff_x2) &

                                   (bboxes[:, 2] > cutoff_x2) &

                                   (bboxes[:, 1] < cutoff_y2) &

                                   (bboxes[:, 3] > cutoff_y1))

        bottom_intersect = np.where((bboxes[:, 0] < cutoff_x2) &

                                    (bboxes[:, 2] > cutoff_x1) &

                                    (bboxes[:, 1] < cutoff_y1) &

                                    (bboxes[:, 3] > cutoff_y1))

        left_intersect = np.where((bboxes[:, 0] < cutoff_x1) &

                                  (bboxes[:, 2] > cutoff_x1) &

                                  (bboxes[:, 1] < cutoff_y2) &

                                  (bboxes[:, 3] > cutoff_y1))



        # Remove redundant indices. e.g. a bbox which intersects in both right and top.

        right_intersect = np.setdiff1d(right_intersect, top_intersect)

        right_intersect = np.setdiff1d(right_intersect, bottom_intersect)

        right_intersect = np.setdiff1d(right_intersect, left_intersect)

        bottom_intersect = np.setdiff1d(bottom_intersect, top_intersect)

        bottom_intersect = np.setdiff1d(bottom_intersect, left_intersect)

        left_intersect = np.setdiff1d(left_intersect, top_intersect)



        bboxes_intersect[:, 1][top_intersect] = cutoff_y2

        bboxes_intersect[:, 0][right_intersect] = cutoff_x2

        bboxes_intersect[:, 3][bottom_intersect] = cutoff_y1

        bboxes_intersect[:, 2][left_intersect] = cutoff_x1



        bboxes_intersect[:, 1][top_intersect] = cutoff_y2

        bboxes_intersect[:, 0][right_intersect] = cutoff_x2

        bboxes_intersect[:, 3][bottom_intersect] = cutoff_y1

        bboxes_intersect[:, 2][left_intersect] = cutoff_x1



        bboxes_intersect = bboxes_intersect[np.concatenate((top_intersect,

                                                            right_intersect,

                                                            bottom_intersect,

                                                            left_intersect), axis=None)]



        # Case 3. Bounding boxes inside cut off patch.

        bboxes_to_be_mixed[:, [0, 2]] = bboxes_to_be_mixed[:, [0, 2]].clip(min=cutoff_x1, max=cutoff_x2)

        bboxes_to_be_mixed[:, [1, 3]] = bboxes_to_be_mixed[:, [1, 3]].clip(min=cutoff_y1, max=cutoff_y2)



        # Integrate all those three cases.

        bboxes_cutmix = np.vstack((bboxes_not_intersect, bboxes_intersect, bboxes_to_be_mixed)).astype(int)

        bboxes_cutmix = bboxes_cutmix[np.where((bboxes_cutmix[:, 2] - bboxes_cutmix[:, 0]) \

                                               * (bboxes_cutmix[:, 3] - bboxes_cutmix[:, 1]) > 500)]

        # End preparing bboxes_cutmix.



        return image_cutmix, bboxes_cutmix



    def load_mixup_image_and_boxes(self, index):

        image, boxes = self.load_image_and_boxes(index)

        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))

        return (image + r_image) / 2, np.vstack((boxes, r_boxes)).astype(np.int32)

    
from tqdm import tqdm_notebook as tqdm

from torch.utils.data import Dataset,DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]



import pandas as pd



log = []

thresh = 0.5



# get x sources

image_source = val_df[['source']].drop_duplicates().to_numpy()



# make df per source

for source in image_source:

    idx = 0

    precision05 = 0

    precisions = 0

    print("evaluating for:", source)

    source_df = val_df[val_df['source'].isin(source)]

    source_id = source_df['image_id'].drop_duplicates().to_numpy()



    validation_dataset = train_wheat(

        root = cfg.DATASETS.ROOT_DIR,

        image_ids=source_id,

        marking=source_df,

        transforms=get_valid_transforms(),

        test=True

    )

    

    val_loader = DataLoader(

        validation_dataset,

        batch_size=1,

        num_workers=2,

        shuffle=False,

        sampler=SequentialSampler(validation_dataset),

        pin_memory=False,

        collate_fn=collate_batch,

    )



    # load data

    for j, (images, target, image_ids) in tqdm(enumerate(val_loader)):  

        # predict

        # predict

        tester = Tester(model=model, device=device, cfg=cfg, test_loader=val_loader)

        predictions = tester.make_tta_predictions(images)

        ps = []

        p05 = []

        for i in range(len(images)):

            boxes_gt = target[i]["boxes"].numpy()



            boxes, scores, labels = run_wbf(predictions, image_index=i, iou_thr=WBF_IOU, skip_box_thr=0.44)

            boxes = boxes.astype(np.int32).clip(min=0, max=1024)



            # 2coco

            boxes = pascal2coco(boxes).astype("float")

            boxes_gt = pascal2coco(boxes_gt)



            preds_sorted_idx = np.argsort(scores)[::-1]

            boxes_sorted = boxes[preds_sorted_idx]



            precision, fn_boxes, fp_boxes = calculate_precision(boxes_sorted, boxes_gt, threshold=0.5, form='coco')

            p05.append(precision)



            image_precision = calculate_image_precision(boxes_sorted, boxes_gt,

                                                thresholds=iou_thresholds,

                                                form='coco', debug=False)

            ps.append(image_precision)



        precision05 += np.mean(p05)

        precisions += np.mean(ps)

        idx += 1



    print("mAP at threshold 0.5: {}".format(precision05/idx))

    print("mAP at threshold 0.5:0.75: {}".format(precisions/idx))

    

    map50 = precision05/idx

    map5075 = precisions/idx

    

    # evaluate mAP based on each source

    result = {"source": source, "watermark": "FRCNN", "map50": map50, "map5075": map5075, "num_images": len(source_id)}

    log.append(result)

    

df = pd.DataFrame(log)
df