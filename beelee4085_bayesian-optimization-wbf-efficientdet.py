!pip install --no-deps '../input/timm-package/timm-0.1.26-py3-none-any.whl' > /dev/null

!pip install --no-deps '../input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl' > /dev/null
import sys

sys.path.insert(0, "../input/timm-efficientdet-pytorch")

sys.path.insert(0, "../input/omegaconf")

sys.path.insert(0, "../input/weightedboxesfusion")



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

from effdet import get_efficientdet_config, EfficientDet, DetBenchEval

from effdet.efficientdet import HeadNet

from sklearn.model_selection import StratifiedKFold

from skopt import gp_minimize, forest_minimize

from skopt.utils import use_named_args

from skopt.plots import plot_objective, plot_evaluations, plot_convergence, plot_regret

from skopt.space import Categorical, Integer, Real



SEED = 42



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
#USE_OPTIMIZE = len(glob(f'../input/global-wheat-detection/test/*.jpg')) == 10

USE_OPTIMIZE = True # used for fast inference in submission

USE_TTA = True
def load_net(checkpoint_path):

    config = get_efficientdet_config('tf_efficientdet_d5')

    net = EfficientDet(config, pretrained_backbone=False)



    config.num_classes = 1

    config.image_size = 512

    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))



    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint["model_state_dict"])



    del checkpoint

    gc.collect()



    net = DetBenchEval(net, config)

    net.eval();

    return net.cuda()



if USE_OPTIMIZE:

    models = [

        load_net('../input/effdet-fold0/best-checkpoint-023epoch.bin'),

        load_net('../input/effdet-fold1/best-checkpoint-017epoch.bin')]
best_final_score = 0.7202

best_iou_thr = 0.460

best_skip_box_thr = 0.394
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
DATA_ROOT_PATH = '../input/global-wheat-detection/test'



class TestDatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        #image /= 255.0

        if self.transforms:

            sample = {'image': image}

            sample = self.transforms(**sample)

            image = sample['image']

        return image, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]



def get_valid_transforms():

    return A.Compose(

        [

            A.Resize(height=512, width=512, p=1.0),

            #ToTensorV2(p=1.0),

        ], 

        p=1.0, 

    )



dataset = TestDatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),

    transforms=get_valid_transforms()

)



def collate_fn(batch):

    return tuple(zip(*batch))



data_loader = DataLoader(

    dataset,

    batch_size=1,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)



def make_predictions(

    images, 

    score_threshold=0.25,

):

    predictions = []

    for fold_number, net in enumerate(models):

        with torch.no_grad():

            det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())

            result = []

            for i in range(images.shape[0]):

                boxes = det[i].detach().cpu().numpy()[:,:4]    

                scores = det[i].detach().cpu().numpy()[:,4]

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]

                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]

                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

                result.append({

                    'boxes': boxes[indexes],

                    'scores': scores[indexes],

                })

            predictions.append(result)

    return predictions





def run_wbf(predictions, image_index, image_size=512, iou_thr=best_iou_thr, skip_box_thr=best_skip_box_thr, weights=None):

    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]

    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels





def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

def to_tensor(images):

    tmp = []

    for img in images:

        img = img.astype(np.float32)

        img /= 255.0

        img = torch.tensor(img, dtype=torch.float32)

        tmp.append(img.permute(2,0,1))

    return torch.stack(tmp)
def run_wbf_tta(predictions, image_size=512, iou_thr=best_iou_thr, skip_box_thr=best_skip_box_thr, weights=None):

    boxes_1 = []

    scores_1 = []

    labels_1 = []

    for i in range(2):

        bb = (predictions['boxes'][i]/(image_size-1)).tolist()

        ss = predictions['scores'][i].tolist()

        ls = predictions['labels'][i].tolist()

        boxes_1.append(bb)

        scores_1.append(ss)

        labels_1.append(ls)



    boxes_1, scores_1, labels_1 = weighted_boxes_fusion(boxes_1, scores_1, labels_1, 

                                                        weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    

    boxes_2 = []

    scores_2 = []

    labels_2 = []

    for i in range(2,4):

        bb = (predictions['boxes'][i]/(image_size-1)).tolist()

        ss = predictions['scores'][i].tolist()

        ls = predictions['labels'][i].tolist()

        boxes_2.append(bb)

        scores_2.append(ss)

        labels_2.append(ls)

    

    boxes_2, scores_2, labels_2 = weighted_boxes_fusion(boxes_2, scores_2, labels_2, 

                                                        weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    

    boxes = [boxes_1.tolist(), boxes_2.tolist()]

    scores = [scores_1.tolist(), scores_2.tolist()]

    labels = [labels_1.tolist(), labels_2.tolist()]

    

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, 

                                                        weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
results = []



for images, image_ids in data_loader:

    

    if USE_TTA:

        image = images[0]

        

        predictions_tta = {

            "boxes": [],

            "scores": [],

            "labels": []

        }

        for index in range(4):

            roated = TTAImage(image, index)

            roated = to_tensor([roated]).cuda()

            predictions = make_predictions(roated)

            boxes, scores, labels = run_wbf(predictions, image_index=0)



            for _ in range(3-index):

                

                boxes = rotBoxes90(boxes, 512, 512)

            

            if index == 3:

                

                boxes = boxes.astype(np.int32)

                

            predictions_tta["boxes"].append(boxes)

            predictions_tta["scores"].append(scores)

            predictions_tta["labels"].append(labels)

        

        boxes, scores, labels = run_wbf_tta(predictions_tta)

        boxes = (boxes*2).astype(np.int32).clip(min=0, max=1023)

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        result = {

                'image_id': image_ids[0],

                'PredictionString': format_prediction_string(boxes, scores)

            }

        results.append(result)

        

    else:

        images = to_tensor(images).cuda()

        predictions = make_predictions(images)

        for i, image in enumerate(images):

            boxes, scores, labels = run_wbf(predictions, image_index=i)

            boxes = (boxes*2).astype(np.int32).clip(min=0, max=1023)

            image_id = image_ids[i]



            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]



            result = {

                'image_id': image_id,

                'PredictionString': format_prediction_string(boxes, scores)

            }

            results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)
test_df.head()