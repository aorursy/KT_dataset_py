import sys

sys.path.insert(0,'../input/weightedboxesfusion')



import cv2

import gc

import numpy as np

import pandas as pd

import seaborn as sns

import torch

import torchvision



from torch.utils.data import DataLoader, Dataset

from tqdm.notebook import tqdm

from matplotlib import pyplot as plt

from glob import glob

from ensemble_boxes import *

from itertools import product



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchvision.models.detection.rpn import AnchorGenerator



%matplotlib inline

DIR = '../input/global-wheat-detection/'

DATA_ROOT_PATH = DIR + "test"



MODELS_DIR_PATH = '../input/fasterrcnn-resnet50-fpn-best' # //



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from albumentations import (Compose, Resize,Normalize )

from albumentations.pytorch.transforms import ToTensorV2



def get_test_transform():

    return Compose([

        Resize(height=512, width=512, p=1.0),

        ToTensorV2(p=1.0)  

    ])

    


def load_model(path):

    model_type = path.split('/')[-1].split('.')[0].split('_')[0]

    backbone = resnet_fpn_backbone(model_type, pretrained=False)



    model = FasterRCNN(backbone, 2)                    

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)



    model_dict = torch.load(path)

    model.load_state_dict(model_dict)



    del model_dict

    gc.collect()

    

    model.to(device)

    model.eval()

    return model
models = [load_model(f'{MODELS_DIR_PATH}/resnet50_0.pth'),

          load_model(f'{MODELS_DIR_PATH}/resnet50_1.pth'),

          load_model(f'{MODELS_DIR_PATH}/resnet50_2.pth'),

          load_model(f'{MODELS_DIR_PATH}/resnet101_3.pth'),

          load_model(f'{MODELS_DIR_PATH}/resnet152_4.pth')]
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



def collate_fn(batch):

    return tuple(zip(*batch))

dataset = DatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.jpg')]),

    transforms=get_test_transform()

)

data_loader = DataLoader(

    dataset,

    batch_size=1,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn)
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


tta_transforms = []

for tta_combination in product([TTAHorizontalFlip(), None], 

                               [TTAVerticalFlip(), None],

                               [TTARotate90(), None]):

    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))



def make_tta_predictions(net, images, score_threshold=0.4):

    with torch.no_grad():

        images = torch.stack(images).float().cuda()

        predictions = []

        for tta_transform in tta_transforms:

            result = []

            det = net(tta_transform.batch_augment(images.clone()))



            for i in range(images.shape[0]):

                boxes = det[i]['boxes'].detach().cpu().numpy()

                scores = det[i]['scores'].detach().cpu().numpy()

                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]

                # boxes[:, 2] = boxes[:, 2] + boxes[:, 0]

                # boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

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

    for net in models:

        outputs = net(images)

        result.append(outputs)

    return result



def run_wbf(predictions, image_index, image_size=512, iou_thr=0.55, skip_box_thr=0.7, weights=None,conf_type='avg'):

    boxes = [prediction[image_index]['boxes']/(image_size) for prediction in predictions]

    scores = [prediction[image_index]['scores'] for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr,conf_type=conf_type)

    boxes = boxes*(image_size)

    return boxes, scores, labels



def TTA_make_ensemble_predictions(images):

    images = list(image.to(device) for image in images)    

    result = []

    for net in models:

        predictions = make_tta_predictions(net, images)

        for i in range(len(images)):

            boxes, scores, labels = run_wbf(predictions, image_index=i, iou_thr=0.5, skip_box_thr=0.7)

            outputs = {'boxes':boxes,

                       'labels':labels,

                       'scores':scores}

            result.append([outputs])

    return result

def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)



def visualize_results(sample, boxes, scores, score_thresh=0.33,  show_error=False):

        sample = cv2.resize(sample,(1024,1024))

        indexes = np.where(scores > score_thresh)[0]

        if not show_error:

            boxes = boxes[indexes]



        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        colors = [(0, 0, 0),(1, 0, 0)]

        for index, (box, score) in enumerate(zip(boxes,scores)):

            

            if index not in indexes:

                c = colors[1]

            else:

                c = colors[0]

            cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]),  c, 2)

            cv2.putText(sample, f'{score:.3f}', (box[0]+20, box[1]+20),

                        cv2.FONT_HERSHEY_COMPLEX,  

                        1, (1, 1, 1), 2, cv2.LINE_AA) 

            

        # ax.set_axis_off()

        ax.imshow(sample);


THRESH = 0.35

results = []

for j, (images, image_ids) in tqdm(enumerate(data_loader), total=len(data_loader), desc='Prediction:'):

    predictions = TTA_make_ensemble_predictions(images)

    for i in range(len(images)):

        boxes, scores, labels = run_wbf(predictions, image_index=i, iou_thr=0.5, skip_box_thr=0.7, conf_type='max')

        boxes = (boxes*2).astype(np.int32).clip(min=0, max=1023)



#         visualize_results(images[i].permute(1,2,0).cpu().numpy(), boxes, scores, THRESH, show_error=True)



        indexes = np.where(scores > THRESH)[0]

        boxes = boxes[indexes]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]

        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]



        result = {

            'image_id': image_ids[i],

            'PredictionString': format_prediction_string(boxes, scores)

        }

        results.append(result)

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

test_df.head()
