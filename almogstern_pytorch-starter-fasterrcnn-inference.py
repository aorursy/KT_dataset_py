!pip install --no-deps '../input/weightedboxesfusion' > /dev/null
import pandas as pd

import numpy as np

import cv2

import os

import re







from glob import glob

import gc

from PIL import Image



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler



from matplotlib import pyplot as plt

import ensemble_boxes



DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'
test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

test_df.shape
class DatasetRetriever(Dataset):



    def __init__(self, image_ids, transforms=None):

        super().__init__()

        self.image_ids = image_ids

        self.transforms = transforms



    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{DIR_TEST}/{image_id}.jpg', cv2.IMREAD_COLOR)

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

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(p=1.0),

        ], p=1.0)
# Albumentations

def get_test_transform():

    return A.Compose([

        # A.Resize(512, 512),

        ToTensorV2(p=1.0)

    ])

test_dataset = DatasetRetriever(

    image_ids=np.array([path.split('/')[-1][:-4] for path in glob(f'{DIR_TEST}/*.jpg')]),

    transforms=get_valid_transforms()

)





def collate_fn(batch):

    return tuple(zip(*batch))





test_data_loader = DataLoader(

    test_dataset,

    batch_size=2,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
def load_net(checkpoint_path):

    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    num_classes = 2  # 1 class (wheat) + background

    # get number of input features for the classifier

    in_features = net.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one

    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint)

    net = net.cuda()

    net.eval()



    del checkpoint

    gc.collect()

    return net



models = [

    load_net('../input/60-epochs/fasterrcnn_resnet50_fpn_60epoch.pth'),

    load_net('../input/new-weights/fasterrcnn_resnet50_fpn_almog.pth'),

    load_net('../input/100epoch/fasterrcnn_resnet50_fpn_100_lr_adam.pth'),

    load_net('../input/wheat-fasterrcnn-folds/fold0-best1.bin'),

    load_net('../input/wheat-fasterrcnn-folds/fold1-best1.bin'),

    load_net('../input/wheat-fasterrcnn-folds/fold2-best1.bin'),

    load_net('../input/wheat-fasterrcnn-folds/fold3-best1.bin'),

    load_net('../input/wheat-fasterrcnn-folds/fold4-best1.bin'),

]
from ensemble_boxes import *



device = torch.device('cuda:0')



def make_ensemble_predictions(images):

    images = list(image.to(device) for image in images)    

    result = []

    for net in models:

        outputs = net(images)

        result.append(outputs)

    return result



def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.55, skip_box_thr=0.7, weights=None):

    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]

    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]

    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    boxes = boxes*(image_size-1)

    return boxes, scores, labels
import matplotlib.pyplot as plt



for j, (images, image_ids) in enumerate(test_data_loader):

    if j > 0:

        break

predictions = make_ensemble_predictions(images)



i = 1

sample = images[i].permute(1,2,0).cpu().numpy()

boxes, scores, labels = run_wbf(predictions, image_index=i)

boxes = boxes.astype(np.int32).clip(min=0, max=511)



fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 2)

    

ax.set_axis_off()

ax.imshow(sample);
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
results = []



for images, image_ids in test_data_loader:

    predictions = make_ensemble_predictions(images)

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

results[0:2]
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.head()
test_df.to_csv('submission.csv', index=False)