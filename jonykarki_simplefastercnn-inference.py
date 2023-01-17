import pandas as pd

import numpy as np

import cv2

import os, re

import gc



import torch

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



from torch.utils.data import DataLoader, Dataset



from matplotlib import pyplot as plt



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



INPUT_DIR = "/kaggle/input/global-wheat-detection"

MODEL_DIR = "/kaggle/input/gwdresnet50cv"
test_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

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

        

        if self.transforms:

            image = self.transforms(image=image)['image']



        return image, image_id
def get_test_transforms():

    return A.Compose([

        ToTensorV2(p=1.0)

    ])
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=2)



model.roi_heads
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model_1.pth")))

model.eval()



x = model.to(device)
def collate_fn(batch):

    return tuple(zip(*batch))



test_dataset = WheatDataset(test_df, os.path.join(INPUT_DIR, "test"), get_test_transforms())



test_data_loader = DataLoader(

    test_dataset,

    batch_size=4,

    shuffle=False,

    num_workers=4,

    drop_last=False,

    collate_fn=collate_fn

)
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))



    return " ".join(pred_strings)
detection_threshold = 0.5

results = []



for images, image_ids in test_data_loader:



    images = list(image.to(device) for image in images)

    outputs = model(images)



    for i, image in enumerate(images):



        boxes = outputs[i]['boxes'].data.cpu().numpy()

        scores = outputs[i]['scores'].data.cpu().numpy()

        

        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        scores = scores[scores >= detection_threshold]

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