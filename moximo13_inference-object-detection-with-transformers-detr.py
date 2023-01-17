import os

import sys

from pathlib import Path



import albumentations as A

import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm





sys.path.append('../input/detrmodels/facebookresearch_detr_master/')
# copy pretrained weights to the folder PyTorch will search by default

Path('/root/.cache/torch/hub/').mkdir(exist_ok=True, parents=True)

Path('/root/.cache/torch/hub/checkpoints/').mkdir(exist_ok=True, parents=True)



detr_path = '/root/.cache/torch/hub/checkpoints/detr-r50-e632da11.pth'

resnet50_pretrained = '/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'

detr_hub = '/root/.cache/torch/hub/facebookresearch_detr_master'



!cp ../input/detrmodels/detr-r50-e632da11.pth {detr_path}

!cp ../input/detrmodels/resnet50-19c8e357.pth {resnet50_pretrained}

!cp -R ../input/detrmodels/facebookresearch_detr_master {detr_hub}
DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TEST = f'{DIR_INPUT}/test'



DIR_WEIGHTS = '../input/detrglobalwheatbestmodels/'

WEIGHTS_FILE = f'{DIR_WEIGHTS}/detr_best_0.pth'



batch_size = 5
def collate_fn(batch):

    return tuple(zip(*batch))





def get_valid_transforms():

    return A.Compose([A.Resize(height=512, width=512, p=1.0),

                      ToTensorV2(p=1.0)], 

                      p=1.0, 

                    )





class WheatDataset(Dataset):

    def __init__(self,image_ids,dataframe,transforms=None):

        self.image_ids = image_ids

        self.df = dataframe

        self.transforms = transforms

        

        

    def __len__(self) -> int:

        return self.image_ids.shape[0]

    

    def __getitem__(self,index):

        image_id = self.image_ids[index]

        

        image = cv2.imread(f'{DIR_TEST}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        

        if self.transforms:

            sample = {

                'image': image,

            }

            sample = self.transforms(**sample)

            image = sample['image']

    

        

        return image, image_id
class DETRModel(nn.Module):

    def __init__(self, num_classes, num_queries, model_name='detr_resnet50'):

        super(DETRModel, self).__init__()

        self.num_classes = num_classes

        self.num_queries = num_queries



        self.model = torch.hub.load('facebookresearch/detr', model_name, pretrained=True)

        self.in_features = self.model.class_embed.in_features



        self.model.class_embed = nn.Linear(in_features=self.in_features,

                                           out_features=self.num_classes)

        self.model.num_queries = self.num_queries



    def forward(self, images):

        return self.model(images)
test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')

test_df.shape
test_df.head()
image_ids = test_df.image_id.unique(); image_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2

num_queries = 100

model = DETRModel(num_classes=num_classes, num_queries=num_queries, model_name='detr_resnet50')

model = model.to(device)

model.load_state_dict(torch.load(WEIGHTS_FILE))

model.eval()
clf_gt = []



test_dataset = WheatDataset(

    image_ids=image_ids,

    dataframe=test_df,

    transforms=get_valid_transforms()

)



data_test_loader = DataLoader(

        test_dataset, batch_size=batch_size, 

        collate_fn=collate_fn,

)
rows = 2

columns = 5

confidence_thrsh = 0.5

fig, ax = plt.subplots(rows, columns, figsize=(15, 8))

for i, (images, image_ids) in enumerate(tqdm(data_test_loader)):

    with torch.no_grad():

        images = list(image.to(device, dtype=torch.float) for image in images)

        outputs = model(images)

    outputs = [{k: v.to('cpu') for k, v in outputs.items()}][0]

    

    row = i % rows

    for j, (image_name, bboxes, logits) in enumerate(zip(image_ids, outputs['pred_boxes'], outputs['pred_logits'])):

        column = j % columns



        oboxes = bboxes.detach().cpu().numpy()

        oboxes = np.array([

            np.array(box).astype(np.int32) 

            for box in A.augmentations.bbox_utils.denormalize_bboxes(oboxes,512,512)

        ])

        prob   = logits.softmax(1).detach().cpu().numpy()[:, 0]

        # scale boxes 

        oboxes = (oboxes*2).astype(np.int32).clip(min=0, max=1023)

        

        # Comment this out of you dont want to plot bboxes #

        sample = cv2.imread(f'{DIR_TEST}/{image_name}.jpg', cv2.IMREAD_COLOR)

        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

        for box,p in zip(oboxes,prob):

            if p > confidence_thrsh:

                color = (0,0,220)

                cv2.rectangle(sample,

                      (box[0], box[1]),

                      (box[2]+box[0], box[3]+box[1]),

                      color, 3)

    

        ax[row, column].set_axis_off()

        ax[row, column].imshow(sample)

        # ================================================= #

        

        clf_gt.append({

                'image_id': image_name,

                'PredictionString': ' '.join(

                    str(round(confidence,4)) 

                    + ' '

                    + ' '.join(str(int(round(float(x)))) for x in box) 

                    for box, confidence in zip(oboxes, prob)

                    if confidence > confidence_thrsh

                )

                ,

            })
submission_df = pd.DataFrame(clf_gt)

submission_df['PredictionString'] = submission_df['PredictionString'].fillna('')

submission_df.to_csv('submission.csv', index=False)
submission_df