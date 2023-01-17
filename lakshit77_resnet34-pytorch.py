import sys

pt_models = "../input/pretrained-models/pretrained-models.pytorch-master/"

pt_models

sys.path.insert(0,pt_models)
import pretrainedmodels
import os

import numpy as np 

import pandas as pd

import ast

import torch

from tqdm import tqdm

import torch.nn as nn

import joblib

import albumentations

from PIL import Image

import glob

from torch.nn import functional as F
IMG_HEIGHT = 137

IMG_WIDTH = 236

TEST_BATCH_SIZE = 16

MODEL_MEAN = (0.485, 0.456, 0.406)

MODEL_STD = (0.229, 0.224, 0.225)

DEVICE = "cuda"
class Resnet34(nn.Module):

    def __init__(self, pretrained):

        super(Resnet34, self).__init__()

        if pretrained is True:

            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = "imagenet")

        else:

            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = None)



        self.l0 = nn.Linear(512,168)

        self.l1 = nn.Linear(512,11)

        self.l2 = nn.Linear(512,7)



    def forward(self, x):

        bs, _,_,_ = x.shape

        x = self.model.features(x)

        x = F.adaptive_avg_pool2d(x,1).reshape(bs, -1)

        l0 = self.l0(x)

        l1 = self.l1(x)

        l2 = self.l2(x)

        return l0, l1, l2
class BengaliDatasetTrain:

    def __init__(self, df, img_height, img_width, mean, std):

        

        self.image_ids = df.image_id.values

        self.img_arr = df.iloc[:, 1:].values

        

        self.aug = albumentations.Compose([

            albumentations.Resize(img_height, img_width, always_apply=True),

            albumentations.Normalize(mean, std, always_apply = True)

        ])

    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self, item):

        image = self.img_arr[item, :]

        img_id = self.image_ids[item]

        image = image.reshape(137,236).astype(float)

        image = Image.fromarray(image).convert("RGB")

        image = self.aug(image = np.array(image))['image']

        image = np.transpose(image, (2,0,1)).astype(np.float32)

        return{

            'image': torch.tensor(image, dtype=torch.float),

            "image_id":img_id

        }

        
model = Resnet34(pretrained=False)

model.load_state_dict(torch.load("../input/bengali-train-model/resnet34_4_weighted.pth", map_location=torch.device('cpu')))

model.eval()
predictions = []

for file_idx in range(4):

    df = pd.read_parquet(f"../input/bengaliai-cv19/test_image_data_{file_idx}.parquet")

    dataset = BengaliDatasetTrain(df = df, 

                                  img_height=IMG_HEIGHT,

                                  img_width=IMG_WIDTH,

                                  mean = MODEL_MEAN,

                                  std=MODEL_STD)

    data_loader = torch.utils.data.DataLoader(dataset,

                                              batch_size=TEST_BATCH_SIZE,

                                              shuffle=False,

                                              num_workers=4)

    with torch.no_grad():

        for bi, d in enumerate(data_loader):

            image = d['image']

            img_id = d['image_id']

            g, v, c = model(image)

            g = np.argmax(g, axis=1)

            v = np.argmax(v, axis=1)

            c = np.argmax(c, axis=1)

            for ii, imid in enumerate(img_id):

                predictions.append((f"{imid}_grapheme_root", int(g[ii])))

                predictions.append((f"{imid}_vowel_diacritic", int(v[ii])))

                predictions.append((f"{imid}_consonant_diacritic", int(c[ii])))
sub = pd.DataFrame(predictions, columns = ['row_id', 'target'])
sub.to_csv('submission.csv', index=False)