# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

import time

import math

import gc

import cv2



import torch

import torch.nn as nn

from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader



import torchvision

import torchvision.transforms as T 

import torchvision.models as models



import matplotlib.pyplot as plt



def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



root_test = '/kaggle/input/bengaliai-cv19'

model_path = '/kaggle/input/trained-model/new_resnet34.pth'
class graphemeDataset(Dataset):

    def __init__(self, img_arrs, target_file = None):

        self.img_arrs = img_arrs

        self.target_file = target_file

        

        if target_file is None:

            self.transforms = T.Compose([T.ToPILImage(), T.CenterCrop(150), T.Resize((128,128)),T.ToTensor()])

        else:

            self.transforms = T.Compose([T.ToPILImage(),T.RandomAffine(90),T.CenterCrop(150), T.Resize((128,128)),T.ToTensor()])

            # add targets for training

            target_df = pd.read_csv(target_file)

            self.grapheme = target_df['grapheme_root'].values

            self.vowel = target_df['vowel_diacritic'].values

            self.consonant = target_df['consonant_diacritic'].values

            del target_df

            gc.collect()

               

    def __getitem__(self, idx):

        img_arr = 255 - self.img_arrs[idx] # flip black and white, so the default padding value (0) could match

        new_tensor = self.transforms(img_arr.reshape(137, 236, 1))

        

        if self.target_file is None:

            return new_tensor

        else:

            grapheme_tensor = torch.tensor(self.grapheme[idx], dtype=torch.long)

            vowel_tensor = torch.tensor(self.vowel[idx], dtype=torch.long)

            consonant_tensor = torch.tensor(self.consonant[idx], dtype=torch.long)

            return new_tensor, grapheme_tensor, vowel_tensor, consonant_tensor

    

    def __len__(self):

        return len(self.img_arrs)
class modified_resnet34(nn.Module):

    def __init__(self):

        super(modified_resnet34, self).__init__()

        

        resnet34 = models.resnet34()

        resnet34.conv1 =  nn.Conv2d(1, 64, kernel_size = 5, stride = 1, padding = 2, bias = False)

        layers = list(resnet34.children())[:-1] + [nn.Flatten()]

        self.features= nn.Sequential(*layers)

        

        self.grapheme_classifier = nn.Sequential(

            nn.Linear(512, 168),

            nn.LogSoftmax(dim=1)

        )



        self.vowel_classifier = nn.Sequential(

            nn.Linear(512, 11),

            nn.LogSoftmax(dim=1)

        )

        

        self.consonant_classifier = nn.Sequential(

            nn.Linear(512, 7),

            nn.LogSoftmax(dim=1)

        )

        

    def forward(self, x):

        x = self.features(x)

        c1 = self.grapheme_classifier(x)

        c2 = self.vowel_classifier(x)

        c3 = self.consonant_classifier(x)

        return c1, c2, c3
model = modified_resnet34()

model.load_state_dict(torch.load(model_path))

model.to(device)
row_id = [];

target = [];

for i in range(4):

    # load testing data

    start = time.time()

    img_df = pd.read_parquet(os.path.join(root_test,'test_image_data_' + str(i) + '.parquet'))

    print(timeSince(start))

    

    img_id = []

    img_id.extend(img_df.image_id.tolist())

    img_arrs = img_df.iloc[:,1:].values

    dataset = graphemeDataset(img_arrs)

    print(dataset.__len__())

    loader = DataLoader(dataset, batch_size = 128, shuffle = False)

    

    # make predictions

    grapheme_pred = []; vowel_pred = []; consonant_pred = []

    with torch.no_grad():

        for img_tensor in loader:

            img_tensor = img_tensor.to(device)

            c1, c2, c3 = model(img_tensor)

            grapheme_pred.extend(c1.argmax(1).cpu().tolist())

            vowel_pred.extend(c2.argmax(1).cpu().tolist())

            consonant_pred.extend(c3.argmax(1).cpu().tolist())

    

    # format the results

    for idx, g, v, c in zip(img_id, grapheme_pred, vowel_pred, consonant_pred):

        row_id.append(idx + '_grapheme_root')

        row_id.append(idx + '_vowel_diacritic')

        row_id.append(idx + '_consonant_diacritic')

        target.append(g)

        target.append(v)

        target.append(c)

        

    # clean up 

    del img_arrs, img_df

    gc.collect()

        

pred = pd.DataFrame({'row_id' : row_id,'target':target})

pred.head()   
pred.to_csv('submission.csv',index=False)