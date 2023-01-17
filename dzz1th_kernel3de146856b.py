# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

from scipy import misc

from glob import glob



import albumentations as albu

from albumentations.pytorch import ToTensor



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, sampler

import cv2

from tqdm import tqdm



import os

        

sys.path.append('/kaggle/input/srnet-model-weight/')

        

from model import Srnet

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
BATCH_SIZE = 40

TESTPATH = '/kaggle/input/alaska2-image-steganalysis/Test/'

WEIGHTS =  '/kaggle/input/srnet-model-weight/SRNet_model_weights.pt'



df_sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')
def transform_test():

    transform = albu.Compose([

        albu.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),

        ToTensor()

    ])

    return transform
df_sub
class AlaskaDataset(Dataset):

    def __init__(self, df, data_folder, transform):

        self.df = df

        self.root = data_folder

        self._transform = transform

        

    def __getitem__(self, idx):

        image_id = self.df.Id.iloc[idx]

        print(image_id)

        image_path = os.path.join(self.root, image_id)

        img = cv2.imread(image_path)

        augment = self._transform(image=img)

        img = augment['image']

        img = torch.mean(img, axis=0,keepdim=True) #mean all channels because of SRNet input format

        return img

    

    def __len__(self):

        return len(self.df)
model = Srnet().cuda()

weights = torch.load(WEIGHTS)

model.load_state_dict(weights['model_state_dict'])
test_transform = transform_test()

test_dataset = AlaskaDataset(df_sub, TESTPATH, test_transform)

test_data = DataLoader(

    test_dataset,

    batch_size = BATCH_SIZE,

    num_workers = 2,

    shuffle = False

)
from tqdm import tqdm

outputs = []

model.eval() #turn model into eval mode before inference

with torch.no_grad():

    for inputs in tqdm(test_data):

        inputs = inputs.cuda()

        output = model(inputs)

        pred = output.data.cpu().numpy()

        pred = np.exp(pred[:,1]) / (np.exp(pred[:,0]) + np.exp(pred[:,1]))

        outputs.append(pred)

outputs = np.concatenate(outputs)

        
df_sub['Label']  = outputs

df_sub.to_csv('submissions.csv', index=None)

df_sub.head()