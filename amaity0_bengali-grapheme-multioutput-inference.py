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
import cv2

from fastai.vision import *

#from tqdm import tqdm_notebook as tqdm

import torch

import torch.nn as nn

import torchvision

PATH = '/kaggle/input/bengali-grapheme-second-try'
class MyRn50(nn.Module):

    def __init__(self):

        super(MyRn50, self).__init__()

        self.model_resnet = torchvision.models.resnet50()

        num_ftrs = self.model_resnet.fc.in_features

        self.model_resnet.fc = nn.Identity()

        self.fc_graph = nn.Linear(num_ftrs, 168)

        self.fc_vowel = nn.Linear(num_ftrs, 11)

        self.fc_conso = nn.Linear(num_ftrs, 7)



    def forward(self, x):

        x = self.model_resnet(x)

        out1 = self.fc_graph(x)

        out2 = self.fc_vowel(x)

        out3 = self.fc_conso(x)

        return out1, out2, out3



mymodel = MyRn50()

mymodel;
model=MyRn50()

weights = torch.load('/kaggle/input/bgraph-rn50-weights/model_wts1.pth',

                      map_location=torch.device('cpu'))

model.load_state_dict(weights['state_dict'])

model.eval();
nworkers = 2

PATH = '/kaggle/input/bengaliai-cv19/'

TEST = [PATH+'test_image_data_0.parquet',

        PATH+'test_image_data_1.parquet',

        PATH+'test_image_data_2.parquet',

        PATH+'test_image_data_3.parquet']



df_test = pd.read_csv(PATH+'test.csv')

df_test.describe()
HEIGHT = 137

WIDTH = 236

SIZE = 128



def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=16):

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))
class GraphemeDataset(Dataset):

    def __init__(self, fname):

        self.df = pd.read_parquet(fname)

        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        name = self.df.iloc[idx,0]

        #normalize each image by its max val

        img = (self.data[idx]*(255.0/self.data[idx].max())).astype(np.uint8)

        img = crop_resize(img)

        img = (img.astype(np.float32)/255.0 - imagenet_stats[0][0])/imagenet_stats[1][0]

        return img, name
row_id,target,bs = [],[],128



for fname in TEST:

    ds = GraphemeDataset(fname)

    dl = DataLoader(ds, batch_size=bs, num_workers=nworkers, shuffle=False)

    with torch.no_grad():

        for x,y in dl:

            x = x.unsqueeze(1)

            x = x.repeat(1,3,1,1)

            p1,p2,p3 = model(x)

            p1 = p1.argmax(-1).view(-1).cpu()

            p2 = p2.argmax(-1).view(-1).cpu()

            p3 = p3.argmax(-1).view(-1).cpu()

            for idx,name in enumerate(y):

                row_id += [f'{name}_grapheme_root',f'{name}_vowel_diacritic',

                           f'{name}_consonant_diacritic']

                target += [p1[idx].item(),p2[idx].item(),p3[idx].item()]

                

sub_df = pd.DataFrame({'row_id': row_id, 'target': target})

sub_df.to_csv('submission.csv', index=False)

sub_df