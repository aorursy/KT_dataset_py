# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

from torchvision import models as mod

from torch import nn



norm_values=([0.485,0.456,0.406],[0.229,0.225,0.224])

iB = ImageDataBunch.from_folder(path='../input/flower_data/flower_data',

                                           size=224,bs=64,

                                ds_tfms=get_transforms(flip_vert=True,do_flip=True,max_rotate=90))

iB=iB.normalize(norm_values)

model = mod.resnext101_32x8d(pretrained=True)





for param in model.parameters():

    param.requires_grad=False

fc = nn.Sequential(        nn.Dropout(p=0.4),                        

                          nn.Linear(2048,1000),

                          nn.BatchNorm1d(1000),

                          nn.ReLU(),

                          nn.Linear(1000,102),

                          nn.LogSoftmax(dim=1))

model.fc=fc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learn = Learner(data=iB,model=model,model_dir='/tmp/model',metrics=[accuracy])

# learn.lr_find()



# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(8,slice(2e-02))
from torch.utils import data as D

import glob

import os.path as osp

import torchvision.transforms as transforms

from PIL import Image



class TestSet(D.Dataset):

    """

    A customized data loader.

    """

    def __init__(self, root):

        """ Intialize the dataset

        """

        self.filenames = []

        self.root = root

        filenames = glob.glob(osp.join(path, '*.jpeg'))

        for fn in filenames:

            self.filenames.append(fn)

        self.len = len(self.filenames)

        

    # You must override __getitem__ and __len__

    def __getitem__(self, index):

        """ Get a sample from the dataset

        """

        image = Image.open(self.filenames[index])

        return {'image':transforms.ToTensor()(image)}



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return self.len

path = '../input/test set/test set'

testset = TestSet(path)

testloader = torch.utils.data.DataLoader(testset,batch_size=1)

im = next(iter(testloader))

mod = learn.model

mod.eval()

diag=[]

for id, d in enumerate(testloader):

    images = d["image"]

    #print(images)

    images = images.to(device)

    output = mod(images)

    pred = torch.exp(output)

    top_ps,top_class = pred.topk(1,dim=1)

    diag.append(int(top_class.cpu()))



import pandas

ddf = pandas.DataFrame(data={'diagnosis':diag})

ddf.to_csv('./submission.csv',sep=',',index=False)

imageNames = glob.glob(osp.join('../input/test set/test set','*.jpeg'))

idx = 0

for ima in imageNames:

    print(str(ima)+str('--->')+str(diag[idx]))

    idx+=1