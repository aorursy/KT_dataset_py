# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/image/image'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

import os, torch, glob

import numpy as np

from torch.autograd import Variable

from PIL import Image  

from torchvision import models, transforms

import torch.nn as nn

import shutil

data_dir = '/kaggle/input/image/image'

features_dir = '/kaggle/output'

shutil.copytree(data_dir, os.path.join(features_dir, data_dir[2:]))
def extractor(img_path, saved_path, net, use_gpu):

    transform = transforms.Compose([

            transforms.Scale(256),

            transforms.CenterCrop(224),

            transforms.ToTensor() ]

    )

    

    img = Image.open(img_path).convert('RGB')

    img = transform(img)

    print(img.shape)

 

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

    if use_gpu:

        x = x.cuda()

        net = net.cuda()

    y = net(x).cpu()

    y = y.data.numpy()

    np.savetxt(saved_path, y, delimiter=',')

extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        

files_list = []

sub_dirs = [x[0] for x in os.walk(data_dir) ]

sub_dirs = sub_dirs[1:]

for sub_dir in sub_dirs:

    for extention in extensions:

        file_glob = os.path.join(sub_dir, '*.' + extention)

        files_list.extend(glob.glob(file_glob))

resnet50_feature_extractor = models.resnet50(pretrained = True)

resnet50_feature_extractor.fc = nn.Linear(2048, 2048)

torch.nn.init.eye(resnet50_feature_extractor.fc.weight)

for param in resnet50_feature_extractor.parameters():

        param.requires_grad = False   

    

use_gpu = torch.cuda.is_available()
use_gpu
import cv2

i=1

for x_path in files_list:

    print(i,x_path)

    i+=1

    print(cv2.imread(x_path).shape)

    fx_path = os.path.join(features_dir, x_path[2:] + '.txt')

    extractor(x_path, fx_path, resnet50_feature_extractor, use_gpu)
