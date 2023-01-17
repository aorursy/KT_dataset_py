import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

import os

from tqdm import tqdm,trange

from sklearn.model_selection import train_test_split

import sklearn.metrics



import torch

import torch.nn as nn

import torch.nn.functional as F



import warnings

warnings.filterwarnings("ignore")
!pip install pytorchcv --quiet

from pytorchcv.model_provider import get_model

model = get_model("xception", pretrained=True)

# model = get_model("resnet18", pretrained=True)

model
model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
model

model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))

# model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
model
class Head(torch.nn.Module):

  def __init__(self, in_f, out_f):

    super(Head, self).__init__()

    

    self.f = nn.Flatten()

    self.l = nn.Linear(in_f, 512)

    self.d = nn.Dropout(0.75)

    self.o = nn.Linear(512, out_f)

    self.b1 = nn.BatchNorm1d(in_f)

    self.b2 = nn.BatchNorm1d(512)

    self.r = nn.ReLU()



  def forward(self, x):

    x = self.f(x)

    x = self.b1(x)

    x = self.d(x)



    x = self.l(x)

    x = self.r(x)

    x = self.b2(x)

    x = self.d(x)



    out = self.o(x)

    return out
model
class FCN(torch.nn.Module):

  def __init__(self, base, in_f):

    super(FCN, self).__init__()

    self.base = base

    self.h1 = Head(in_f, 1)

  

  def forward(self, x):

    x = self.base(x)

    return self.h1(x)



model = FCN(model, 2048)
model
# !pip install torchtoolbox --quiet

# from torchtoolbox.tools import summary



# model.cuda()

# summary(model, torch.rand((1, 3, 150, 150)).cuda())