# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



print(os.listdir("../input"))



%matplotlib inline

import torch

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import torchvision

import torchvision.transforms as transforms

from torch.autograd import Variable

from torch.autograd import Function

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



use_gpu = torch.cuda.is_available()

if use_gpu:

    print('GPU is available!')

else:

    print('GPU is not available')

# Any results you write to the current directory are saved as output.