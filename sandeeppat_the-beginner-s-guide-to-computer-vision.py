# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        (os.path.join(dirname, filename));



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/flower-recognition-he/he_challenge_data/data/train.csv')
img=cv2.imread('../input/flower-recognition-he/he_challenge_data/data/train/12.jpg')
img.shape
plt.imshow(img)
plt.imshow(cv2.resize(img, (50,50)))
#Importing the libraries

from fastai.vision import *

from fastai.vision.models import *

from fastai.vision.learner import model_meta
model = nn.Sequential(

        nn.Conv2d(3,8,kernel_size=3,stride=2,padding=2), #8*250*250

        nn.BatchNorm2d(8),

        nn.ReLU(),

        

        nn.Conv2d(8,16,kernel_size=3,stride=2,padding=2), #16*125*125

        nn.BatchNorm2d(16),

        nn.ReLU(),

        

        nn.Conv2d(16,32,kernel_size=3,stride=2,padding=2), #32*63*63

        nn.BatchNorm2d(32),

        nn.ReLU(),

    

        nn.Conv2d(32,64,kernel_size=3,stride=2,padding=2), #64*32*32

        nn.BatchNorm2d(64),

        nn.ReLU(),

    

        nn.Conv2d(64,128,kernel_size=3,stride=2,padding=2), #128*16*16

        nn.BatchNorm2d(128),

        nn.ReLU(),

            

        nn.Conv2d(128,256,kernel_size=3,stride=2,padding=2), #256*8*8

        nn.BatchNorm2d(256),

        nn.ReLU(),

    

        nn.Conv2d(256,128,kernel_size=3,stride=2,padding=2), #128*4*4

        nn.BatchNorm2d(128),

        nn.ReLU(),

        

        nn.Conv2d(128,102,kernel_size=3,stride=2,padding=2), #102*2*2

        nn.BatchNorm2d(102),

        nn.ReLU(),

    

        nn.Conv2d(102,102,kernel_size=3,stride=2,padding=2), #102*1*1

        nn.BatchNorm2d(102),

        Flatten()        

)
model
path = pathlib.Path('../input/flower-recognition-he/he_challenge_data/data/');path.ls()

np.random.seed(20)

data = ImageDataBunch.from_csv(path, folder='train', csv_labels='train.csv',suffix='.jpg',

                               valid_pct=0.15, test='test',

                               size=128,bs = 64)
learn = Learner(data,model,metrics=[accuracy])
learn.summary()
learn.fit(3)