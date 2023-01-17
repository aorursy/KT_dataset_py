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
%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *

from fastai.metrics import error_rate

from fastai import *

from fastai.callbacks import *



import cv2 as cv

import numpy as np

import pandas as pd

import scipy.io as sio



torch.backends.cudnn.benchmark = True
print(os.listdir("../input/car_data/car_data/car_data"))
folder_dir = Path('../input/car_data/car_data/car_data')
folder_dir
ImageDataBunch.from_folder(

        folder_dir/'train',

        train='train',

        size=(224,224),

        valid_pct=.2).normalize(imagenet_stats)