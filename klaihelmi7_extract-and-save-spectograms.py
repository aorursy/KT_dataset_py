# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install librosa -U --quiet

!pip install torchaudio --quiet
import warnings

warnings.simplefilter('ignore')



import os

import sys

import cv2

import glob

import math

import random

import librosa

import zipfile

import numpy as np

import pandas as pd

from librosa import display as libdisplay

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import log_loss

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchaudio

from torchaudio import transforms

from torchvision import models

from keras.utils import to_categorical

import IPython.display as ipd

from matplotlib import pyplot as plt
import random

import numpy as np

seed = 2020



random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)



if torch.cuda.is_available(): 

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


#will store our models here

os.makedirs('MODELS/', exist_ok=True)
#Placeholder for the training and test spectogram's images

#It is going to store the spec, we will shortly generate.

os.makedirs('Imgs/Train/', exist_ok=True)

os.makedirs('Imgs/Test/', exist_ok=True)




def melspectogram_dB(file_path, cst=5, top_db=80.):

  row_sound, sr = librosa.load(file_path)

  sound = np.zeros((cst*sr,))



  if row_sound.shape[0] < cst*sr:

    sound[:row_sound.shape[0]] = row_sound[:]

  else:

    sound[:] = row_sound[:cst*sr]



  spec = librosa.feature.melspectrogram(sound, sr)

  spec_db = librosa.power_to_db(spec, top_db=top_db)



  return spec_db



def spec_to_image(spec, eps=1e-6):

  mean = spec.mean()

  std = spec.std()

  spec_norm = (spec - mean) / (std + eps)

  spec_min, spec_max = spec_norm.min(), spec_norm.max()

  spec_img = 255 * (spec_norm - spec_min) / (spec_max - spec_min)

  

  return spec_img.astype(np.uint8)



def save_spec_image(spec_img, fname):

  cv2.imwrite(fname, spec_img)

train = pd.read_csv('../input/birdsong-recognition/train.csv')

train.head()
sub = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')

sub.head()
train.columns
test = pd.read_csv('../input/birdsong-recognition/test.csv')

test.head()




# Add file names

train['spec_name'] = '../input/output/Imgs/Train/'+str(train['filename'])+'.png'

sub['spec_name'] = '../input/output/Imgs/Test/'+sub['row_id']+'.png'
train.head()
#we will save just 5 rows to save time
train=train.head(5)
#Training specs

for row in tqdm(train.values):

  sound_path = '../input/birdsong-recognition/train_audio/'+str(row[2])+'/'+str(row[7]) #this corresponds to 'file_name'

  spec_name = row[-1] #this corresponds to 'spec_name'



  spec = melspectogram_dB(sound_path, 15)

  spec = spec_to_image(spec)

  save_spec_image(spec, spec_name)

spec
plt.imshow(spec)