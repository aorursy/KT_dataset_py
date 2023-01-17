import os
files = os.listdir('../input/diabetic-retinopathy-detection/')
print('trainLabels.csv.zip' in files) #Is the labels csv in the directory?
print(len(files)) #There should be 1000 images + 1 csv file = 1001 files
%reload_ext autoreload
%autoreload 2
%matplotlib inline
!pip install fastai==1.0.42
from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.pyplot as plt
print('cuda :', torch.cuda.is_available())
print('cudnn:', torch.backends.cudnn.enabled)
#base_image_dir = os.path.join('..', 'input/diabetic-retinopathy-detection')
df = pd.read_csv('../input/diabetic-retinopathy-detection/trainLabels.csv.zip')
df['path'] = df['image'].map(lambda x: os.path.join(base_image_dir,'{}.jpeg'.format(x)))
df['exists'] = df['path'].map(os.path.exists) #Most of the files do not exist because this is a sample of the original dataset
df = df[df['exists']]
df = df.drop(columns=['image','exists'])
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)