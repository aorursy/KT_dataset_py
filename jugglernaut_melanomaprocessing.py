# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai import *
from fastai.vision import *

import warnings
warnings.filterwarnings("ignore")

%reload_ext autoreload
%autoreload 2
%matplotlib inline

np.random.seed(1)
#Load images and apply some basic transformations and augmentations
trainCsvPath = '/kaggle/input/jpeg-melanoma-768x768/train.csv'
imgPath = '/kaggle/input/jpeg-melanoma-768x768/train'
df = pd.read_csv(trainCsvPath)
df['withJpg']=df['image_name']+'.jpg'
tfms = get_transforms(flip_vert=True, max_zoom=1.05, max_warp=0.)
data = ImageDataBunch.from_df(imgPath, df, fn_col='withJpg', label_col='benign_malignant', ds_tfms=tfms, size=224, bs=256)
data.normalize(imagenet_stats)
fullSizeData = ImageDataBunch.from_df(imgPath, df, fn_col='withJpg', label_col='benign_malignant', ds_tfms=tfms, size=768, bs=32)
#can use data instead of fullSizeData here when testing for faster processing
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.data = fullSizeData
learn.fit_one_cycle(2)
learn.freeze()
learn.fit_one_cycle(2)
learn.model_dir = '../../../kaggle/working'
learn.save('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(1, max_lr=slice(3e-6,3e-4))
learn.save('stage-2')

testImageFilePaths = get_image_files('/kaggle/input/jpeg-melanoma-768x768/test')
    
outputDf = pd.DataFrame(columns=['image_name','target'])

testDir = '/kaggle/input/jpeg-melanoma-768x768/test/'
i = 1
for filename in os.listdir(testDir):
    fullFilePath = testDir + filename
    classification,_,probabilities = learn.predict(open_image(fullFilePath))
    malignProbability = 1-probabilities[0]
    if str(classification) == 'malignant':
        malignProbability = probabilities[0]       
    formatted = float("{:.7f}".format(malignProbability))
    outputDf = outputDf.append({'image_name': filename[:-4],'target':formatted}, ignore_index=True)
    if i % 1000 == 0:
        print(f'{i}/{len(os.listdir(testDir))}')
    i+=1
outputDf.to_csv('output.csv', index=False)