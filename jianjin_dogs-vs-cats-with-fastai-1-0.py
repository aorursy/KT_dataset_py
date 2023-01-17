# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from fastai import *
from fastai.vision import * 
from fastai.datasets import *  # this is needed for untar_data
from fastai.metrics import * # for accuracy
from fastai.vision.data import * 
from fastai.vision.image import *
path = untar_data(URLs.DOGS)
path
os.listdir(path)
# This file contains all the main external libs we'll use
from fastai.imports import *
torch.cuda.is_available()
torch.backends.cudnn.enabled
sz=224
arch=models.resnet34
data = ImageDataBunch.from_folder(path,test='test1',
                                  ds_tfms=get_transforms(), 
                                  size=sz,
                                  num_workers=0)
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
learn.fit(3)
PATH = '../input/'
os.listdir(PATH)
os.listdir("../input/test")[:5]
submission = pd.DataFrame(os.listdir("../input/test"),columns = ['FileName'])
submission['id']= submission['FileName'].apply(lambda s: s[:len(s)-4])
submission['label']=0.5
submission.head()
len(submission)
test_path = PATH+'test/'
test_path
for i in range(10):
    img = open_image(test_path+submission['FileName'][i])
    result = learn.predict(img)
    print(submission['FileName'][i],result[0],result[1])
submission.loc[0,'label']
for i in range(len(submission)):
    img = open_image(test_path+submission['FileName'][i])
    result = learn.predict(img)
    submission.loc[i,'label']=float(result[2][1])
submission[:100]
submission[['id','label']].to_csv('submission.csv',index=False)