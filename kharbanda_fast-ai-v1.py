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
!rm -rf input
!cp -r ../input .
from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
path = Path('input/train/')
np.random.seed(42)
data = ImageDataBunch.from_folder(path,test='../test', ds_tfms=get_transforms(),valid_pct=0.25,size=299,bs=32,num_workers=0)
data.normalize(imagenet_stats)
print(data.classes)
len(data.classes),data.c
data.show_batch(rows=3,figsize=(7,6))
learn = create_cnn(data,models.resnet50,metrics=error_rate)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(6,slice(1e-2))
learn.save('stg-1')
learn.lr_find()
learn.recorder.plot()

#learn.load('stg-1')
learn.unfreeze()
learn.fit_one_cycle(4,max_lr=slice(1e-4,1e-3))
data = ImageDataBunch.from_folder(path,test='../test', ds_tfms=get_transforms(),valid_pct=0.25,size=350,bs=32,num_workers=0)
data.normalize(imagenet_stats)
interp = ClassificationInterpretation.from_learner(learn)
from sklearn import metrics
print(metrics.classification_report(interp.y_true.numpy(), interp.pred_class.numpy(),target_names =data.classes))
learn.save('stg-2')
learn.data=data
learn.unfreeze()
learn.fit_one_cycle(4,max_lr=slice(1e-5,1e-4))
preds,y=learn.get_preds(ds_type=DatasetType.Test)
preds = np.argmax(preds, axis = 1)
preds_classes = [data.classes[i] for i in preds]
submission = pd.DataFrame({ 'file': os.listdir('input/test'), 'species': preds_classes })
submission.to_csv('test_classification_results.csv', index=False)
submission
!rm -rf input
