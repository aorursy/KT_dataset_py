# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
bs=64
!mkdir data
mycsv=pd.read_csv("../input/train.csv")

mycsv.head()
!mkdir ./data/train

!mkdir ./data/train/1

!mkdir ./data/train/0

for i in mycsv.values:

    shutil.copy("../input/train/train/"+i[0],"./data/train/"+str(i[1])+"/"+i[0])
path=Path('./data')
data= ImageDataBunch.from_folder(path,train="./train",valid_pct=0.2,ds_tfms=get_transforms(),size=bs,num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3,figsize=(7,6))
#!mkdir ../test

#for file in os.listdir('../input/test/test'):

    #shutil.copy('../input/test/test/'+file,'../test/'+file)
data.classes
learn=create_cnn(data,models.resnet34, metrics=error_rate)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4)
interp= ClassificationInterpretation.from_learner(learn)

losses,indxs=interp.top_losses()

len(data.valid_ds)==len(losses)==len(indxs)
interp.plot_top_losses(9, figsize=(8,8))
interp.plot_confusion_matrix(figsize=(12,12),dpi=60)
learn.save('stage-1')
!cd data/models && ls
os.listdir('../input/test/test/')[0]
img=open_image("../input/test/test/c662bde123f0f83b3caae0ffda237a93.jpg")
learn.unfreeze()

learn.fit_one_cycle(2,max_lr=slice(1e-4,1e-2))

learn.save('stage-2')
data2= ImageDataBunch.single_from_classes(path,["0","1"],ds_tfms=get_transforms(),size=64).normalize(imagenet_stats)
learn=create_cnn(data2,models.resnet34).load('stage-2')
mydf={'id':[],'has_cactus':[]}

for i in os.listdir('../input/test/test'):

    img=open_image("../input/test/test/"+i)

    pred_class, pred_idxs, outputs = learn.predict(img)

    mydf['id'].append(i)

    mydf['has_cactus'].append(pred_class)

mydf=pd.DataFrame(mydf)

mydf.head()
file=mydf.to_csv('test.csv',sep=',',index=False)
!rm -r ./data
from IPython.display import FileLink

FileLink('test.csv')