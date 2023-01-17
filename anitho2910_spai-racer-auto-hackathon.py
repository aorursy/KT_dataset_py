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
from fastai.vision import *

from fastai.metrics import error_rate
path=pathlib.Path(os.path.normpath('../input/car_data/car_data'))
os.listdir(path)
train=path/'train'

test=path/'test'
tfms=get_transforms()
data=ImageDataBunch.from_folder(path=path,valid='test',bs=32,ds_tfms=tfms,size=(320,320)).normalize(imagenet_stats)
data.show_batch(rows=3,figsize=(12,12))
learn=cnn_learner(data,models.resnet50,pretrained=True,metrics=error_rate,model_dir='/tmp/model/')
data.classes
learn.fit_one_cycle(10)
learn.save('stage-1')

interp=ClassificationInterpretation.from_learner(learn)

losses,idx=interp.top_losses()

interp.plot_top_losses(9,figsize=(12,12))
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10,max_lr=slice(3e-06,4e-06))
learn.save('stage-2')
data.classes
listofSubdirectory=os.listdir(test)

len(listofSubdirectory)
import glob

from PIL import Image

root='../input/car_data/car_data/test/'

images=list()

for i in listofSubdirectory:

    images.append(os.listdir(root+i))

    

images
learn.load('stage-2')
learn.export('/tmp/model/export.pkl')
learn=load_learner('/tmp/model/')
columns = ['Id','Predicted']

listofprediction=list()

for i in range(len(listofSubdirectory)):

    for j in images[i]:

        pt=root+listofSubdirectory[i]+'/'+j

        img=open_image(pt)

        name=j.replace('.jpg','')

        _,pred_idx,output=learn.predict(img)

        

        listofprediction.append([name,pred_idx.item()])

array=np.array(listofprediction)
submission=pd.DataFrame(array,columns=columns)

submission
submission.to_csv('submission.csv',index=False)