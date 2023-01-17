# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from albumentations import *

import cv2



import os

print(os.listdir("../input"))



#!pip install pretrainedmodels

from tqdm import tqdm_notebook as tqdm

from torchvision.models import *

#import pretrainedmodels



from fastai.vision import *

from fastai.vision.models import *

from fastai.vision.learner import model_meta

from fastai.callbacks import * 



#from utils import *

import sys



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train/train.csv')
{'Cargo': 1, 

'Military': 2, 

'Carrier': 3, 

'Cruise': 4, 

'Tankers': 5}
wedge = [train['category'].value_counts()[1],train['category'].value_counts()[2],

         train['category'].value_counts()[3],train['category'].value_counts()[4],

         train['category'].value_counts()[5]]



perc = [train['category'].value_counts()[1]/len(train),

        train['category'].value_counts()[2]/len(train),

        train['category'].value_counts()[3]/len(train),

        train['category'].value_counts()[4]/len(train),

        train['category'].value_counts()[5]/len(train),

       ]

plt.pie(wedge,labels=['Cargo - '+ format(perc[0]*100, '.2f') + '%','Military - '+ format(perc[1]*100, '.2f') + '%','Carrier - '+ format(perc[2]*100, '.2f') + '%','Cruise - '+ format(perc[3]*100, '.2f') + '%','Tankers - '+ format(perc[4]*100, '.2f') + '%'],

        shadow=True,radius = 2.0)
#This is the code for Over-Sampling the images in order to make a new dataset.



# I used OpenCV2.0 for the same.



#  TRANSFORMATION -1 



#     scr = ShiftScaleRotate(p=1,rotate_limit=15)

#     hor = HorizontalFlip(p=1)

#     rbc = RandomBrightnessContrast(p=1)

#     image1 = scr(image = img)['image']

#     image1 = hor(image=image1)['image']

#     image1 = rbc(image=image1)['image']



#  TRANSFORMATION -2 



#     hor = HorizontalFlip(p=1)

#     rbc = RandomBrightnessContrast(p=1)

#     cut = Cutout(num_holes = 12,max_h_size=12,max_w_size=12,p = 1)

#     image2 = hor(image = img)['image']

#     image2 = rbc(image = image2)['image']

#     image2 = cut(image = image2)['image']

    

#  TRANSFORMATION -3 

    

#     rr = MotionBlur(p=1)

#     cs = ChannelShuffle(p=1)

#     hor = HorizontalFlip(p=1)

#     image3 = rr(image = img)['image']

#     image3 = cs(image = image3)['image']

#     image3 = hor(image = image3)['image']
path = pathlib.Path('../input/train')
tfms = get_transforms(do_flip=True,max_rotate=20.0,p_affine=0.75,

                      max_lighting=0.5, max_warp=0.3, p_lighting=0.75)



#np.random.seed(20)

np.random.seed(31)

data = ImageDataBunch.from_csv(path, folder='images', csv_labels='train.csv',

                               valid_pct=0.15, test='test', ds_tfms=tfms,

                               size=256,bs = 32)



data.show_batch(rows=3, figsize=(5,5))
fbetaW = FBeta(beta=1, average="weighted")



learn = create_cnn(data, models.resnet152, metrics=[accuracy,fbetaW],model_dir="/tmp/model/")



learn.fit_one_cycle(5, callbacks=[SaveModelCallback(learn,every='improvement',monitor='f_beta',mode='max',name='resnet-152')]) 
learn = learn.load('resnet-152')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr



learn.fit_one_cycle(5,max_lr=slice(1e-6,min_grad_lr*10),callbacks=[SaveModelCallback(learn,monitor='f_beta',every='improvement',mode='max',name='resnet-152')])

learn = learn.load('resnet-152')
#This is the part to switch out the optimizer from Adam to SGD



learn = create_cnn(data, models.resnet152, metrics=[accuracy,fbetaW],model_dir="/tmp/model/",opt_func=optim.SGD)

learn = learn.load('resnet-152')



learn.fit_one_cycle(5, callbacks=[SaveModelCallback(learn,every='improvement',monitor='f_beta',mode='max',name='resnet-152')])
learn = learn.load('resnet-152')



learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr



learn.fit_one_cycle(5,max_lr=slice(1e-6,min_grad_lr*10),callbacks=[SaveModelCallback(learn,monitor='f_beta',every='improvement',mode='max',name='resnet-152')])

learn = learn.load('resnet-152')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9,figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused()
# preds, y = learn.get_preds(ds_type=DatasetType.Test)

# y = torch.argmax(preds, dim=1)



# testFold = pd.DataFrame({'image':[],'category':[]})

# yArr = y.numpy()

# iterator = 0

# for imgName in os.listdir('images/test'):

#     testFold.loc[iterator,'image'] = imgName

#     testFold.loc[iterator,'category'] = int(yArr[iterator]+1)

#     iterator = iterator + 1

    

# test = pd.read_csv('../input/test_ApKoW4T.csv')



# test['category'] = 0



# for row in tqdm(test['image'].unique()):

#     test.loc[test['image']==row,'category'] = int(testFold.loc[testFold['image']==row,'category'].values[0])



# test['category'] = test['category'].astype('int')



# test.to_csv('submission-resnet152.csv',index=False)