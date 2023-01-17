# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/art-images-drawings-painting-sculpture-engraving"))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.vision import *
#Take PATH to be the parent folder of training and valid set.

PATH = "../input/art-images-drawings-painting-sculpture-engraving/art_dataset_cleaned/art_dataset"
torch.cuda.is_available()
#do_flip=False because we do not want our paintings to be trained flipped or in any other rotation.

tfms = get_transforms(do_flip=False)



#size = the maximum size of our images, use 224 for most cases as told by Jeremy. 

#num_workers = 0, the number of CPUs to use. 0 due to lower hardware in Kaggle. 

#If training and valid set are already available, direct the function to them via the method below via arguments train and valid. 

data = ImageDataBunch.from_folder(PATH, train="training_set", valid="validation_set", ds_tfms=tfms, size=200, num_workers=0)
data.show_batch(rows=3, figsize=(6,6))
data.classes
#BEFORE THIS STEP, Click on Add Data and Search for Resnet34 from Kaggle. Because Kaggle serves read-only dirs, we cannot download pre-trained weights for Resnet. Now we have to make a dir 

#for copying those weights and hence we are making ~/.torch/models folder. 

cache_dir = os.path.expanduser(os.path.join('~','.torch'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir) #first make ~/.torch if not already available.
models_dir = os.path.join(cache_dir,'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir) #then make ~/.torch/models, if not already available. 
#Copied resnet34.pth, which are pretrained weights on Resnet34 to our folder into resnet<version>-<sha-hash>.pth

!cp ../input/resnet34/resnet34.pth ~/.torch/models/resnet34-333f7ec4.pth 
#MODEL_PATH is declared this way and glued to model_dir attr of cnn_learner.

MODEL_PATH = '/tmp/models'

learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir=MODEL_PATH)
#Fitting and checking for the first time. 

learn.fit_one_cycle(4)
#Saving the model with ACCURACY = 93.4%

learn.save('stage-1')
#Initiating refit and checking LR

learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
#The lowest loss is at 1e-06 and loss increases from after 1e-04. Refitting by modulating LR

learn.fit_one_cycle(2,max_lr=slice(1e-05,1e-04))
#Saving the model with accuracy 93.6%

learn.save('stage-2')
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2,max_lr=slice(1e-06,1e-05))
#Saving model with acc 94.7%

learn.save('stage-3')
#Uploaded a separate dataset for testing each of the above classes.

PRED_PATH = "../input/for-testing-art-images-cnn"

img_icono = open_image(f'{PRED_PATH}/icono.jpg')

img_drawing = open_image(f'{PRED_PATH}/drawing.jpg')

img_engraving = open_image(f'{PRED_PATH}/engraving.png')

img_painting = open_image(f'{PRED_PATH}/painting.jpg')

img_sculpt = open_image(f'{PRED_PATH}/sculpture.jpg')
img_icono
img_drawing
img_engraving
img_painting
img_sculpt
learn.load('stage-3')

pred_class = learn.predict(img_icono)

pred_class

#The iconography is correctly identified in its specified category.
pred_class = learn.predict(img_drawing)

pred_class

#The sketch of Robert Downey Jr. is correctly identified as a drawing. 
pred_class = learn.predict(img_painting)

pred_class

#The MonaLisa painting has been incorrectly identified as a drawing too. 
pred_class = learn.predict(img_sculpt)

pred_class

#The Abraham Lincoln statue is correctly predicted as a sculpture.
pred_class = learn.predict(img_engraving)

pred_class

#The engraving is correctly identified in its specified category.