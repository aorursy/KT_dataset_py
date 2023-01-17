from fastai import *

from fastai.vision import *
import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))

from glob import glob

import random

import cv2

import matplotlib.pylab as plt

import random as rand

import keras

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization, Input

from keras.models import Sequential

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from pathlib import Path

from keras.optimizers import Adam,RMSprop,SGD
df = pd.read_csv("../input/10-monkey-species/monkey_labels.txt")

df.head()
path = Path('../input/10-monkey-species')
(path/'').ls()
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(10,10))
learn = create_cnn(data, models.resnet34, metrics= accuracy, model_dir="/tmp/model/")   #metrics= error_rate
learn.fit_one_cycle(4) 
learn.save('stage-1') #save the model
learn.unfreeze()
learn.lr_find() #finding best learning rate
learn.recorder.plot()
lr = 0.01
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-2')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-4))
learn.save('stage-3')
#np.random.seed(42)

#src = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

#        ds_tfms=get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.), size=64, num_workers=0.1).normalize(imagenet_stats)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=256, num_workers=0).normalize(imagenet_stats)



learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()

learn.recorder.plot()
lr=1e-2/2
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-4')
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.recorder.plot_losses()
learn.save('stage-5')
learn.load('stage-3');  #load the model
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
#from fastai.widgets import *
#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
#ImageCleaner(ds, idxs, path)
#ds, idxs = DatasetFormatter().from_similars(learn)
#get the image .jpg nummer 

image_path = "../input/10-monkey-species/training/training/"

images_dict = {}





for image in os.listdir(image_path):

    folder_path = os.path.join(image_path, image)

    images = os.listdir(folder_path)

    

    images_dict[image] = [folder_path, image]

    img_idx = rand.randint(0,len(image)-1)

    image_img_path = os.path.join(image_path, image, images[img_idx])

    #printing image

    img = cv2.imread(image_img_path)

    print(image_img_path) # to get the path of one image with the .jpg number; uncommen this line

    #plt.imshow(img);
import fastai

#fastai.defaults.device = torch.device('cpu')
img = open_image('../input/10-monkey-species/training/training/n4/n4146.jpg')

img
classes = ['n0', 'n1', 'n2', 'n3', 'n4','n5','n6','n7', 'n8','n9']
data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data2, models.resnet34, model_dir="/tmp/model/").load('stage-3')
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))