import os

from fastai.vision import *

from fastai.metrics import error_rate
%reload_ext autoreload

%autoreload 2

%matplotlib inline
bs = 64
img_dir='../input/cell_images/cell_images/' #Get the cell_images folder and call it img_dir

print(os.listdir(img_dir)) #View the content of the img_dir
path = Path(img_dir); #Get the path of img_dir

#Create a separate path for the affected and unaffected images

pathAffected = path/os.listdir(img_dir)[0]

pathUnaffected = path/os.listdir(img_dir)[-1]

#List the items in paths

print(f'Some Images in Parasitized folder: \n {os.listdir(pathAffected)[0:4]} \n')

print(f'Some Images in Uninfected folder: \n {os.listdir(pathUnaffected)[0:4]}')
#

data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224,bs=bs, 

                                  num_workers=0).normalize(imagenet_stats)
#Print the labels of the data and its size

print(f'Classes: \n {data.classes}')

print(f'Size of the data classes: \n {len(data.classes),data.c}')

#Show 

data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
learn.fit_one_cycle(4)