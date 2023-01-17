# Fast AI

from fastai import *

from fastai.vision import *



# Basic Numeric Computation

import numpy as np

import pandas as pd



# Looking at directory

import os

from pathlib import Path
commit_no = 5 # Helpful for naming output file

epochs = 4 # Increase when commiting



# Properties

base_dir = Path("../input")

model_dir = '/tmp/models'

working_dir = Path('/kaggle/working')

data = slice(10)



print(os.listdir(base_dir))



# Training Folders

malaria = base_dir/os.listdir(base_dir)[0]

print(malaria)
il = ImageList.from_folder(malaria) # There is a image list. It's in a folder
il[0]
il_split = il.split_by_rand_pct() # randomly split it by some percentage (default 0.2)
il_labeled = il_split.label_from_folder() # label according to folder name (Parasitized; Uninfected)
tfms = get_transforms(

    do_flip=True, 

    flip_vert=True, 

    max_rotate=90.0, 

    max_zoom=0, 

    max_lighting=0, 

    max_warp=0, 

    p_affine=0.75, 

    p_lighting=0.75,

)



il_transformed = il_labeled.transform(tfms,size=100, padding_mode = 'zeros') # transform them and make them the same size
data = il_transformed.databunch(no_check=True, num_workers=8) # make a databunch

data = data.normalize(imagenet_stats) # normalize it with the data it was trained on so that model will converge faster
def _plot(i,j,ax):

    x,y = data.train_ds[3]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(8,8))
data.c
data.classes
data.show_batch(3, figsize=(7,6))
data
data.classes
learner = cnn_learner(data, models.resnet34, metrics=[accuracy], model_dir=model_dir)
learner.fit_one_cycle(epochs)
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(epochs)
learner.unfreeze()
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(epochs, 0.0001)
learner.freeze()
learner.fit_one_cycle(epochs, 0.0001)
learner.save(working_dir/f'resnet34_{commit_no}')
interp = ClassificationInterpretation.from_learner(learner)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(15,14), heatmap = False)
