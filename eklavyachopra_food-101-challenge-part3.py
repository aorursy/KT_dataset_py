import os

import random

import shutil



from pathlib import Path

import urllib



import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt



from fastai.vision import *



import warnings

warnings.filterwarnings("ignore")
# Input root directory

input_root_dir = "/kaggle/input/food-101/food-101/food-101"

input_root_path = Path(input_root_dir)

print(os.listdir(input_root_dir))
# Image directory

image_dir_path = input_root_path/'images'


src = (ImageList.from_folder(image_dir_path)

       .split_by_rand_pct(0.2)       

       .label_from_folder()) 
tfms = get_transforms(max_rotate=10,

                      max_zoom=1.1,

                      max_lighting=None,

                      max_warp=0.2,

                      xtra_tfms=[

                          brightness(change=(0.5-0.2, 0.5+0.2), p=0.75),

                          contrast(scale=(1-0.4, 1+0.2), p=0.75),

                          squish(scale=(1-0.3, 1+0.5), p=0.75),

                          skew(direction=(0, 7), magnitude=random.randint(0,6)/10, p=0.75),]

                      )



# batch size

bs = 64
# create databunch



data = (src.transform(tfms, size=256)  #resize images to 256

        .databunch(bs=bs)              #batch size=64

        .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(9, 7))


print(data.classes)

len(data.classes)
food_learn = cnn_learner(data, models.resnet50,  metrics=accuracy, wd=1e-1,model_dir='/kaggle/working')
food_learn.load("/kaggle/input/model2/256_stage2") 
food_learn.unfreeze()
food_learn.summary()
# Finding a good learning rate



food_learn.lr_find()

food_learn.recorder.plot()




# Train with 1 cycle policy

food_learn.fit_one_cycle(5, slice(1e-6))



food_learn.save("256_stage3", return_path=True)

# Plot loss history

food_learn.recorder.plot_losses()
food_learn.recorder.plot_metrics()
# Check the final validation accuracy

food_learn.validate(food_learn.data.valid_dl)
# Visualize a few results

food_learn.show_results(ds_type=DatasetType.Valid)