import os
os.listdir("../input/ammi-2020-convnets/train/train")
from fastai import *
from fastai.vision import *

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve

from math import floor
train_path = "../input/ammi-2020-convnets/train/train"
test_path = "../input/ammi-2020-convnets/test/test/0"

def get_labels(file_path): 
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_levels = len(split_dir_name)
    label  = split_dir_name[dir_levels - 1]
    return(label)
from glob import glob
imagePatches = glob("../input/ammi-2020-convnets/train/train/*/*.*", recursive=True)
imagePatches[0:10]
path=""
tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=0.10, max_zoom=1.5, max_warp=0.2, max_lighting=0.2,
                     xtra_tfms=[(symmetric_warp(magnitude=(-0,0), p=0)),]) 
data = ImageDataBunch.from_name_func(path, imagePatches, label_func=get_labels,  size=500, 
                                     bs=20,num_workers=2,test = test_path,ds_tfms=tfms
                                  ).normalize(imagenet_stats)
learner= cnn_learner(data, models.densenet121, metrics=[accuracy], ps = 0.25 ,model_dir='/tmp/models/')
learner.lr_find()
learner.recorder.plot()
lr=1e-2
learner.fit_one_cycle(1, lr)
learner.save('model-1')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.load('model-1')
learner.fit_one_cycle(3, slice(1e-4,1e-3))
learner.recorder.plot_losses()
learner.validate()
interp = ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused(min_val=2)
preds,y = learner.TTA(ds_type=DatasetType.Test)
len(preds)
SAMPLE_SUB = '../input/ammi-2020-convnets/sample_submission_file.csv'
sample_df = pd.read_csv(SAMPLE_SUB)
sample_df.head()
predictions = preds.numpy()
class_preds = np.argmax(predictions, axis=1)
for c, i in learner.data.train_ds.y.c2i.items():
    print(c,i)
categories = ['cbb','cbsd','cgm','cmd','healthy']

def map_to_categories(predictions):
    return(categories[predictions])

categories_preds = list(map(map_to_categories,class_preds))
filenames = list(map(os.path.basename,os.listdir(test_path)))
df_sub = pd.DataFrame({'Category':categories_preds,'Id':filenames})
df_sub.head()
# Export to csv
df_sub.to_csv('submission_categories.csv', header=True, index=False)
