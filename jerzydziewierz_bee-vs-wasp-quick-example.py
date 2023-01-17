!pip install fastai --upgrade
%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai.vision.all import *

from fastai.metrics import error_rate



# additional classic imports

from pathlib import Path

import pandas as pd

import numpy as np

import random
bs = 64 # Batch size

resize_size = 96 # for training, resize all the images to a square of this size

training_subsample = 0.1 # for development, use a small fraction of the entire dataset rater than full dataset
bees_vs_wasps_dataset_path=Path('../input/bee-vs-wasp/kaggle_bee_vs_wasp') # this is relative to the "example_notebook" folder. Modify this to reflect your setup

df_labels = pd.read_csv(bees_vs_wasps_dataset_path/'labels.csv')

df_labels=df_labels.set_index('id')

df_labels.head()
for idx in df_labels.index:    

    df_labels.loc[idx,'path']=df_labels.loc[idx,'path'].replace('\\','/')    
df_labels = df_labels.sample(frac=training_subsample, axis=0) 
data = ImageDataLoaders.from_df(

    df = df_labels,

    path = Path(bees_vs_wasps_dataset_path),

    valid_pct=0.2,

    seed = 42,

    fn_col='path',

    folder=None,

    label_col='label',

    bs=bs,

    shuffle_train=True,

    batch_tfms=aug_transforms(),

    item_tfms=Resize(resize_size),device='cpu', num_workers=0,

)
data.show_batch()
learn = cnn_learner(data, resnet18, metrics=error_rate)

learn.model_dir='/kaggle/temp/'
best_lr=learn.lr_find(start_lr=1e-04, end_lr=1, num_it=30) 
learn.fine_tune(1,base_lr=best_lr[0])
learn.show_results()
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

interp.plot_top_losses(12, figsize=(14,14))
interp.plot_confusion_matrix(figsize=(4,4), dpi=120)