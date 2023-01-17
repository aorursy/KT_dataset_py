# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from glob import glob

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Put these at the top of every notebook, to get automatic reloading and inline plotting

%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate, accuracy
image_dir = "../input"



imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(image_dir, '*', '*.jpg'))}



lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'dermatofibroma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}
image_dataframe = pd.read_csv(os.path.join(image_dir, 'HAM10000_metadata.csv'))

image_dataframe.head(5)
image_dataframe['path'] = image_dataframe['image_id'].map(imageid_path_dict.get)

image_dataframe['cell_type'] = image_dataframe['dx'].map(lesion_type_dict.get) 

image_dataframe['cell_type_idx'] = pd.Categorical(image_dataframe['cell_type']).codes

image_dataframe.head(5)
import matplotlib.pyplot as plt



fig, ax = plt.subplots(1, 1, figsize = (12, 6))

image_dataframe['cell_type'].value_counts().plot(kind='bar', ax=ax)
image_dataset = pd.concat([image_dataframe['path'], image_dataframe['cell_type']], axis=1, keys=['name', 'label'])



image_dataset.head(5)
bs = 64



tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_df(".", image_dataset, ds_tfms=tfms, size=28, bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.model
learn.fit_one_cycle(5)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, max_lr=slice(1e-2,1e-4))
data = ImageDataBunch.from_df(".", image_dataset, ds_tfms=tfms, size=28, bs=bs//2).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10)
learn.save('stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))
learn = cnn_learner(data, models.resnet152, metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)