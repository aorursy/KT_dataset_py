# Import Required Python Packages :



import warnings

warnings.filterwarnings('ignore')



# Setting up our enviroment

# Data Viz & Regular Expression Libraries :

%reload_ext autoreload

%autoreload 2

%matplotlib inline



# Scientific and Data Manipulation Libraries :

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Import FastAI library 

from fastai import *

from fastai.vision import *

from fastai.metrics import error_rate

import os
x  = '/kaggle/input/cat-and-dog/training_set/training_set'

path = Path(x)

path.ls()



np.random.seed(40)

data = ImageDataBunch.from_folder(path, train = '.', valid_pct=0.2,

                                  ds_tfms=get_transforms(), size=224,

                                  num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=4, figsize=(7,6),recompute_scale_factor=True)
data


print(data.classes)

len(data.classes)

data.c
learn = cnn_learner(data, models.resnet34, metrics=[accuracy], model_dir = Path('../kaggle/working'),path = Path("."))

learn.lr_find()

learn.recorder.plot(suggestions=True)
lr1 = 1e-3

lr2 = 1e-1

learn.fit_one_cycle(4,slice(lr1,lr2))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
img = open_image('../input/cat-and-dog/test_set/test_set/dogs/dog.4003.jpg')

print(learn.predict(img)[0])

img
learn.export(file = Path("/kaggle/working/export.pkl"))

learn.model_dir = "/kaggle/working"

learn.save("stage-1",return_path=True)