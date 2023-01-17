# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
## Declaring path of dataset

path_img = Path('../input/petfinder_images/images')

## Loading data 

data = ImageDataBunch.from_folder(path=path_img, train='train', valid='validation', ds_tfms=get_transforms(),size=224, bs=64)

## Normalizing data based on Image net parameters

data.normalize(imagenet_stats)
kappa = KappaScore()

kappa.weights = "quadratic"



learn = cnn_learner(data, models.resnet152, metrics=[ kappa], model_dir='/kaggle/working/models')

learn.loss = nn.MSELoss()

#learn.loss = nn.CrossEntropyLoss()
learn.fit_one_cycle(10)
learn.save('petfinder_fastai')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(20,20), dpi=60)



preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)
learn.lr_find()

learn.recorder.plot()
learn.purge()