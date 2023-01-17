# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

from fastai.metrics import error_rate

import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.autograd import Variable

import numpy as np 



import matplotlib.pyplot as plt

import time

import os

from fastai.callbacks import ActivationStats

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
#path = untar_data(URLs.MNIST_SAMPLE)

#data = ImageDataBunch.from_folder(path)

#learn = cnn_learner(data, models.resnet18, callback_fns=ActivationStats)

#learn = Learner(data, simple_cnn((3,16,16,2)), callback_fns=ActivationStats)

#learn.wd = 0.0001

#learn.fit(2)



#bs = 192

#xfit = np.linspace(0, bs-1,bs)

#mean_y = learn.activation_stats.stats[0][1][-(bs-1):]

#std_y = learn.activation_stats.stats[1][1][-(bs-1):]

# Visualize the result

#plt.plot(xfit, mean_y, '-', color='gray')



#plt.fill_between(xfit, mean_y - std_y, mean_y + std_y,

#                 color='gray', alpha=0.2)
class_names = os.listdir('../input/mushrooms/Mushrooms/')
bs = 64

path = Path("../input/mushrooms/Mushrooms/")

fnames = []

for fpath in class_names:

    print(path/f'{fpath}/')

    fnames += get_image_files(path/f'{fpath}/')
np.random.seed(2)

pat = r"/(\w+)/\d+(_).+\.jpg$"

data = ImageDataBunch.from_name_re('.', fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs, num_workers = 0).normalize(imagenet_stats)

train_dataset, test_dataset = torch.utils.data.random_split(data.train_ds, [len(data.train_ds) - 600, 600])

data = DataBunch.create(train_dataset.dataset, data.valid_ds, test_dataset.dataset, num_workers = 0)
data.show_batch(rows=3, figsize=(5,5))
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
learn = create_cnn(data, models.resnet34, metrics=[accuracy])
learn.fit_one_cycle(4, max_lr = 1e-2)
learn.freeze_to(-2)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6,5e-4))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-6,5e-5))
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)

interp.plot_top_losses(9, figsize=(10,10))

interp.plot_confusion_matrix(figsize=(10,10))

preds_test,y_test, losses_test= learn.get_preds(ds_type=data.test_ds, with_loss=True)
print("Accuracy on test set: ", accuracy(preds_test,y_test).item())