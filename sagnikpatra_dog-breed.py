# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate,accuracy
bs = 64
help(untar_data)
path = untar_data(URLs.PETS); path
path.ls()
path_anno = path/'annotations'

path_img = path/'images'
fnames = get_image_files(path_img)

fnames[:5]
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs

                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.plot_confusion_matrix(normalize=True,figsize=(12,12), dpi=90)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.save('stage-2')
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),

                                   size=299, bs=bs//2).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=[error_rate,accuracy])
learn.lr_find()

learn.recorder.plot()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(8)
learn.save('stage-1-50');
learn.fit_one_cycle(8,max_lr=slice())
learn.unfreeze()

learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
learn.load('stage-1-50');
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
path = untar_data(URLs.MNIST_SAMPLE); path
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)
!ls -1 /tmp/.fastai/data/mnist_sample/train/
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet18, metrics=accuracy)

learn.fit(2)
df = pd.read_csv(path/'labels.csv')

df.head()
data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)
data.show_batch(rows=3, figsize=(5,5))

data.classes
data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)

data.classes
fn_paths = [path/name for name in df['name']]; fn_paths[:2]
pat = r"/(\d)/\d+\.png$"

data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)

data.classes
data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,

        label_func = lambda x: '3' if '/3/' in str(x) else '7')

data.classes
tfms
labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]

labels[:5]
data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)

data.classes
data.show_batch()
data.save('Stage-3')
path = untar_data(URLs.PETS); path
path_anno = path/'annotations'

path_img = path/'images'
fnames = get_image_files(path_img)

fnames[:5]
np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=32

                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.densenet161, metrics=error_rate)
learn.model
learn.fit_one_cycle(3)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)