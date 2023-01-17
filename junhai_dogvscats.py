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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.widgets import *
path = Path('/kaggle/input/')
path.ls()
path_train = path/'train'

path_test = path/'test'
fname_train = get_image_files(path_train)

fname_test = get_image_files(path_test)
fname_train[:5]
np.random.seed(42)

pat = r'/([^/.]+).\d+.jpg$'
tfms = get_transforms()
data = (ImageList.from_folder(path_train)

       .random_split_by_pct()

       .label_from_re(pat)

        .add_test_folder('../test')

        .transform(tfms, size=224)

        .databunch(bs=64, num_workers=0)

        .normalize(imagenet_stats))
print(len(data.train_ds))

print(len(data.valid_ds))

print(len(data.test_ds))
data.classes
data.show_batch(rows=3, figsize=(10, 10))
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/kaggle/working/dogsVsCats')
learn.fit_one_cycle(4)
learn.save('stage-1-resnet34-v1')
learn.load('stage-1-resnet34-v1');
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-06, 1e-05))
learn.save('stage-2-resnet34-v1')
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))

learn.load('stage-2-resnet34-v1');
data = (ImageList.from_folder(path_train)

       .split_by_rand_pct()

       .label_from_re(pat)

        .add_test_folder('../test')

        .transform(tfms=get_transforms(xtra_tfms=[contrast(scale=(0.1,2.0),p=0.9)]),

                   size=224,

                   padding_mode='border',

                   resize_method=ResizeMethod.PAD)

        .databunch(bs=64, num_workers=0)

        .normalize(imagenet_stats))
data.show_batch(row=3, figsize=(7,8))
print(len(data.train_ds))

print(len(data.valid_ds))

print(len(data.test_ds))
leran = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/kaggle/working/DogsVsCats')
learn.fit_one_cycle(4)
learn.save('stage-1-resnet34-v2')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(3e-05, 1e-04))
learn.recorder.plot_losses()
learn.save('stage-2-resnet34-v2')
learn.load('stage-2-resnet34-v1')
preds, y = learn.get_preds(ds_type=DatasetType.Test)
dog_preds = preds[:,1]
submission = pd.DataFrame({'id':os.listdir('../input/test'), 'label':dog_preds})
submission['id'] = submission['id'].map(lambda x: x.split('.')[0])
submission['id'] = submission['id'].astype(int)

submission = submission.sort_values('id')

submission.to_csv('submission.csv', index=False)