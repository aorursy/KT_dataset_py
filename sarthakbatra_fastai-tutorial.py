import numpy as np

import pandas as pd



import torch



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reload latest version of dependencies

%reload_ext autoreload

%autoreload 2

%matplotlib inline
# Fast AI Imports

from fastai import *

from fastai.vision import *
# Set batchsize

bs = 64
path = Path('../input')

path_train = path/'train/train'

path_test = path/'test/test/'

path, path_train, path_test
labels_df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'sample_submission.csv')

labels_df.head()
np.random.seed(42)

test = ImageList.from_df(test_df, path=path_test)

data = (

    ImageList.from_df(labels_df, path=path_train)

                     .split_by_rand_pct(0.01)

                     .label_from_df()

                     .add_test(test)

                     .transform(get_transforms(

                         flip_vert = True,

                     ), size = 128)

                     .databunch(path=path, bs = bs).normalize(imagenet_stats)

)
data
data.show_batch(rows = 3, figsize = (10,8))
# Print classes of our classification problem

data.classes
learn = cnn_learner(data, models.resnet101, metrics = accuracy, model_dir='/tmp/model/')
learn.lr_find()
learn.recorder.plot()
lr = 3e-02
learn.fit_one_cycle(3, slice(lr))
learn.save('resnet-101-1')
learn = cnn_learner(data, models.densenet161, metrics = accuracy, model_dir='/tmp/model/')
learn.lr_find()
learn.recorder.plot()
lr = 3e-02
learn.fit_one_cycle(3, slice(lr))
learn.save('densenet-161-1')
learn = cnn_learner(data, models.resnet101, metrics = accuracy, model_dir='/tmp/model/')
learn.load('resnet-101-1');
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
preds[:, 0]
test_df['has_cactus'] = np.array(preds[:, 0])

test_df.head()
test_df.to_csv('submission_resnet_101.csv', index = False)
learn = cnn_learner(data, models.densenet161, metrics = accuracy, model_dir='/tmp/model/')
learn.load('densenet-161-1');
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
preds[:, 0]
test_df['has_cactus'] = np.array(preds[:, 0])

test_df.head()
test_df.to_csv('submission_densenet_161.csv', index = False)
from IPython.display import FileLinks

FileLinks('.')