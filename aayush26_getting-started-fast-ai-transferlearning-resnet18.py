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
from pathlib import Path 

from fastai import *

from fastai.vision import *

import torch
# stores Path of input directory to be used as an argument in fetching test and train 

data_folder = Path("../input")

data_folder.ls()
# Read CSV train and test files using Pandas

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
# Load test images using from_df. Get the filenames in cols of test_df with folder in front of them

# Read more: https://docs.fast.ai/vision.data.html#ImageList.from_df

test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')
# do_flip = True --> a random flip is applied with probability 0.5

# flip_vert = True --> requires do_flip = True, the image will be flipped vertically or rotated by 90 degrees

# max_rotate = 10.0 --> random rotation between -10 to +10 with probability of p_affine

# max_zoom = 1.1 --> random zoom applied between 1 and 1.1 with probability of p_affine

# max_lighting = 0.2 --> lighting and contrast of magnitude between -0.2 and 0.2 applied with probability of p_lighting

# max_wrap = 0.2 --> symmetric warp of magnitude between -0.2 and 0.2 with probability of p_affine

# p_affine = 0.75 --> 0.75 probability used to apply max_rotate, max_zoom, max_wrap

# p_lighting = 0.75 --> 0.75 probability used to apply max_lighting

# Read more: https://docs.fast.ai/vision.transform.html#get_transforms

trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')  # load training images using from_df()

        .split_by_rand_pct(0.01)                                                    # randomly puts 0.01 = 1% of data into validation set

        .label_from_df()                                                            # fetches all labels with corresponding images. Only works with from_df()

        .add_test(test_img)                                                         # adds test images

        .transform(trfm, size=128)                                                  # transformation are applied to the training set

        .databunch(path='.', bs=64, device= torch.device('cuda:0'))                 # creates DataBunch with batchsize = 64, and device = cuda index 0 GPU

        .normalize(imagenet_stats)                                                  # using imagenet_stats to normalize the dataset. Other valid values are cifar_stats and mnist_stats

       )
# Displays 2 rows of images from the training dataset

train_img.show_batch(rows=2, figsize=(7,6))
# Using resnet18 base architecture for transfer learning and metrics as error_rate and accuracy.

# The various models fastai offers for vision are torch models (https://pytorch.org/docs/stable/torchvision/models.html) + fastai models (https://docs.fast.ai/vision.models.html)

# Read more: https://docs.fast.ai/vision.learner.html#cnn_learner

learn = cnn_learner(train_img, models.resnet18, metrics=[error_rate, accuracy])
learn.lr_find()

learn.recorder.plot()
lr = 3e-02

learn.fit_one_cycle(5, slice(lr))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test_df.has_cactus = preds.numpy()[:, 0]
test_df.to_csv('submission_resnet_18.csv', index=False)