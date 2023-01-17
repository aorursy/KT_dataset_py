# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the libraries

from fastai.vision import *

from fastai.metrics import accuracy, error_rate

from fastai.layers import MSELossFlat



import warnings

warnings.filterwarnings("ignore")
# Reading the train dataset

train_data = pd.read_csv('/kaggle/input/i2a2-bone-age-regression/train.csv')

train_data
# Creating the databunch, using ImageList making the labels FloatList for regression

data = (ImageList.from_df(train_data, '/kaggle/input/i2a2-bone-age-regression/images', cols='fileName')

         .split_by_rand_pct(valid_pct=0.1)

         .label_from_df(cols='boneage', label_cls=FloatList)

         .transform(get_transforms(), size=224)

         .databunch(bs=64))



# Normalizing to ImageNet mean and std

data.normalize(imagenet_stats)
# Checking one batch

data.show_batch(rows=3, figsize=(7,6))
# Creating our learner, transfer learning from a Resnet50 model, and metrics as Kaggle competition metrics RSME

learn = cnn_learner(data, models.resnet50, loss_func=MSELossFlat(), metrics=root_mean_squared_error, model_dir='/tmp/model')
# Checking for the best initial learning rate for a freezed model

learn.lr_find()

learn.recorder.plot(suggestion=True)
# Lets fit on one cycle.... or maybe a little more

learn.fit_one_cycle(4, 1e-1)
# Unfreezing the model and checking the best lr for another cycle

learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
# Lets fit once more on one cycle.... or maybe a little more

learn.fit_one_cycle(4, 3e-6)
# Let's submit

sub = pd.read_csv('/kaggle/input/i2a2-bone-age-regression/sample_submission.csv')
# Reading all the images and predicting

for i in range(len(sub)):

    # Reading the image with fastai image_open

    imageT = open_image('/kaggle/input/i2a2-bone-age-regression/images/' + sub.loc[i,'fileName'])

    

    # Cropping to have only one hand

    thresh = imageT.px.size()

    if thresh[2] > 900:

        imageT.px = imageT.px[:,:,:530]



    tensor = learn.predict(imageT)[2].numpy()

    print(tensor.item())

    sub.loc[i,'boneage'] = tensor.item()
# We can see some strange values on the predictions, this need attention

sub.describe()
# We can see the model goes ok on the train and validation, but cannot but simple used to predict

# with both hands or one cropped.

sub.drop(columns=['patientSex'], inplace=True)

sub.to_csv('sample_submission_fastia.csv', index=False)