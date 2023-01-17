# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm    # colormap



import tensorflow as tf

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# settings

LEARNING_RATE = 1e-4

TRAINING_ITERATIONS = 2500



DROPOUT = 0.5

BATCH_SIZE = 50



# set to 0 to train on all available data

VALIDATION_SIZE = 2000



# image number to output

IMAGE_TO_DISPLAY = 10
data = pd.read_csv('../input/train.csv')



print('Dimension of the dataset is: {0[0]},{0[1]}'.format(data.shape))

data.head()