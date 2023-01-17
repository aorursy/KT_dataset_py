import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import itertools

import codecs

import re

import datetime

import cairocffi as cairo

import editdistance

from scipy import ndimage

import pylab

from keras import backend as K

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import Input, Dense, Activation

from keras.layers import Reshape, Lambda

from keras.layers.merge import add, concatenate

from keras.models import Model

from keras.layers.recurrent import GRU

from keras.optimizers import SGD

from keras.utils.data_utils import get_file

from keras.preprocessing import image

import keras.callbacks
# https://www.kaggle.com/anokas/kuzushiji-visualisation

df_train = pd.read_csv('../input/train.csv')

df_unicode = pd.read_csv('../input/unicode_translation.csv')

df_submission = pd.read_csv('../input/sample_submission.csv')

train_image_path = "../input/train_images"

test_image_path = "../input/test_images"

unicode_map = {codepoint: char for codepoint, char in df_unicode.values}
df_train.head()
df_train['labels'].isna().value_counts()
df_train = df_train.fillna('')
df_submission.head()
for i, row in df_submission.iterrows():

    samples = df_train.iloc[i%df_train.shape[0]]['labels']

    result = ''

    tmp = ''

    j=0

    for sample in samples.split(' '):

        if j == 0:

            tmp += sample + ' '

            j += 1

        elif j == 1:

            tmp += sample + ' '

            j += 1

        elif j == 2:

            tmp += sample + ' '

            j += 1

        elif j == 3:

            j += 1

        else:

            result += tmp

            tmp = ''

            j = 0



    if len(result) == 0:

        df_submission.set_value(i, 'labels', '')

    else:

        df_submission.set_value(i, 'labels', result[:-1])
df_submission.head()
df_submission.to_csv("submission.csv", index=False)