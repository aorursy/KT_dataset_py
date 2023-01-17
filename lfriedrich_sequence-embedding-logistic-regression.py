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
!pip install laserembeddings
!python -m laserembeddings download-models
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences













from keras.utils import to_categorical

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import tensorflow as tf
df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv", names=['comment', 'label'], header=0, encoding='utf-8')
df.head(10)
document_lenghts = list(map(len, df.comment.values))

print(np.max(document_lenghts))

print(np.min(document_lenghts))

print('mean_size:',np.mean(document_lenghts))

print('median_size:',np.median(document_lenghts))
df.label.value_counts()
from laserembeddings import Laser

laser = Laser()
x = laser.embed_sentences(df.comment,lang='en')  # lang is only used for tokenization
y = df['label'].values

y = 1*(y=='positive')
x_original = x

y_original = y
x, y = shuffle(x_original, y_original, random_state=23)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=23)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=23)
print("train set:", x_train.shape)

print("validation set:", x_val.shape)

print("test set:", x_test.shape)
from sklearn.linear_model import LogisticRegression
print('Simple logistic regression')

clf = LogisticRegression(random_state=0).fit(x_train, y_train)
print('train loss:',np.sum(clf.predict(x_train)==y_train)/x_train.shape[0])

print('validation loss:',np.sum(clf.predict(x_val)==y_val)/x_val.shape[0])