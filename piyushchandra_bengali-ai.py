# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib as mlt

##!pip install pyarrow 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pyarrow as pa

import pyarrow.parquet as pq 
import pandas as pd

class_map = pd.read_csv("../input/bengaliai-cv19/class_map.csv")

sample_submission = pd.read_csv("../input/bengaliai-cv19/sample_submission.csv")

test = pd.read_csv("../input/bengaliai-cv19/test.csv")

train = pd.read_csv("../input/bengaliai-cv19/train.csv")
class_map.head(10)
sample_submission.head(10)
test.head(10)
train.head(4)
y_train_grapheme_root=train["grapheme_root"]
y_train_grapheme_root.head(3)
y_train_vowel_diacritic=train["vowel_diacritic"]
y_train_vowel_diacritic.head(3)
y_train_consonant_diacritic=train["consonant_diacritic"]
y_train_consonant_diacritic.head(3)
test0 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_0.parquet")

test0.head(5)
test0=test0.drop(["image_id"],axis=1)
test1 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_1.parquet")
test1.head(4)
test1=test1.drop(["image_id"],axis=1)
test2 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_2.parquet")
test2=test2.drop(["image_id"],axis=1)
test3 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_3.parquet")

test3=test3.drop(["image_id"],axis=1)
x_test=pd.concat([test0,test1,test2,test3],ignore_index=True)
x_test.shape
x_test=x_test.values.reshape(-1,137,59,4)
g=plt.imshow(x_test[0][:,:,0])
train0=pd.read_parquet("../input/bengaliai-cv19/train_image_data_0.parquet")
train0.head(5)
train0=train0.drop(["image_id"],axis=1)
train1=pd.read_parquet("../input/bengaliai-cv19/train_image_data_1.parquet")

train1=train1.drop(["image_id"],axis=1)
train2=pd.read_parquet("../input/bengaliai-cv19/train_image_data_2.parquet")

train2=train2.drop(["image_id"],axis=1)
train3=pd.read_parquet("../input/bengaliai-cv19/train_image_data_3.parquet")

train3=train3.drop(["image_id"],axis=1)
x_train=pd.concat([train0,train1,train2,train3],ignore_index=True)
x_train.shape
x_train=x_train.values.reshape(-1,137,59,4)
g=plt.imshow(x_train[1000][:,:,0])
del test

del train

del train0

del train1

del train2

del train3

del test0

del test1

del test2

del test3

del class_map

del sample_submission