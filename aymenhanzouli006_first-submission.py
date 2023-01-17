# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/digit-recognizer/train.csv")

print(train.shape)

train.head()
sample=pd.read_csv("../input/digit-recognizer/sample_submission.csv")

print(sample.shape)

sample.head(100)
# put labels into y_train variable

Y_train = train["label"]

# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1)
# visualize number of digits classes

plt.figure(figsize=(15,7))

g = sns.countplot(Y_train, palette="icefire")

plt.title("Number of digit classes")

Y_train.value_counts()
pd.read_csv("../input/digit-recognizer/test.csv").to_csv("first.csv")