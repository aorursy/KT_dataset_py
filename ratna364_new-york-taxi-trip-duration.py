# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.

# data analysis and wrangling

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning



#importing Data



train_df = pd.read_csv('../input/train.csv', parse_dates=['pickup_datetime'])

test_df = pd.read_csv('../input/test.csv', parse_dates=['pickup_datetime'])

combine = [train_df, test_df]

print(train_df.shape)

print(test_df.shape)
# we find that 2 columns a re not present in testing set lets find out which attributes are missing

print(train_df.columns.values)

print(test_df.columns.values)
# we can see that dropoff_datetime and Trip_duration are absent in test data so we have to eliminate dropoff_datetime

train_df.head()
train_df.tail()
train_df.info()
train_df.describe()