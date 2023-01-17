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
import pandas as pd

import numpy as np

import math

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn import preprocessing
# load data to pandas dataframe

data = pd.read_csv('../input/scmp2k19.csv')

print("Data heads:")

print(data.head())

print("Null values in the dataset before preprocessing:")

print(data.isnull().sum())

print("Filling null values with mean of that particular column")

data=data.fillna(np.mean(data))

print("Mean of data:")

print(np.mean(data))

print("Null values in the dataset after preprocessing:")

print(data.isnull().sum())

print("\n\nShape: ",data.shape)
print("Info:")

print(data.info())
print("Group by:")

data.groupby('SUBDIVISION').size()