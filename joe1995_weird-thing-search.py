# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plot

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head()
mean_array = train.iloc[:,2:].mean()

std_array = train.iloc[:,2:].std()
abolute_mean = []



for i in range(len(train)):

    tmp_abs_mean = sum([abs(train.iloc[i,j+2]-mean_array[j])/std_array[j] for j in range(200)])/200

    abolute_mean.append(tmp_abs_mean)

train['abs_mean'] = abolute_mean
import seaborn as sns
sns.distplot(train.loc[train.target==0,'abs_mean'])

sns.distplot(train.loc[train.target==1,'abs_mean'])