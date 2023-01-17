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
train = pd.read_csv('../input/train.csv')
print(train.shape)

print("Number of features in the dataset = %i" % train.shape[0])

print("Number of features in the daataset = %i" % train.shape[1])
train.head()
train.head(7)
train.corr()
import seaborn as sns





import matplotlib.pyplot as plt





corr = train.corr()

f, ax = plt.subplots(figsize=(25, 25))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5)

train.isna().sum()
train['Elevation'] = None
train.drop(['Aspect'],inplace = True, axis = 1)
train.drop(['Cover_Type'],inplace = True, axis = 1)
train.isna().sum()
train.head()
train.dropna(inplace = True, axis = 1)
numerical = train.drop(["Soil_Type"+str(x) for x in range(1, 41)], axis = 1)
numerical = numerical.drop(["Wilderness_Area" + str(x) for x in range(1, 5)], axis = 1)
from scipy import stats

import numpy as np



z = np.abs(stats.zscore(numerical))



print(len(np.unique(np.where(z > 3)[0])) )
from scipy import stats

import numpy as np



t_rain = train[(z <= 3).all(axis = 1)]



print( t_rain.shape )