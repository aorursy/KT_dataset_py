# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mlt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#reading of dataset
dataset=pd.read_csv("../input/StudentsPerformance.csv")
dataset.head()
dataset.tail()
dataset['test preparation course'].loc[dataset['test preparation course'] =='none'] = None
dataset.isnull().sum()
dataset.drop(['test preparation course'], axis=1, inplace=True)
dataset.tail()
mlt.figure(figsize=(20,10))
sns.countplot(y="gender", hue="race/ethnicity", data=dataset)
mlt.show()
mlt.figure(figsize=(20,10))
sns.countplot(y="gender", hue="parental level of education", data=dataset)
sns.distplot(dataset['math score']);
mlt.show()
sns.distplot(dataset['reading score']);
mlt.show()
sns.distplot(dataset['writing score']);
np.mean(dataset['writing score'])
np.mean(dataset['reading score'])
np.mean(dataset['math score'])

import seaborn as sns
%matplotlib inline

# calculate the correlation matrix
corr = dataset.corr()
print(corr)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
