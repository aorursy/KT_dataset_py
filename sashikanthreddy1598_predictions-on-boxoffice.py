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

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as st

import missingno as mno

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.describe()
df_train.head()
df_train.tail()
df_train.shape, df_test.shape


#Let us examine numerical features in the train dataset

numerical_features = df_train.select_dtypes(include=[np.number])



numerical_features.columns
#Let us examine categorical features in the train dataset

categorical_features = df_train.select_dtypes(include=[np.object])



categorical_features.columns
import missingno as msno
msno.matrix(df_train)
msno.heatmap(df_train)
msno.dendrogram(df_train)
df_train.skew()
df_train.kurt()
#Correlation Heatmap

correlation= df_train.corr()

f , ax = plt.subplots(figsize = (14,12))



plt.title('Correlation of Numeric Features ',y=1,size=16)



sns.heatmap(correlation,square = True,  vmax=0.8)
df_train.isna().sum()
df_test.isna().sum()
df_train = df_train[['budget','rating','totalVotes','popularity','runtime','release_year','release_month','release_dayofweek','revenue']]

f,ax = plt.subplots(figsize=(10, 8))

sns.heatmap(df_train.corr(), annot=True)

plt.show()