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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

gender_submission=pd.read_csv('../input/gender_submission.csv')
train.head()
print(train.shape)

print(test.shape)

print(gender_submission.shape)
print(train.columns)

print('-'*10)

print(train.columns)
train.info()
test.info()
train.head()
train.isnull().sum()
test.isnull().sum()
df_full = pd.concat([train,test],axis=0,sort=False)

print(df_full.shape)

df_full.describe()
df_full.describe(percentiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9])
df_full.tail()
df_full.describe(include='O')
import pandas_profiling as pdp
pdp.ProfileReport(train)
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(x='Survived',data=train)

plt.title('dead or alive')

plt.xticks ([0,1],['dead','alive'])

plt.show()



display(train['Survived'].value_counts())

display(train['Survived'].value_counts()/len(train['Survived']))