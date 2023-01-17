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

test = pd.read_csv("../input/black-friday/test.csv")

train = pd.read_csv("../input/black-friday/train.csv")
train.head()
test.head()
train=train.fillna(0)

train.head()
import seaborn as sns

sns.distplot(train['Purchase'])
import matplotlib.pyplot as plt

train['Age'].value_counts(normalize=True).plot.bar()

plt.show()
sns.boxplot(x='Age',y='Purchase',data=train)
train.groupby('Age')['Purchase'].agg(['mean']).plot.bar(color='r')
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=[15, 15])



train['Gender'].value_counts().plot(kind='barh', ax=axes[0,0], title='Gender')

train['Age'].value_counts().plot(kind='barh', ax=axes[0,1], title='Age')

train['City_Category'].value_counts().plot(kind='barh', ax=axes[1,0], title='City_Category')

train['Marital_Status'].value_counts().plot(kind='barh', ax=axes[1,1], title='Marital_Status')

train['Occupation'].value_counts().plot(kind='barh', ax=axes[2,0], title='Occupation')

train['Stay_In_Current_City_Years'].value_counts().plot(kind='barh', ax=axes[2,1], title='Stay_In_Current_City_Years')
train['Purchase'].values.reshape(-1,1).shape
from sklearn.preprocessing import StandardScaler,MinMaxScaler

mm=MinMaxScaler()

data_mm= mm.fit_transform((train['Purchase']).values.reshape(-1,1))

train['Purchase_mm'] = data_mm



ss=StandardScaler()

data_ss= ss.fit_transform((train['Purchase']).values.reshape(-1,1))

train['Purchase_ss']=data_ss
train[['Purchase','Purchase_mm','Purchase_ss']].head()
train[['Purchase','Purchase_mm','Purchase_ss']].describe()
fig,ax=plt.subplots(1,3, figsize=(10,3))

sns.distplot(train['Purchase'],ax=ax[0])

sns.distplot(train['Purchase_mm'],ax=ax[1])

sns.distplot(train['Purchase_ss'],ax=ax[2])

plt.show()