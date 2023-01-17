# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/facebook-data/pseudo_facebook.csv')
df.head()
df.tail()
df.info()
df.describe()
df = pd.get_dummies(df,columns=['gender'])
df.head()
df = df.dropna()
df.info()
df['age'].value_counts()
labels=['10-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120']

df['age_group'] = pd.cut(df.age,bins=np.arange(10,121,10),labels=labels,right=True)
df.head()
df['age_group'].unique()
df['age_group'].value_counts()
sns.pairplot(data = df, hue="gender_female")
sns.countplot(x='age',data=df)
sns.countplot(x='age',hue='gender_male',data=df)
sns.barplot(df['gender_female'],df['likes'])

sns.barplot(df['gender_male'],df['likes'])
sns.lmplot( x="age", y="friend_count", data=df, fit_reg=False, hue='tenure', legend=False)
sns.jointplot(x='age',y='friend_count',data=df)
sns.stripplot(x='likes',y='friend_count',data=df,jitter=False)
df.corr()
plt.subplots(figsize=(12,12))

sns.heatmap(df.corr(), annot=True)

plt.show()