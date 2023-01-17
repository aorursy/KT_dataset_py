# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head(10)
corr1=df.corr()

corr1
sns.heatmap(corr1)
plt.hist(df['age'],bins=10)

plt.xlabel('Age')
sns.scatterplot(x=df['age'],y=df['chol'],hue=df['target'])

plt.ylabel('Cholestrol')

plt.xlabel('Age')

plt.show()
plt.boxplot(df['chol'])

plt.ylabel('Cholestrol')

plt.show()
sns.countplot(x='target',hue='sex',data=df)

female=len(df[df.sex==0])

male=len(df[df.sex==1])

print('No of female patients:',(female/len(df))*100)

print('No of male patients:',(male/len(df))*100)
sns.countplot(x='target',hue='cp',data=df)
sns.countplot(x='target',hue='fbs',data=df)
sns.countplot(x='target',hue='exang',data=df)
sns.countplot(x='target',hue='slope',data=df)