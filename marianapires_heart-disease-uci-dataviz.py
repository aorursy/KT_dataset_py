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

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
age = df['age']

age.describe()
sns.distplot(df['age'])

plt.show()
sns.boxplot(y = df['age'])

plt.show()
sns.boxplot(y = df['age'], x = df['target'])

plt.show()
sns.countplot(df['sex'])

plt.show()
sns.countplot(x='sex', hue='target', data=df)

plt.show()
sns.countplot(df['cp'])

plt.show()
sns.countplot(x='cp', hue='target', data=df)

plt.show()
df['trestbps'].describe()
sns.distplot(df['trestbps'])

plt.show()
sns.boxplot(y = df['trestbps'])

plt.show()
sns.boxplot(y = df['trestbps'], x = df['target'])

plt.show()
df['chol'].describe()
sns.distplot(df['chol'])

plt.show()
sns.boxplot(y = df['chol'])

plt.show()
sns.boxplot(y = df['chol'], x = df['target'])

plt.show()
sns.countplot(df['fbs'])

plt.show()
sns.countplot(df['fbs'], hue = df['target'])

plt.show()
sns.countplot(df['restecg'])

plt.show()
sns.countplot(df['restecg'], hue = df['target'])

plt.show()
df['thalach'].describe()
sns.distplot(df['thalach'])

plt.show()
sns.boxplot(y = df['thalach'])

plt.show()
sns.boxplot(y = df['thalach'], x = df['target'])

plt.show()
sns.countplot(df['exang'])

plt.show()
sns.countplot(df['exang'], hue = df['target'])

plt.show()
df['oldpeak'].describe()
sns.distplot(df['oldpeak'])

plt.show()
sns.boxplot(y = df['oldpeak'])

plt.show()
sns.boxplot(y = df['oldpeak'], x = df['target'])

plt.show()
sns.countplot(df['slope'])

plt.show()
sns.countplot(df['slope'], hue = df['target'])

plt.show()
sns.countplot(df['ca'])

plt.show()
sns.countplot(df['ca'], hue = df['target'])

plt.show()
sns.countplot(df['thal'])

plt.show()
sns.countplot(df['thal'], hue = df['target'])

plt.show()