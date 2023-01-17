# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.svm import LinearSVC

from sklearn.metrics import mean_squared_error

from math import sqrt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/cardiogoodfitness/CardioGoodFitness.csv')

df.head()
df.isnull().sum()
df.describe()
df.hist(figsize=(20,7))

plt.show()
sns.boxplot(x='Age',y='Gender',data=df)

plt.show()
sns.boxplot(x='Income',y='Gender',data=df)

plt.show()
sns.boxplot(x='Education',y='Gender',data=df)

plt.show()


sns.heatmap(df.corr(),annot=True)

plt.show()
X= df[['Fitness','Usage']]

y = df['Miles']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
lr = LinearRegression()

lr.fit(X_train,y_train)
lr.coef_
lr.intercept_