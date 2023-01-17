# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/cardiogoodfitness/CardioGoodFitness.csv')

df.head()
df.describe(include='all')
df.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df.hist(figsize=(20,30))
sns.boxplot(df.Age, orient='h') 
sns.boxplot(x="Gender", y="Age", data=df, orient='v')
pd.crosstab(df['Product'],df['Gender'] )
sns.countplot(x="Product", hue="Gender", data=df)
pd.pivot_table(df, index=['Product', 'Gender'],

                     columns=[ 'MaritalStatus'], aggfunc=len)
pd.pivot_table(df,'Income', index=['Product', 'Gender'],

                     columns=[ 'MaritalStatus'])
pd.pivot_table(df,'Miles', index=['Product', 'Gender'],

                     columns=[ 'MaritalStatus'])
sns.pairplot(df)
print(df['Age'].mean())

print(df['Age'].std())

print(df['Age'].mode())

print(df['Age'].median())
df.hist(by='Gender',column = 'Age')
df.hist(by='Gender',column = 'Income')
df.hist(by='Gender',column = 'Miles')
df.hist(by='Product',column = 'Miles', figsize=(10,20))
df.hist(by='Product',column = 'Miles', figsize=(10,20))
corr = df.corr()

corr
sns.heatmap(corr, annot=True)
sns.lmplot(x='Age', y = 'Fitness', data = df, col= "Product", aspect = 0.6, height = 5, hue = "Gender", palette="PuBuGn_r")
q1, q3= np.percentile(df.Miles, [25,75])
iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 

upper_bound = q3 +(1.5 * iqr)
print(lower_bound)

print(upper_bound)
sorted(df.Miles)
df.Miles[df.Miles >= upper_bound]
sns.boxplot(data=df.Miles)
sns.boxplot(data=df)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.svm import LinearSVC

from sklearn.metrics import mean_squared_error

from math import sqrt
X= df[['Fitness','Usage']]

y = df['Miles']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
lr = LinearRegression()

lr.fit(X_train,y_train)
print(lr.coef_)

print(lr.intercept_)