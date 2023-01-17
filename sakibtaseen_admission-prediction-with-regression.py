# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualization library

import seaborn as sns #data visualization library

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ad= pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
ad.head()
ad.columns
ad.info()
sns.countplot(x='Research', data=ad)
sns.boxplot(x='Research', y='Chance of Admit ', data=ad)
sns.boxplot(x='University Rating', y='Chance of Admit ', data=ad)
plt.figure(figsize=(10,8))

sns.scatterplot(x='GRE Score', y='TOEFL Score', size='Chance of Admit ', data=ad)
sns.countplot(x='SOP', data=ad)
plt.figure(figsize=(12,8))

sns.boxplot(x='SOP', y='Chance of Admit ', data=ad)
sns.countplot(x='LOR ', data=ad)
plt.figure(figsize=(12,8))

sns.boxplot(x='LOR ', y='Chance of Admit ', data=ad)
plt.figure(figsize=(8,8))

sns.scatterplot(x='CGPA', y='Chance of Admit ', data=ad)
plt.figure(figsize=(7,6))

sns.heatmap(ad.corr(),annot=True, cmap='magma')
from sklearn.model_selection import train_test_split
ad.drop('Serial No.', axis=1, inplace=True)
X= ad.drop('Chance of Admit ', axis=1)

y=ad['Chance of Admit ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=105)
from sklearn.linear_model import LinearRegression
lreg= LinearRegression()
lreg.fit(X_train, y_train)
pred= lreg.predict(X_test)
pred[:10]
y_test.head(10)
from sklearn import metrics
print('Mean Squared Error: ',metrics.mean_squared_error(y_test,pred))

print('Root Mean Squared Error: ',np.sqrt(metrics.mean_squared_error(y_test,pred)))