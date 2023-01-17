# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
wine=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
wine.head(2)
wine.isnull().sum()
wine.shape
wine.info()
sns.swarmplot(x='quality',y='fixed acidity',data=wine)
plt.figure(figsize=(15,15))

sns.boxplot(x='quality',y='volatile acidity',data=wine)
sns.barplot(x = 'quality', y = 'alcohol', data = wine)
sns.relplot(y='residual sugar',x='quality',data=wine)
sns.barplot(x='quality',y='chlorides',data=wine)
sns.boxplot(x='quality',y='total sulfur dioxide',data=wine)
sns.boxplot(x='quality',y='density',data=wine)
wine.head(1)
sns.relplot(y='pH',x='quality',data=wine)
sns.boxplot(x='quality',y='alcohol',data=wine)
sns.countplot(wine['quality'])
wine.quality.value_counts()
bins=(3,6.1,8)

wine_quality=['bad','good']

wine['quality']=pd.cut(wine['quality'],bins=bins,labels=wine_quality)

wine.head(1)
wine.quality.value_counts()
sns.countplot(wine['quality'])
wine = wine.dropna()
wine.isnull().sum()
from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()

X = wine.drop('quality', axis = 1)

y = le.fit_transform(wine['quality'])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='gini',splitter='best')

print(dtc.fit(X_train,y_train))

print(dtc.fit(X_test,y_test))

print(dtc.score(X_test,y_test))

y_predict=dtc.predict(X_test)

print('y prediction',y_predict)
#dicsion tree implement

from sklearn.metrics import classification_report,accuracy_score

print(classification_report(y_test,y_predict))
accuracy_score(y_test,y_predict)