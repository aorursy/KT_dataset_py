# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/iris/Iris.csv')
df.head(5)
df.info()
df.describe()
df.drop('Id',axis=1)
sns.scatterplot(data=df,x='SepalLengthCm',y='SepalWidthCm',hue='Species')

plt.title("Sepal Length VS Sepal Width")
sns.scatterplot(data=df,x='PetalLengthCm',y='PetalWidthCm',hue='Species')

plt.title("Petal Length VS Petal Width")
df1=df.drop('Id',axis=1)

sns.pairplot(df1,hue='Species')
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=df1)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=df1)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=df1)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=df1)
plt.figure(figsize=(10,8))

sns.heatmap(df1.corr(),annot=True,cmap="YlGnBu")
X = df.drop(['Id', 'Species'], axis=1)

y = df['Species']

# print(X.head())

print(X.shape)

# print(y.head())

print(y.shape)
from sklearn.model_selection import train_test_split  #to split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn import metrics # for checking the model accuracy

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

import warnings

warnings.filterwarnings("ignore")
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
Model = DecisionTreeClassifier()

Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
Model=RandomForestClassifier(max_depth=2)

Model.fit(X_train,y_train)

y_pred=Model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
Model = GaussianNB()

Model.fit(X_train, y_train)



y_pred = Model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))