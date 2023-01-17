# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import os
print(os.listdir("../input"))
sns.set(style='whitegrid')
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Iris.csv')
df.head()
df.describe()
#distribution for speal-length
plt.subplots(figsize=(7,5))
sns.distplot(df['SepalLengthCm'], hist=False, rug=True, color='c')
#distribution of sepal-width
plt.subplots(figsize=(7,5))
sns.distplot(df['SepalWidthCm'], hist=False, rug=True, color='r')
#distribution of petal-length
plt.subplots(figsize=(7,5))
sns.distplot(df['PetalLengthCm'], hist=False, rug=True, color='m')
#distribution of petal-width
plt.subplots(figsize=(7,5))
sns.distplot(df['PetalWidthCm'], hist=False, rug=True, color='g')
df['Species'].value_counts()
df['Species'] = df['Species'].map({'Iris-versicolor': 'versicolor', 'Iris-virginica':'virginica', 'Iris-setosa':'setosa'})
#mapped the species from Iris-x to x, not needed but I did it anyway
df.head()
setosa = df[df['Species'] == 'setosa']            #make a dataframe setosa from df where species is setosa
virginica = df[df['Species'] == 'virginica']      #make a dataframe virginica from df where species is virginica
versicolor = df[df['Species'] == 'versicolor']    #make a dataframe versicolor from df where species is versicolor
sns.jointplot(setosa['SepalWidthCm'], setosa['SepalLengthCm'], kind='kde', color='b')
plt.title('Setosa')
sns.jointplot(versicolor['SepalWidthCm'], versicolor['SepalLengthCm'], kind='kde', color='c')
plt.title('Versicolor')
sns.jointplot(virginica['SepalWidthCm'], virginica['SepalLengthCm'], kind='kde', color='m')
plt.title('Virginica')
sns.jointplot(setosa['PetalWidthCm'], setosa['PetalLengthCm'], kind='kde', color='b')
plt.title('setosa')
sns.jointplot(versicolor['PetalWidthCm'], versicolor['PetalLengthCm'], kind='kde', color='c')
plt.title('versicolor')
sns.jointplot(virginica['PetalWidthCm'], virginica['PetalLengthCm'], kind='kde', color='m')
plt.title('virginica')
df.drop(columns=['Id'], inplace=True) #dropping id column beacuse it's not needed for pairplot and for future analysis
sns.pairplot(df, hue='Species')       #hue is species because we want see the species' distribution
#splitting the data 80-20 for training and test 
train, test = train_test_split(df, test_size=0.2)
print(train.shape)
print(test.shape)
X_train = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_train = train['Species']

X_test = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_test = test['Species']
print(X_train.head())
print(y_train.head())

print(X_test.head())
print(y_test.head())
model1 = LogisticRegression()
model1.fit(X_train, y_train)
prediction1 = model1.predict(X_test)

print('Accuracy of this model is :', metrics.accuracy_score(prediction1, y_test))
model2 = svm.SVC()
model2.fit(X_train, y_train)
prediction2 = model2.predict(X_test)

print('Accuracy of SVM model is :', metrics.accuracy_score(prediction2, y_test))
model3 = DecisionTreeClassifier()
model3.fit(X_train, y_train)
prediction3 = model3.predict(X_test)

print('Acuracy of Decision Tree model is :', metrics.accuracy_score(prediction3, y_test))
