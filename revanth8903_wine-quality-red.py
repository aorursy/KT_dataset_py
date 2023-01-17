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

data= pd.read_csv('../input/winequality-red.csv')

data.head()
data.info()
data.describe().transpose()
data.isnull().sum()
data['quality'].unique()
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(8,8))

sns.boxplot(x='quality', y='fixed acidity', data=data)
plt.figure(figsize=(8,8))

sns.barplot(x='quality', y='volatile acidity', data=data)

plt.title('Plotting quality vs volatile acidity')
plt.figure(figsize=(8,8))

sns.boxplot(x='quality', y='citric acid', data=data)

plt.title('Plotting quality vs citric acid')
plt.figure(figsize=(8,8))

sns.boxplot(x='quality', y='residual sugar', data=data)

plt.title('plotting quality vs residual sugar')
plt.figure(figsize=(8,8))

sns.boxplot(x='quality', y='chlorides', data=data)

plt.title('plotting quality vs chlorides')
plt.figure(figsize=(8,8))

sns.boxplot(x='quality', y='free sulfur dioxide', data=data)

plt.title('plotting quality vs free sulfur dioxide')
plt.figure(figsize=(8,8))

sns.barplot(x='quality', y='total sulfur dioxide', data=data)

plt.title('Plotting qulaity vs total sulfur dioxide')
plt.figure(figsize=(8,8))

sns.boxplot(x='quality', y='density', data=data)

plt.title('plotting quality vs density')
plt.figure(figsize=(8,8))

sns.boxplot(x='quality', y='pH', data=data)

plt.title('plotting quality vs pH')
plt.figure(figsize=(8,8))

sns.boxplot(x='quality', y='sulphates', data=data)

plt.title('plotting quality vs sulphates')
plt.figure(figsize=(8,8))

sns.boxplot(x='quality', y='alcohol', data=data)

plt.title('plotting quality vs alcohol')
a=data.corr()

a
plt.figure(figsize=(12,12))

sns.heatmap(a, annot=True)

plt.figure(figsize=(12,12))

sns.barplot(data=data)

plt.xticks(rotation=60)

plt.show()
from sklearn.preprocessing import minmax_scale

data[['total sulfur dioxide','free sulfur dioxide','fixed acidity']]=minmax_scale(data[['total sulfur dioxide','free sulfur dioxide','fixed acidity']])

data.head()
data[['residual sugar','pH','alcohol']]=minmax_scale(data[['residual sugar','pH','alcohol']])

data.head()
plt.figure(figsize=(12,12))

sns.barplot(data=data)

plt.xticks(rotation=60)

plt.show()
import pandas as pd

bins=(2, 6.5, 8)

group_names=['bad','good']

data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)

data.head()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data['quality']=le.fit_transform(data['quality'])

data.head()
data['quality'].value_counts()
sns.countplot(data['quality'])
a=data.drop('quality', axis=1)

b=data['quality']

a.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(a, b, test_size=0.2, random_state=3)

x_train.shape, y_train.shape, x_test.shape, y_test.shape
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=200)

rfc.fit(x_train, y_train)

pred_rfc=rfc.predict(x_test)

pred_rfc
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))
from sklearn.linear_model import SGDClassifier

sgd=SGDClassifier(penalty=None)

sgd.fit(x_train, y_train)

pred_sgd=sgd.predict(x_test)

pred_sgd
print(classification_report(y_test, pred_sgd))
print(confusion_matrix(y_test, pred_sgd))
from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train, y_train)

pred_svc=svc.predict(x_test)

pred_svc
print(classification_report(y_test, pred_sgd))
print(confusion_matrix(y_test, pred_svc))