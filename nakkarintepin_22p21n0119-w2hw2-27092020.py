# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.naive_bayes import GaussianNB

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train1 = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
train.info()
train.isnull().sum()
train_test_data = [train]

for dataset in train_test_data:
  dataset['Titel'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {'Mr':0, 'Miss':1, 'Mrs':2,
                 'Master':3, 'Dr':3, 'Rev':3, 'Col':3, 'Major':3, 'Mlle':3, "Countess":3,
                 'Ms':3, 'Lady':3, 'Jonkheer':3, 'Don':3, 'Dona':3, 'Mme':3, 'Capt':3, 'Sir':3}
for dataset in train_test_data:
  dataset['Titel'] = dataset['Titel'].map(title_mapping)
train["Age"].fillna(train.groupby("Titel")["Age"].transform("median"), inplace = True)
facet = sns.FacetGrid(train, hue= 'Survived', aspect = 4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(Xlim =(0, train['Age'].max()))
facet.add_legend()
plt.show()
train.isnull().sum()
train.head()
sns.pairplot(train)
train = train[['Pclass', 'Age', 'SibSp','Titel']]
ytrain = train1['Survived'].values
x = np.array(train)
y = np.array(ytrain)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x, y)
predictions = knn.predict(x)
recall = recall_score(y,predictions)
precision = precision_score(y,predictions)
f_measure = f1_score(y,predictions)
print("Recall",recall)
print("Precision",precision)
print("F Measure",f_measure)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x,y)
predictions = nb.predict(x)
recall = recall_score(y,predictions)
precision = precision_score(y,predictions)
f_measure = f1_score(y,predictions)
print("Recall",recall)
print("Precision",precision)
print("F Measure",f_measure)
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier()
MLP.fit(x, y)
predictions = MLP.predict(x)
recall = recall_score(y,predictions)
precision = precision_score(y,predictions)
f_measure = f1_score(y,predictions)
print("Recall",recall)
print("Precision",precision)
print("F Measure",f_measure)
