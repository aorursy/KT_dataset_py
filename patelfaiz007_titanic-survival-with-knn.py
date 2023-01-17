# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
training_data = pd.read_csv('../input/train.csv', index_col = 0)
testing_data = pd.read_csv('../input/test.csv', index_col = 0)
print(training_data.shape)
print(testing_data.shape)
training_data.head()
testing_data.head()
training_data = training_data[['Name', 'Pclass', 'Sex', 'Age', 'Survived']]
print(training_data.shape)
print(training_data.isnull().sum())
sns.boxplot(data = training_data, x = 'Age')
training_data['Age'].fillna(int(training_data['Age'].mean()), inplace = True)
print(training_data.isnull().sum())
from sklearn.preprocessing import LabelEncoder
colname = ['Sex']

le = {}

for x in colname:
    le[x] = LabelEncoder()
for x in colname:
    training_data[x] = le[x].fit_transform(training_data[x])
for x in colname:
    testing_data[x] = le[x].fit_transform(testing_data[x])    
training_data.head()
X_train = training_data.values[:, 1:-1]
y_train = training_data.values[:, -1]

# Since we don't have Testing values we will use training data for testing.
X_test = training_data.values[:, 1:-1]
y_test = training_data.values[:, -1]
y_train = y_train.astype(int)
y_test = y_test.astype(int)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
model_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')

model_KNN.fit(X_train, y_train)

y_pred = model_KNN.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# Let's try different values of k
for K in range(15):
    Kvalue = K+1
    model_KNN = KNeighborsClassifier(Kvalue)
    model_KNN.fit(X_train, y_train) 
    y_pred = model_KNN.predict(X_test)
    print ("Accuracy is ", accuracy_score(y_test,y_pred), "for K-Value:",Kvalue)
