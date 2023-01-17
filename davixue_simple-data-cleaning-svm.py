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

# Importing the dataset
dataset = pd.read_csv('../input/train.csv')
datasetTest = pd.read_csv('../input/test.csv')


# Dealing with Objects in 'dependency'
dataset.dependency.value_counts()
# Calculating seperate dependency values
dataset['dependency_calculated'] = (dataset.hogar_nin + dataset.hogar_mayor)/(dataset.hogar_adul - dataset.hogar_mayor)
dataset[['dependency','dependency_calculated']]
# Replacing no, yes, inf
dataset.dependency.replace('no','0',inplace=True)
dataset.dependency.replace('yes','1',inplace=True)
dataset.dependency_calculated.replace(float('inf'),8,inplace=True)
# Set dataset as float
dataset.dependency = dataset.dependency.astype('float')
# Dropping calculated column
dataset.drop('dependency_calculated', axis=1, inplace=True)

# Dealing with Objects in 'dependency' in test data
datasetTest.dependency.replace('no','0',inplace=True)
datasetTest.dependency.replace('yes','1',inplace=True)
datasetTest.dependency = datasetTest.dependency.astype('float')

# Dealing with Objects in 'edjefe'
dataset.edjefe.value_counts()
dataset.edjefe.replace('no','0',inplace=True)
dataset.edjefe.replace('yes','1',inplace=True)
dataset.edjefe = dataset.edjefe.astype('float')
datasetTest.edjefe.replace('no','0',inplace=True)
datasetTest.edjefe.replace('yes','1',inplace=True)
datasetTest.edjefe = datasetTest.edjefe.astype('float')

# Dealing with Objects in 'edjefa'
dataset.edjefa.value_counts()
dataset.edjefa.replace('no','0',inplace=True)
dataset.edjefa.replace('yes','1',inplace=True)
dataset.edjefa = dataset.edjefa.astype('float')
datasetTest.edjefa.replace('no','0',inplace=True)
datasetTest.edjefa.replace('yes','1',inplace=True)
datasetTest.edjefa = datasetTest.edjefa.astype('float')

# Filling NaN columns with 0
col_fillna = ['v18q1', 'meaneduc', 'SQBmeaned']
dataset[col_fillna] = dataset[col_fillna].fillna(0)
datasetTest[col_fillna] = datasetTest[col_fillna].fillna(0)

# Excluding columns
col_exclude = ['Id','idhogar','v2a1','rez_esc']
dataset.drop(col_exclude, axis=1, inplace=True)
datasetTest.drop(col_exclude, axis=1, inplace=True)

# Selecting training set
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 138].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 'auto', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





# Selecting test set
X_sub = datasetTest.iloc[:, :].values
subTest = pd.read_csv('../input/test.csv')

# Feature Scaling
X_sub = sc.transform(X_sub)

# Predicting the Test set results
y_sub = classifier.predict(X_sub)

# Submission
subs = pd.DataFrame()
subs['Id'] = subTest['Id']
subs['Target'] = y_sub
subs.to_csv('submission.csv', index=False)
subs.head()