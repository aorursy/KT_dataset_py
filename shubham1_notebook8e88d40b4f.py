# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn import preprocessing

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.metrics import accuracy_score

from sklearn import tree

# Any results you write to the current directory are saved as output.
# Reading the csv file

full_data = pd.read_csv('../input/HR_comma_sep.csv')



# Encoding the salary into numeric data

full_data.ix[full_data.salary == 'low', 'salary'] = 1

full_data.ix[full_data.salary == 'medium', 'salary'] = 2

full_data.ix[full_data.salary == 'high', 'salary'] = 3

print('full_data\n',full_data.head(5))

# Using label Encoder to encode sales feature

le = preprocessing.LabelEncoder()

le.fit(full_data.sales)

print('Label encode classes -',le.classes_)

full_data.sales = le.transform(full_data.sales)

#y = pd.DataFrame(full_data['left'])

y = full_data['left']

X = full_data.drop('left', axis=1) 



#X = X.drop('sales', axis=1) 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print('\nTrain X, y shape:',X_train.shape,y_train.shape)

print('Test X, y shape:',X_test.shape,y_test.shape)

# Training Support Vector Classification classifier

clf = SVC()

clf.fit(X_train, y_train)

# Prediction here

y_pred = clf.predict(X_test)

# Calculate accuracy

accu = accuracy_score(pd.DataFrame(y_pred),pd.DataFrame(y_test))

print('SVM, accuracy -',"{0:.02f}%".format(accu*100))
#Decision Tree Classifier

clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)
# Prediction here

y_pred = clf.predict(X_test)

# Calculate accuracy

accu = accuracy_score(pd.DataFrame(y_pred),pd.DataFrame(y_test))

print('Decision tree, accuracy  -',"{0:.02f}%".format(accu*100))