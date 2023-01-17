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
filename = "/kaggle/input/uci-secom.csv"



semi = pd.read_csv(filename)

semi.head()
semi.isnull().sum().describe()
columns_to_remove = []

j = semi.isnull().sum()

for i in j.keys():

    if j[i] >= 900:

        print(i, j[i])

        columns_to_remove.append(i)

        

data = semi.drop(columns_to_remove, axis = 1)
from sklearn.impute import SimpleImputer 

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,IsolationForest

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, average_precision_score

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate
x = data.iloc[:, 1:-1]

y = data.iloc[:, -1]



imp = SimpleImputer()

x = imp.fit_transform(x)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 10)
print(x_train.shape)

print(x_test.shape)

print(data.shape)
sc = StandardScaler()

x_train_std = sc.fit_transform(x_train)

x_test_std = sc.transform(x_test)

## logistic regression, regularized





Cs = [(i+1) for i in range(10)]

f1 = []



for c in Cs:

    lr = LogisticRegression(C = c, solver = 'lbfgs', max_iter=400, penalty = 'l2')

    f1 = cross_val_score(lr, x_train_std, y_train, cv=5, scoring='f1_macro')

    print(c)

    print(f1.mean())
## logistic regression, regularized



## based on CV, C = 1

c = 1



lr = LogisticRegression(C = c, solver = 'lbfgs', max_iter=400, penalty = 'l2')

lr.fit(x_train_std, y_train)

y_pred_lr = lr.predict(x_test_std)

print('accuracy score', accuracy_score(y_test, y_pred_lr))

y_pred_lr = lr.predict(x_train_std)

cm = confusion_matrix(y_train, y_pred_lr)

print('confusion matrix_train {}'.format(cm))

y_pred_lr = lr.predict(x_test_std)

cm = confusion_matrix(y_test, y_pred_lr)

print('confusion matrix_test {}'.format(cm))

print('precision_score', average_precision_score(y_test, y_pred_lr))

print('recall_score', recall_score(y_test, y_pred_lr))

print('f1_score', f1_score(y_test, y_pred_lr))

#f1.append(f1_score(y_test, y_pred_lr))



#index = [i for i in range(len(Cs))]

#plt.plot(index, f1, 'bs')
## svm, regularized

from sklearn.svm import SVC



Cs = [(i+1)/10 for i in range(10)]

f1 = []



for c in Cs:

    clf_svm = SVC(C = c, kernel='linear', gamma='scale')

    print(c)

    f1 = cross_val_score(clf_svm, x_train_std, y_train, cv=5, scoring='f1_macro')

    print(f1.mean())
# check models on test data

clf_svm = SVC(C = 0.6, kernel='linear', degree=1, gamma='scale')

clf_svm.fit(x_train_std, y_train)

y_test_pred = clf_svm.predict(x_test_std)

print('confusion_matrix', confusion_matrix(y_test, y_test_pred))

print('f1_score',f1_score(y_test, y_test_pred))
## logistic regression and SVM show similar accuracy. SVM is slightly better