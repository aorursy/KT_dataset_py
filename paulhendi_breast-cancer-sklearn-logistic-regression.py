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
breast_cancer = '../input/duke-breast-cancer.txt'
data = pd.read_table(breast_cancer, header=None)
data.head()
data.info()
print("Nb of Nan in the whole dataset : %d\n" %np.sum(data.isnull().sum())) # => No Nan
print("Nb of duplicated features in the whole dataset : %d" %(data.transpose().drop_duplicates().transpose().shape[1] - data.shape[1])) # => No duplicated features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

X = data.drop(data.columns[0], axis = 1)
y = data[data.columns[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)

clf = LogisticRegression()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)


print("Actual breast cancer : ")
print(y_test.values)

print("\nPredicted breast cancer : ")
print(y_pred)

print("\nAccuracy score : %f" %(accuracy_score(y_test, y_pred) * 100))
print("Recall score : %f" %(recall_score(y_test, y_pred) * 100))
print("ROC score : %f\n" %(roc_auc_score(y_test, y_pred) * 100))
print(confusion_matrix(y_test, y_pred)) 