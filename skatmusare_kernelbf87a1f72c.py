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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
TrainDf=pd.read_csv('../input/creditcard.csv')
TrainDf.head(5)
X = TrainDf.iloc[:, TrainDf.columns != 'Class']

y = TrainDf.iloc[:, TrainDf.columns == 'Class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, 

                                                    random_state = 45)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

X_test_res, y_test_res = sm.fit_sample(X_test, y_test)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
models = []

acc = []

precision = []

recall = []

f1 = []
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'entropy', 

                            max_depth= 3,

                            random_state = 2)

dt.fit(X_train_res, y_train_res)
print('Confusion Matrix for DTrees: \n',confusion_matrix(y_test_res, dt.predict(X_test_res)))

print('Accuracy for DTrees: \n',accuracy_score(y_test_res, dt.predict(X_test_res)))

acc.append(accuracy_score(y_test_res, dt.predict(X_test_res)))

print('Precision for DTrees: \n',precision_score(y_test_res, dt.predict(X_test_res)))

precision.append(precision_score(y_test_res, dt.predict(X_test_res)))

print('Recall for DTrees: \n',recall_score(y_test_res, dt.predict(X_test_res)))

recall.append(recall_score(y_test_res, dt.predict(X_test_res)))

print('f1_score for DTrees: \n',f1_score(y_test_res, dt.predict(X_test_res)))

f1.append(f1_score(y_test_res, dt.predict(X_test_res)))
from sklearn.ensemble import AdaBoostClassifier

adb = AdaBoostClassifier(base_estimator=dt, n_estimators=31, 

                         algorithm='SAMME', random_state=40)

adb.fit(X_train_res, y_train_res)
print('Confusion Matrix for RF: \n  ',confusion_matrix(y_test_res, adb.predict(X_test_res)))

print('Accuracy for RF: \n',accuracy_score(y_test_res, adb.predict(X_test_res)))

acc.append(accuracy_score(y_test_res, adb.predict(X_test_res)))

print('Precision for RF: \n',precision_score(y_test_res, adb.predict(X_test_res)))

precision.append(precision_score(y_test_res, adb.predict(X_test_res)))

print('Recall for RF: \n',recall_score(y_test_res,adb.predict(X_test_res)))

recall.append(recall_score(y_test_res, adb.predict(X_test_res)))

print('f1_score for RF: \n',f1_score(y_test_res, adb.predict(X_test_res)))

f1.append(f1_score(y_test_res, adb.predict(X_test_res)))