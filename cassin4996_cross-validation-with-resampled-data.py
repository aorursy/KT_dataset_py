# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from collections import Counter

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.Class.value_counts()
y = df.Class

X = df.iloc[:,:-1]
Counter(y)
rus = RandomUnderSampler(random_state = 42)

X_res,y_res = rus.fit_resample(X,y)

print('Resampled dataset target shape : {}'.format(Counter(y_res)))
X = pd.DataFrame(X_res)

y = pd.DataFrame(y_res)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 40)

clf = LogisticRegression()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print('Accuracy Score for Logistic Regression {}%'.format(round(accuracy_score(y_test,y_pred)*100,2)))
score = cross_val_score(clf,X,y, cv = 10)

score
print('K-fold Accuracy score: {}%'.format(round(score.mean()*100,2)))
accuracy = []

skf = StratifiedKFold(n_splits = 10, random_state = None)

skf.get_n_splits(X,y)



for train_index, test_index in skf.split(X,y):

    print('Train: ', train_index, 'Test: ', test_index)

    X1_train,X1_test = X.iloc[train_index,:],X.iloc[test_index,:]

    y1_train, y1_test = y.iloc[train_index,:],y.iloc[test_index,:]

    clf.fit(X1_train,y1_train)

    y1_pred = clf.predict(X1_test)

    score = accuracy_score(y1_pred,y1_test)

    accuracy.append(score)



print(accuracy)
accuracy = np.array(accuracy)

print('Accuracy score of Stratified K-fold: {}%'.format(round(accuracy.mean()*100,2)))
y = df.Class

X = df.iloc[:,:-1]

sm = SMOTE(random_state = 42)

X_res, Y_res = sm.fit_resample(X,y)

print('Resampled dataset target shape: {}'.format(Counter(Y_res)))
X = pd.DataFrame(X_res)

y = pd.DataFrame(Y_res)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 55)

clf = LogisticRegression()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print('Accuracy score: {}%'.format(round(accuracy_score(y_pred,y_test)*100,2)))
score = cross_val_score(clf,X,y, cv = 10)

score = score.mean()

print('K-fold accuracy score: {}%'.format(round(score*100,2)))
accuracy = []

skf = StratifiedKFold(n_splits = 10, random_state = None)



for train_index, test_index in skf.split(X,y):

    print('Train: ',train_index, 'Test: ',test_index)

    X1_train, X1_test = X.iloc[train_index,:], X.iloc[test_index,:]

    y1_train, y1_test = y.iloc[train_index,:], y.iloc[test_index,:]

    clf.fit(X1_train, y1_train)

    y1_pred = clf.predict(X1_test)

    score = accuracy_score(y1_pred, y1_test)

    accuracy.append(score)

print(accuracy)
accuracy = np.array(accuracy)

print('Accuracy Score of Stratified K-Fold : {}%'.format(round(accuracy.mean()*100,2)))