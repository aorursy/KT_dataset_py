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
dataset = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

dataset_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

y = dataset.iloc[:,0].values.astype('int32')

X = dataset.iloc[:,1:].values.astype('float32')

test_X = dataset_test.values.astype('float32')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 40)
#standard feature scalling

from sklearn import preprocessing

normal = preprocessing.Normalizer().fit(X)

X = normal.transform(X)



normal = preprocessing.Normalizer().fit(test_X)

test_X = normal.transform(test_X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=42)

X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier
#Logistic Regression



# logreg = LogisticRegression()

# logreg.fit(X_train, y_train)

# Y_pred = logreg.predict(X_test)

# acc_log = logreg.score(X_train, y_train)

# acc_log
# Support Vector Machines



# svc = SVC()

# svc.fit(X_train, y_train)

# Y_pred = svc.predict(X_test)

# acc_svc = svc.score(X_train, y_train)

# acc_svc
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = random_forest.score(X_train, y_train)

acc_random_forest
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(test_X)

acc_random_forest = random_forest.score(X_train, y_train)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machines',

              'Random Forest'],

    'Score': [acc_log, acc_svc,

              acc_random_forest]})

models.sort_values(by='Score', ascending=False)
y_pred = random_forest.predict(test_X)

y_pred = pd.Series(y_pred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_pred],axis = 1)



submission.to_csv('sample_submission1.csv', index=False)