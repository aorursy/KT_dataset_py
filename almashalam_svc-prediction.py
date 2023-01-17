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

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')
gender_submission
train
test
sex_and_embark_train = pd.get_dummies(train[['Sex','Embarked']])



sex_and_embark_test = pd.get_dummies(test[['Sex','Embarked']])
train = train.drop(['Sex','Embarked'], axis=1)



test = test.drop(['Sex','Embarked'], axis=1)
train = train.join(sex_and_embark_train)



test = test.join(sex_and_embark_test)
train.fillna((train.mean()), inplace=True)
train
test.fillna((test.mean()), inplace=True)
test
X_train = train.drop(['Survived','Name','Ticket','Cabin'], axis=1)



y_train = train['Survived']
X_test = test.drop(['Name','Ticket','Cabin'], axis=1)



y_test = gender_submission['Survived']
X_train.info()
X_test.info()
y_test
y_train
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
steps = [('scaler', StandardScaler()),

         ('SVM', SVC())]



pipeline = Pipeline(steps)



parameters = {'SVM__C':[1, 10, 100],

              'SVM__gamma':[0.1, 0.01]}



cv = GridSearchCV(pipeline, parameters, cv=3)



cv.fit(X_train, y_train)



y_pred = cv.predict(X_test)



print("Accuracy: {}".format(cv.score(X_test, y_test)))



print(classification_report(y_test, y_pred))



print("Tuned Model Parameters: {}".format(cv.best_params_))
my_submission = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': y_pred})
my_submission
my_submission.to_csv('submission.csv',header=True, index=False)