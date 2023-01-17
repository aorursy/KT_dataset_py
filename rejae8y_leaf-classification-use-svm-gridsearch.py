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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV

%matplotlib inline
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
def encode(train, test):

    le = LabelEncoder().fit(train.species) 

    labels = le.transform(train.species)           # encode species strings

    classes = list(le.classes_)                    # save column names for submission

    test_ids = test.id                             # save test ids for submission

    

    train = train.drop(['species', 'id'], axis=1)

    test = test.drop(['id'], axis=1)

    

    return train, labels, test, test_ids, classes



train, labels, test, test_ids, classes = encode(train, test)

train.head(1)
# from sklearn.svm import SVC

# C_range = np.logspace(-2, 10, 13)

# gamma_range = np.logspace(-9, 3, 13)

# param_grid = dict(gamma=gamma_range, C=C_range)

# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)

# grid.fit(train, labels)



# print("The best parameters are %s with a score of %0.2f"

#       % (grid.best_params_, grid.best_score_))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train.values, labels, test_size=0.20)

X_test.shape
y_test.shape
from sklearn.svm import SVC

model = SVC(C=100.0,gamma=1.0)

model.fit(X_train,y_train)

prediction=model.predict(X_test)

print('The accuracy of the DecisionTreeClassifier is',metrics.accuracy_score(prediction,y_test))
# Predict Test Set

favorite_clf = SVC(C=100.0,gamma=1.0,probability=True)

favorite_clf.fit(X_train, y_train)

test_predictions = favorite_clf.predict_proba(test)



# Format DataFrame

submission = pd.DataFrame(test_predictions, columns=classes)

submission.insert(0, 'id', test_ids)

submission.reset_index()



# Export Submission

#submission.to_csv('submission.csv', index = False)

submission.tail()