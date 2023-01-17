# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read training data

data = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')
# first few rows

data.head()
# null check

data.isna().values.any()
# extract labels

X = data

y = X.pop('Cover_Type')
# split training data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
lgr_simple = LogisticRegression()

lgr_simple.fit(X_train, y_train)

lgr_simple.score(X_test, y_test)
# we expect this to do better than the simpler

lgr = LogisticRegression(solver='newton-cg', multi_class='multinomial')

lgr.fit(X_train_scaled, y_train)

lgr.score(X_test_scaled, y_test)
lgr = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=50, max_iter=100)

lgr.fit(X_train_scaled, y_train)

lgr.score(X_test_scaled, y_test)
lgr_cv = LogisticRegression(multi_class='multinomial')

parameters = {'solver': ['lbfgs', 'newton-cg'], 'C': [ 0.1, 1, 10, 30, 50, 60, 70, 90]}

gscv = GridSearchCV(lgr_cv, parameters, cv=5)

gscv.fit(X_train_scaled, y_train)

print("Best Accuracy For Logistic Regression: .{:02}".format(gscv.best_score_))

# Params giving the best accuracy

gscv.best_estimator_
# Linear classifier didn't do very well earlier, so lets jump right into a complex non-linear kernel

svc = SVC(gamma='scale', kernel='rbf')

# select a sufficiently big range and hope that find optimal C lies in this range. If not, we will atleast get an idea about the right direction, i.e go higher or lower 

parameters= {'C': [ 0.01, 0.1, 1, 10, 100, 500, 800, 900, 1000, 1100, 3000]} 

gscv_svc = GridSearchCV(svc, parameters, cv=5)

gscv_svc.fit(X_train_scaled, y_train)

print("Best Accuracy : .{:02}".format(gscv_svc.best_score_))
gscv_svc.best_estimator_
best_svc = SVC(gamma='scale', kernel='rbf', C=900)

t_scaler = preprocessing.StandardScaler().fit(X)

X_scaled = t_scaler.transform(X)

best_svc.fit(X_scaled, y)
test_data = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')
test_scaled = t_scaler.transform(test_data)

predictions = best_svc.predict(test_scaled)
# write predictions to csv in specified format

output = pd.DataFrame({'ID': test_data.index,

                       'Cover_Type': predictions})



output.to_csv('submission.csv', index=False)