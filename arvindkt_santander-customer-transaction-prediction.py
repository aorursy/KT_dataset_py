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
dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

dataset.head()
X_train = dataset.iloc[:, 2:].values

y_train = dataset.iloc[:,1:2]
dataset['target'].value_counts()
## Randomforest

"""from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,

                                    criterion = 'entropy')

classifier.fit(X_train, y_train)

test_data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

X_test = test_data.iloc[:,1:].values

y_pred = classifier.predict(X_test)

submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

submission.drop('target',inplace=True, axis = 1)

submission['target'] = y_pred

submission.to_csv('submission.csv', index = False)

"""
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting XGBoost to the Training set

from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
test_data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

X_test = test_data.iloc[:,1:].values

y_pred = classifier.predict(X_test)
submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

submission.drop('target',inplace=True, axis = 1)

submission['target'] = y_pred
submission.to_csv('submission.csv', index = False)