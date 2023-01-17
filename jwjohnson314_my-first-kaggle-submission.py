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
train = pd.read_csv('/kaggle/input/predict-customer-churn/train.csv')

test = pd.read_csv('/kaggle/input/predict-customer-churn/test.csv')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, accuracy_score
X = train.drop(['RowNumber', 'Exited', 'Surname'], axis=1)

y = train['Exited']



X_w_indicators = pd.get_dummies(X)



Xtr, Xva, ytr, yva = train_test_split(X_w_indicators, y, test_size=0.2, random_state=0)
model = LogisticRegression().fit(Xtr, ytr)





pred = model.predict(Xva)

print(accuracy_score(yva, pred))
probs = model.predict_proba(Xva)[:,1]

print(log_loss(yva, probs))
X_test = pd.get_dummies(test.drop(['RowNumber', 'Surname'], axis=1))

test_probs = model.predict_proba(X_test)[:,1]



# make submission csv

submission = pd.DataFrame({'Id': test['RowNumber'], 'Predicted': test_probs})

submission.head()
submission.to_csv('my_first_submission.csv', index=False)