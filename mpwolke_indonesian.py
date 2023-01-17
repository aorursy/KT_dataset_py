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

import seaborn as sns



pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

%matplotlib inline
!ls '../input'

!ls '../input/hmif-data-science-bootcamp-2019'
train = pd.read_csv('../input/hmif-data-science-bootcamp-2019/train-data.csv')

test = pd.read_csv('../input/hmif-data-science-bootcamp-2019/test-data.csv')

sub = pd.read_csv('../input/hmif-data-science-bootcamp-2019/sample-submission.csv')
train.head()
train.columns
plt.title('Proporsi kelas')

sns.countplot(train['akreditasi'])

plt.show()
X = train.drop(['id', 'akreditasi'], axis=1)

y = train['akreditasi']

test = test.drop(['id'], axis=1)
X = pd.get_dummies(X)

test = pd.get_dummies(test)
dummy_absent = set(X.columns) - set(test.columns)

for col in dummy_absent:

    test[col] = 0

    

test = test[X.columns]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

print(classification_report(y_test, y_test_pred))

print('accuracy', accuracy_score(y_test, y_test_pred))

print('mae', mean_absolute_error(y_test, y_test_pred))
X_full = pd.concat([X_train, X_test])

y_full = pd.concat([y_train, y_test])
clf = LogisticRegression()

clf.fit(X_full, y_full)

test_data_pred = clf.predict(test)

test_data_pred[0:5]
sub['akreditasi'] = test_data_pred

sub.to_csv('baseline-submission.csv', index=False)