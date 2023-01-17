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
import matplotlib.pyplot as plt

%matplotlib inline

plt.rc("font", size=14)

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

from scipy import stats

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.metrics import accuracy_score, confusion_matrix
train = pd.read_csv('../input/train.csv')

print(train.shape)

train.head()
train.columns
train.info()
train.dtypes
train.target = train.target.astype('category')
# check for missing values

train.isnull().any().sum()
train.target.value_counts()
# train-test split

X = train.drop(['ID_code','target'], axis=1)

Y = train['target']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=7)
train_Y.value_counts()
sc_X = StandardScaler()



train_X = sc_X.fit_transform(train_X)

test_X = sc_X.fit_transform(test_X)
pd.DataFrame(train_X).head()
ad = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=0.7, 

                        algorithm='SAMME.R', random_state=7).fit(train_X, train_Y)
print('Train Accuracy:' , accuracy_score(train_Y, ad.predict(train_X) ))

print('Test Accuracy:' , accuracy_score(test_Y, ad.predict(test_X) ))



print('Confusion matrix for Train: \n' , confusion_matrix(train_Y, ad.predict(train_X)))

print('Confusion matrix for Test: \n' , confusion_matrix(test_Y, ad.predict(test_X) ))
test = pd.read_csv('../input/test.csv')

print(test.shape)

test.head()
test_X = sc_X.fit_transform(test.drop('ID_code',axis=1))
pd.DataFrame(test_X).head()
submission = pd.DataFrame(test['ID_code'],columns=['ID_code'])

submission['target'] = ad.predict(test_X)

submission.head()
submission.target.value_counts()
submission.to_csv('submission.csv',index=False)