import numpy as np

import pandas as pd

import category_encoders as ce

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
train.sort_index(inplace=True)

y_train = train['target']

x_train = train.drop(['target', 'id'], axis=1)
test_id = test['id']

test.drop('id', axis=1, inplace=True)
for col in x_train.columns:

    if x_train[col].isna().sum()>0:

        x_train.loc[x_train[col].isna(),col] = x_train[col][-x_train[col].isna()].sample(n= x_train[col].isna().sum()).values
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)
X_train.isna().sum()
high_dim_cols=[]

low_dim_cols=[]



for col in X_train.columns:

    if X_train[col].nunique()<20:

        low_dim_cols.append(col)



for col in X_train.columns:

    if X_train[col].nunique()>20:

        high_dim_cols.append(col)
print(low_dim_cols)

print(high_dim_cols)
ohEncoder = ce.OneHotEncoder(cols=low_dim_cols)

X_train = ohEncoder.fit_transform(X_train)

X_test = ohEncoder.transform(X_test)
hashingEncoder = ce.HashingEncoder(cols=high_dim_cols)

X_train = hashingEncoder.fit_transform(X_train)

X_test = hashingEncoder.transform(X_test)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=2020, fit_intercept=True, penalty='none', verbose=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_pred)
test = ohEncoder.transform(test)

test = hashingEncoder.transform(test)
pd.DataFrame({'id': test_id, 'target': clf.predict_proba(test)[:,1]}).to_csv('submission.csv', index=False)