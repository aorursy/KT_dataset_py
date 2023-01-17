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
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

train.head()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X_train = train[['LotFrontage','LotArea']].fillna(0)
X_test  = test[['LotFrontage','LotArea']].fillna(0)
y_train = train['SalePrice']

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train.astype(float))
X_test_scaled  = X_scaler.transform(X_test.astype(float))
y_train_scaled = y_scaler.fit_transform(y_train.astype(float).values.reshape(-1, 1))

lr = LinearRegression()

lr.fit(X_train_scaled, y_train_scaled);

preds = lr.predict(X_test_scaled)

preds = y_scaler.inverse_transform(preds).reshape(-1)

pd.read_csv('../input/sample_sumbission.csv').head()

sub = pd.DataFrame({'Id': test.Id, 'SalePrice': preds})
# sub.to_csv('submission.csv', index=False)

sub.head()

sub = pd.DataFrame({'Id': test.Id, 'SalePrice': np.full((test.shape[0]), y_train.mean())})
sub.to_csv('submission.csv', index=False)