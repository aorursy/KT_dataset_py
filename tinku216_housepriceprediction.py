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
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
y_train = train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

X_train = train[predictor_cols]
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train=ss.fit_transform(X_train)
model = RandomForestRegressor()

model.fit(X_train, y_train)
test_val=test[predictor_cols]
test_val=ss.transform(test_val)
val=model.predict(test_val)
val
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': val})

my_submission.to_csv('submission.csv', index=False)