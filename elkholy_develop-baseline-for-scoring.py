sale_price='SalePrice'

ID='Id'
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyRegressor



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')



sample_submission=pd.read_csv('../input/sample_submission.csv')

test=pd.read_csv('../input/test.csv')
train_data, test_data = train_test_split(train, test_size=0.2,random_state=1234)

print(train_data.columns)
#dummy_solution=train_data[sale_price].mean()



X_train=train_data.drop(sale_price,axis=1)

y_train=train_data[sale_price]

dummy_regressor=DummyRegressor(strategy='mean')

dummy_regressor.fit(X_train,y_train)



X_test=train_data.drop(sale_price,axis=1)

y_test=train_data[sale_price]

y_pred=dummy_regressor.predict(X_test)

baseline_mae=sk.metrics.mean_absolute_error(y_test,y_pred)

print(baseline_mae)
test_pred=dummy_regressor.predict(test)

test[sale_price]=test_pred

test2=test[[ID,sale_price]]

test2.to_csv('submission.csv',index=False)

!pip install kaggle
!kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "trivial dummy submission to be used as baseline"
#print(train.head()

print(sample_submission.head())
dummy_price=train.mean()['SalePrice']