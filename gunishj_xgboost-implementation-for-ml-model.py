import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
data.dropna(axis=0,subset=['SalePrice'],inplace=True)

y = data.SalePrice

X = data.drop(['SalePrice'],axis =1 ).select_dtypes(exclude =['object'])

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.25)
X_test = test.select_dtypes(exclude=['object'])

X_test.head()
my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(train_X, train_y, verbose=True)

mean_absolute_error(my_model.predict(test_X),test_y)

test_preds = my_model.predict(X_test)

output = pd.DataFrame({'Id':test.Id,'SalePrice': test_preds})

output.to_csv('submission.csv',index=False)
my_model1 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model1.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)

mean_absolute_error(my_model1.predict(test_X),test_y)

test_preds1 = my_model1.predict(X_test)

output = pd.DataFrame({'Id':test.Id,'SalePrice': test_preds1})

output.to_csv('submission1.csv',index=False)
pd.read_csv('submission1.csv').head()
output.head()