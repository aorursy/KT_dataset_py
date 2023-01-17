import os
print(os.listdir('../input/gt192-1'))
import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv('../input/gt192-1/train.csv')

data_x= pd.read_csv('../input/gt192-1/test.csv')
data.info()
data.head()
data.describe(include ='all').T
data.columns
import seaborn as sns
import xgboost as xgb
model = xgb.XGBRegressor(max_depth=6, n_estimators=1000)
model.fit(data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad','tax', 'ptratio', 'b', 'lstat']], data.medv)


y_pred = model.predict(data_x[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad',

       'tax', 'ptratio', 'b', 'lstat']])
y_pred
sub = pd.DataFrame()

sub[0] = data_x.id

sub[1] = y_pred



sub.columns = ['id','medv']
sub.head()
sub.to_csv('submission.csv', index=False)