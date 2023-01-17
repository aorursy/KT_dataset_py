import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('../input/train.csv', index_col='Id')
test = pd.read_csv('../input/test.csv', index_col='Id')
target = train['SalePrice']  #target variable
train = train.drop('SalePrice', axis=1)
train['training_set'] = True
test['training_set'] = False
full = pd.concat([train, test])
full = full.interpolate()
full = pd.get_dummies(full)
train = full[full['training_set']==True]
train = train.drop('training_set', axis=1)
test = full[full['training_set']==False]
test = test.drop('training_set', axis=1)

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(train, target)
preds = rf.predict(test)
my_submission = pd.DataFrame({'Id': test.index, 'SalePrice': preds})
my_submission.head(10000)

my_submission.to_csv('submission.csv',index=False)