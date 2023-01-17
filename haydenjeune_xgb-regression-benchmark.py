import pandas as pd

import xgboost as xg

import numpy as np

import sklearn.model_selection as cv
# load files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.columns)
# extract target values

target = train['SalePrice'].values



# join datasets and convert categorical variables

both = train.append(test)

both = pd.get_dummies(both, dummy_na=True)



# remove non feature columns and resplit data

both = both.drop(['Id', 'SalePrice'], axis=1)

train_features = both.iloc[range(0,len(train))]

test_features = both.iloc[range(len(train),len(both))]
rg = xg.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=300)



scores = cv.cross_val_score(rg,train_features.values,target,cv=5)



print('Score: %0.2f +/- %0.2f' % (scores.mean(), scores.std()*2))



rg = rg.fit(train_features.values,target)
# make and output prediction

prediction = rg.predict(test_features.values)

output = pd.DataFrame({'Id':test['Id'].values,'SalePrice':prediction})

output.to_csv('output.csv',index=False)