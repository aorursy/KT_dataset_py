import numpy as np

import pandas as pd

import xgboost

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score



train = pd.read_csv('../input/kc_house_data.csv')

train.drop('id', axis=1, inplace=True)

train.drop('date', axis=1, inplace=True)

traindf, testdf = train_test_split(train, test_size = 0.3)



xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

xgb.fit(traindf.ix[:, traindf.columns != 'price'], traindf['price'])

print(explained_variance_score(xgb.predict(testdf.ix[:, testdf.columns != 'price']), testdf['price']))