import pandas as pd

import numpy as np

from sklearn.cross_validation import train_test_split

np.set_printoptions(threshold=np.inf)
# Data Import

train = pd.read_csv("../input/train_2016_v2.csv")   

props = pd.read_csv("../input/properties_2016.csv") 

samp = pd.read_csv("../input/sample_submission.csv") 
train = train.drop(["transactiondate"], axis=1)
DF = pd.merge(train, props, how='left', on='parcelid')
X = DF.drop(['logerror'],axis=1,inplace=False)

X = X.select_dtypes(exclude=[object]) 

y = DF['logerror']
X = X.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)
## xgboost

from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X, y)
model.score(X_test, y_test)
## random forest

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X, y)
forest.score(X_test, y_test)
## ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
ext = ExtraTreesRegressor()
ext.fit(X, y)
ext.score(X_test, y_test)
# TEST data

test = samp.loc[:,['ParcelId']].merge(props,how='left',left_on='ParcelId',right_on='parcelid')

test_x = test.drop(['ParcelId'],axis=1,inplace=False)
test_x = test_x.select_dtypes(exclude=[object]) 
test_x = test_x.fillna(0)
test_y = forest.predict(test_x)

test_y = pd.DataFrame(test_y)
test_y[1] = test_y[0]

test_y[2] = test_y[0]

test_y[3] = test_y[0]

test_y[4] = test_y[0]

test_y[5] = test_y[0]  

test_y[6] = test_y[0]  

test_y[0] = samp["ParcelId"]

test_y.columns = ["ParcelId", "201610","201611","201612","201710","201711","201712"]
# output to csv

test_y.to_csv("result_forest.csv",index=False)