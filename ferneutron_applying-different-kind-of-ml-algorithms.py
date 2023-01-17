import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import Imputer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_float = train.select_dtypes(exclude=['object']).copy()
test_float = test.select_dtypes(exclude=['object']).copy()
train_category = train.select_dtypes(include=['object']).copy()
test_category = test.select_dtypes(include=['object']).copy()
Y = train_float.SalePrice
train_float = train_float.drop(['Id','SalePrice'],axis=1)
test_float = test_float.drop(['Id'],axis=1)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train_float)
Xtrain_float = imp.transform(train_float)
Xtest_float = imp.transform(test_float)
cols_with_missing = [col for col in train_category.columns 
                                 if train_category[col].isnull().any()]
Xtrain_category = train_category.drop(cols_with_missing, axis=1)
Xtest_category  = test_category.drop(cols_with_missing, axis=1)
Xtrain_category.columns
Xtest_category.columns
encoder = ce.BackwardDifferenceEncoder(cols=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating',
       'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive',
       'SaleType', 'SaleCondition'])
Xtrain_encoded = encoder.fit_transform(Xtrain_category)
Xtest_encoded = encoder.transform(Xtest_category)
Xtrain_float = pd.DataFrame(Xtrain_float, columns=train_float.columns)
Xtest_float = pd.DataFrame(Xtest_float, columns=test_float.columns)
min_max_scaler = preprocessing.MinMaxScaler()
Xtrain_float = pd.DataFrame(min_max_scaler.fit_transform(Xtrain_float), columns = Xtrain_float.columns)
Xtest_float = pd.DataFrame(min_max_scaler.transform(Xtest_float), columns=Xtest_float.columns)
Xtrain = pd.concat([Xtrain_float, Xtrain_encoded], axis=1)
Xtest = pd.concat([Xtest_float, Xtest_encoded], axis=1)
Xtest.head()
Xtest = Xtest.drop(['col_Functional_7', 'col_SaleType_9', 'col_KitchenQual_4', 'col_MSZoning_5'], axis=1)
Xtrain = Xtrain.drop(['col_RoofMatl_4', 'col_HouseStyle_7', 'col_Exterior1st_14', 
                      'col_Heating_4', 'col_Heating_5', 'col_RoofMatl_6', 'col_Condition2_7', 
                      'col_RoofMatl_7', 'col_RoofMatl_5', 'col_Condition2_6', 'col_Condition2_5'], axis=1)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', learning_rate=0.1, max_depth=3, n_estimators=200)
xg_reg.fit(Xtrain ,Y)
Ypredict = xg_reg.predict(Xtest)
test = pd.read_csv('../input/test.csv')
Ypred = pd.DataFrame({'SalePrice':Ypredict})
prediction = pd.concat([test['Id'], Ypred], axis=1)
prediction.to_csv('predictions_new_2.csv', sep=',', index=False)
prediction.head()
clf = SVR(C=1, epsilon=0.2, kernel='rbf')
clf.fit(Xtrain, Y)
Ypredict = clf.predict(Xtest)
test = pd.read_csv('../input/test.csv')
Ypred = pd.DataFrame({'SalePrice':Ypredict})
prediction = pd.concat([test['Id'], Ypred], axis=1)
prediction.to_csv('predictions_svmr.csv', sep=',', index=False)
prediction.head()
regr = RandomForestRegressor(n_estimators=100)
regr.fit(Xtrain, Y)
Ypredict = regr.predict(Xtest)
test = pd.read_csv('../input/test.csv')
Ypred = pd.DataFrame({'SalePrice':Ypredict})
prediction = pd.concat([test['Id'], Ypred], axis=1)
prediction.to_csv('predictions_randomforest.csv', sep=',', index=False)
prediction.head()
ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.1)
ada.fit(Xtrain, Y)
Ypredict = ada.predict(Xtest)
test = pd.read_csv('../input/test.csv')
Ypred = pd.DataFrame({'SalePrice':Ypredict})
prediction = pd.concat([test['Id'], Ypred], axis=1)
prediction.to_csv('predictions_adaBoost.csv', sep=',', index=False)
prediction.head()
gbt = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=2)
gbt.fit(Xtrain, Y)
Ypredict = gbt.predict(Xtest)
test = pd.read_csv('../input/test.csv')
Ypred = pd.DataFrame({'SalePrice':Ypredict})
prediction = pd.concat([test['Id'], Ypred], axis=1)
prediction.to_csv('predictions_GradientBoostingTree.csv', sep=',', index=False)
prediction.head()
