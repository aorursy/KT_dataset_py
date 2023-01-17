import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import *
from sklearn import linear_model
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import metrics
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#extracting only numeric features
num_train = train.select_dtypes(include=[np.number]).interpolate().dropna()
num_test = test.select_dtypes(include=[np.number]).interpolate().dropna()
y = num_train.SalePrice
num_train = num_train.drop(['Id', 'SalePrice'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(num_train, y, test_size=0.2)
def model(Xtrain,Xtest,ytrain,ytest,model_name):
 model_name.fit(Xtrain, ytrain)
 y_pred = model_name.predict(Xtest)
 mse = mean_squared_error(ytest, y_pred)
 rms = sqrt(mse)
 return(rms)
lm = LinearRegression()
model(X_train, X_test, y_train,y_test,lm)
est = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, loss='ls')
model(X_train, X_test, y_train,y_test,est)
xgb=XGBRegressor(learning_rate=0.1, n_estimators=1000)
model(X_train, X_test, y_train,y_test,xgb)
#combining train, test and dummy coding
dataset = pd.concat(objs=[train, test], axis=0)
train_objs_num = len(train)
dataset = pd.get_dummies(dataset)
dataset = dataset.interpolate().dropna()
dum_train = dataset[:train_objs_num]
dum_test = dataset[train_objs_num:] 
dum_test_final=dum_test.drop(['Id', 'SalePrice'], axis=1)
dum_y = dum_train.SalePrice
dum_X = dum_train.drop(['Id', 'SalePrice'], axis=1)
dum_X_train, dum_X_test, dum_y_train, dum_y_test = train_test_split(dum_X, dum_y, test_size=0.2)
lm = LinearRegression()
model(dum_X_train, dum_X_test, dum_y_train,dum_y_test,lm)
est = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, loss='ls')
model(dum_X_train, dum_X_test, dum_y_train,dum_y_test,est)
xgb=XGBRegressor(learning_rate=0.1, n_estimators=1000)
model(dum_X_train, dum_X_test, dum_y_train,dum_y_test,xgb)
#using xgb with dummy coding for submission
predicted = xgb.predict(dum_test_final)
submission = pd.DataFrame({'Id': dum_test.Id,'SalePrice': predicted})
submission.to_csv('submission.csv', index=False)
submission.head()
#feature elimination and taining the features
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
#lr =LogisticRegression(max_iter=200)
#rfe=RFE(lr,n_features_to_select=20,step=1)
#features=rfe.fit(dum_X,dum_y)
#print (features.n_features_)
#print (features.ranking_)
features=['BedroomAbvGr', 'BsmtExposure_No', 'BsmtFullBath', 'BsmtQual_Gd', 'ExterQual_TA', 'Fireplaces', 'Foundation_CBlock', 'GarageFinish_RFn', 'GarageType_Attchd', 'HalfBath', 'HouseStyle_1Story', 'KitchenQual_TA', 'LotConfig_Inside', 'LotShape_Reg', 'MasVnrType_None', 'OverallCond', 'OverallQual', 'RoofStyle_Gable', 'SaleCondition_Normal', 'TotRmsAbvGrd']   
fea_X_train, fea_X_test, fea_y_train, fea_y_test = train_test_split(dum_X[features], dum_y, test_size=0.2)
lm = LinearRegression()
model(fea_X_train, fea_X_test, fea_y_train, fea_y_test,lm)
est = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, loss='ls')
model(fea_X_train, fea_X_test, fea_y_train, fea_y_test,est)
xgb = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, loss='ls')
model(fea_X_train, fea_X_test, fea_y_train, fea_y_test,xgb)