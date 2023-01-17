import os
print((os.listdir('../input/')))
import xgboost as xgb
from sklearn.metrics import mean_squared_error

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from math import log as log
df_train = pd.read_csv('../input/web-club-recruitment-2018/train.csv')
df_test = pd.read_csv('../input/web-club-recruitment-2018/test.csv')
import math
df_train.head()
def fn(X):
#     X.drop(X.columns[17:23], axis=1, inplace=True)
    X['X24']=(X.X6+X.X7+X.X8+X.X9+X.X10+X.X11)/X.X1
    X['X25']=(X.X12+X.X13+X.X14+X.X15+X.X16+X.X17)/X.X1
    X['X26']=(X.X18+X.X19+X.X20+X.X21+X.X22+X.X23)/X.X1
    X['X28']=(X.X6+X.X7+X.X8+X.X9+X.X10+X.X11)
    X['X29']=(X.X12+X.X13+X.X14+X.X15+X.X16+X.X17)
    X['X30']=(X.X18+X.X19+X.X20+X.X21+X.X22+X.X23)
    X['X31']=(X.X2==-1)*1+(X.X3==-1)*1+(X.X4==-1)*1+(X.X5==-1)*1
#     X['X27']=((X.X5>=20)) * X.X5
#     X['X27']=((X.X27<=50)) * X.X27
#     X['X28']=X.X1+X.X2+X.X3+X.X4+X.X5
#     X.drop(X.columns[0:0], axis=1, inplace=True)
    print(X.head())
    return X
train_X = df_train.loc[0:19999, 'X1':'X23']
train_X = fn(train_X)
# train_X.drop(train_X.columns[17:23], axis=1, inplace=True)
print(train_X.head())
train_y = df_train.loc[0:19999, 'Y']
data_dmatrix = xgb.DMatrix(data=train_X,label=train_y)
# rf = RandomForestClassifier(n_estimators=300)
xg_reg = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.024,
                max_depth = 6, alpha = 10, n_estimators = 230,eval_metric='auc')
# rf.fit(train_X, train_y)
xg_reg.fit(train_X,train_y)
# xg_reg.set_param(learning_rate=0.005)
train_X = df_test.loc[:, 'X1':'X23']
train_X = fn(train_X)
# train_X.drop(train_X.columns[17:23], axis=1, inplace=True)
# pred = rf.predict_proba(train_X)
pred = xg_reg.predict(train_X)
pred=pred*(pred>=0)
pred= (pred*(pred<=1))+ 1*(pred>1)
result = pd.DataFrame(pred)
result.index.name = 'id'
result.columns = ['predicted_val']
result.to_csv('output.csv', index=True)
X = df_train.loc[18000:19999, 'X1':'X23']
X = fn(X)
# X.drop(X.columns[17:23], axis=1, inplace=True)
Y = df_train.loc[18000:19999, 'Y']
# pred = rf.predict_proba(X)
pred = xg_reg.predict(X)
pred=pred*(pred>=0)
pred= (pred*(pred<=1))+ 1*(pred>1)
# print (pred)
print(roc_auc_score(Y, pred))
