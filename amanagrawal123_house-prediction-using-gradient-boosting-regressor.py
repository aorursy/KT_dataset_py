import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.head()
label = train['SalePrice'].values
train = train.drop(['SalePrice','Id'],axis=1)
train.head()
null_train_col = train.columns[train.isnull().sum()>0]
print(null_train_col)
for col in null_train_col:
    print(col,train[col].isnull().sum(),train[col].dtype)
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)
train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
for col in null_train_col:
    train[col] = train[col].fillna("None")
train.isnull().sum().sum()==0
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
train['MSSubClass'] = train['MSSubClass'].astype(str)
num_col = [col for col in train.columns if train[col].dtype!=object]
obj_col = [col for col in train.columns if train[col].dtype==object]
train_obj_new = pd.get_dummies(train[obj_col])
train_obj_new.head(10)
feature = pd.concat([train[num_col],train_obj_new],axis=1)
feature.head(10)
gdb = GradientBoostingRegressor(n_estimators=331,learning_rate=0.1)
gdb.fit(feature,label)
print(gdb.score(feature,label))
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.head(10)
test = test.drop(['Id'],axis=1)
test['YrSold'] = test['YrSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)
test['MSSubClass'] = test['MSSubClass'].astype(str)
null_test_col = test.columns[test.isnull().sum()>0]
null_test_col
test['LotFrontage'] = test['LotFrontage'].fillna(train['LotFrontage'].mean())
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(0)
for col in null_test_col:
    if col in null_train_col:
        test[col] = test[col].fillna("None")
    else:
        test[col] = test[col].fillna(test[col].mode()[0])
test.isnull().sum().sum()==0
test_obj_new = pd.DataFrame(data=np.zeros((test.shape[0],train_obj_new.shape[1])),columns=train_obj_new.columns)
test_obj_new.head()
test_obj_prev = pd.get_dummies(test[obj_col])
for col in test_obj_prev:
    test_obj_new[col] = test_obj_prev[col]
test_obj_new = test_obj_new.drop(['MSSubClass_150'],axis=1)
test_obj_new.head()
predict_feature = pd.concat([test[num_col],test_obj_new],axis=1)
predict_feature.head()
y_pred = gdb.predict(predict_feature)
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission.head()
submission['SalePrice'] = y_pred
submission.head()
submission.to_csv('submission.csv',index=False)