import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Lasso

from sklearn.svm import  SVR

from sklearn.ensemble import IsolationForest
train_data_csv = pd.read_csv("../input/train.csv")

test_data_csv = pd.read_csv("../input/test.csv")
train_SalePrice = pd.DataFrame(train_data_csv["SalePrice"])
train_SalePrice.shape
train_SalePrice.head()
train_data = train_data_csv.drop(["SalePrice"],axis=1)
full_features = pd.concat([train_data,test_data_csv],ignore_index= True)
full_features.to_csv("full.csv")
full_features.info()
full_features = full_features.drop(["Id"],axis=1)
full_features.shape
full_features.isnull().sum().sort_values(ascending = False)
full_features.isnull().sum().sort_values(ascending = False)
full_features.loc[2576,'GarageArea'] = 0

full_features.loc[2576,'GarageCars'] = 0

full_features.loc[2120,'BsmtUnfSF'] = 0

full_features.loc[2120,'BsmtFinSF1'] = 0

full_features.loc[2120,'TotalBsmtSF'] = 0

full_features.loc[2120,'BsmtFinSF2'] = 0

full_features.loc[2120,'BsmtFullBath'] = 0

full_features.loc[2188,'BsmtFullBath'] = 0

full_features.loc[2120,'BsmtHalfBath'] = 0

full_features.loc[2188,'BsmtHalfBath'] = 0

full_features['Functional'] = full_features['Functional'].fillna('None')

full_features['Utilities'] = full_features['Utilities'].fillna('None')

full_features.loc[1555,'KitchenQual'] = 'None'

full_features.loc[2151,'Exterior1st'] = 'None'

full_features.loc[2151,'Exterior2nd'] = 'None'

full_features["Electrical"] = full_features["Electrical"].fillna('SBrkr')

full_features["MasVnrArea"] = full_features["MasVnrArea"].fillna(0)

full_features["MasVnrType"] = full_features["MasVnrType"].fillna('None')

full_features["GarageFinish"] = full_features["MasVnrType"].fillna(0)

full_features['SaleType']=full_features['SaleType'].fillna(full_features['SaleType'].mode()[0])





cols = ['BsmtQual','BsmtCond','FireplaceQu','GarageType','GarageQual','GarageCond',

        'PoolQC','MiscFeature','Fence','BsmtFinType1','Alley','BsmtFinType2','BsmtExposure']

for c in cols:

    full_features[c].fillna('None', inplace=True)
imp=Imputer(missing_values="NaN", strategy="median" )

imp.fit(full_features[["GarageYrBlt"]])

full_features["GarageYrBlt"]=imp.transform(full_features[["GarageYrBlt"]]).ravel()



imp=Imputer(missing_values="NaN", strategy="mean" )

imp.fit(full_features[["LotFrontage"]])

full_features["LotFrontage"]=imp.transform(full_features[["LotFrontage"]]).ravel()



mean = full_features['LotFrontage'].agg(['mean'])

full_features['LotFrontage'] = full_features['LotFrontage'].fillna(value=mean)
subclass_group = full_features.groupby('MSSubClass')

Zoning_modes = subclass_group['MSZoning'].apply(lambda x : x.mode()[0])

Zoning_modes
full_features['MSZoning'] = full_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
full_features.isnull().sum().sort_values(ascending = False)
full_features.info()
full_features = pd.get_dummies(data=full_features,columns=['MSSubClass','Fence','Alley','MiscFeature','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','SaleType','SaleCondition'])
train_data = full_features.iloc[:1460,:]

test_data = full_features.iloc[1460:,:]
print(train_data.shape)

print(test_data.shape)
train_data['SalePrice'] = train_SalePrice
train_data.shape
y_train = pd.DataFrame(index = train_data.index, columns=["SalePrice"])

y_train["SalePrice"] = np.log(train_data["SalePrice"])

X_train = train_data.drop('SalePrice',axis= 1 )
X_train
y_train
print(test_data.shape)

test_data.head(n = 10)
test_data.reset_index(drop= True)
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X_train, y_train)
y_pred_tree =regressor.predict(test_data)
svr = SVR(kernel= 'rbf')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(test_data)
best_alpha = 0.00099

regr = Lasso(alpha=best_alpha, max_iter=50000)

regr.fit(X_train, y_train)
y_pred_lasso =regressor.predict(test_data)
#y_pred = (y_pred_lasso + y_pred_svr) / 2
#y_pred = np.exp(y_pred)
from xgboost import  XGBRegressor
xgboost = XGBRegressor(learning_rate=0.05, n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006, random_state=42)
xgboost.fit(X_train,y_train)
predictions = xgboost.predict(test_data)

y_pred = (y_pred_lasso + predictions) / 2

y_pred = np.exp(y_pred)
pred_df = pd.DataFrame(y_pred, index=test_data_csv["Id"], columns=["SalePrice"])

pred_df.to_csv('output_xgb.csv', header=True, index_label='Id')