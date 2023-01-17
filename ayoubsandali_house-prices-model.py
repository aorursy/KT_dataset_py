import numpy as np # linear algebra

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer







######preprocessing data



X_full=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

X_test_full=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

X_full=X_full.drop(X_full[(X_full['GrLivArea']>4000) & (X_full['SalePrice']<300000)].index)



X_full.dropna(axis=0,subset=['SalePrice'], inplace=True)

y = np.log1p(X_full.SalePrice)

X_full.drop(['SalePrice'], axis=1, inplace=True)



X_full['Alley']=X_full.fillna('None')



for col in ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):

    X_full[col]=X_full[col].fillna('None')

    

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    X_full[col] = X_full[col].fillna(0)



X_full['MasVnrType']=X_full['MasVnrType'].fillna('None')

X_full["MasVnrArea"] = X_full["MasVnrArea"].fillna(0)



X_full['MSZoning']=X_full['MSZoning'].fillna(X_full['MSZoning'].mode()[0])

    

X_full['FireplaceQu']=X_full['FireplaceQu'].fillna('None')



for cols in ('GarageType','GarageFinish','GarageQual','GarageCond'):

    X_full[cols]=X_full[cols].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    X_full[col] = X_full[col].fillna(0)

    

X_full['PoolQC']=X_full['PoolQC'].fillna('None')



X_full['Fence']=X_full['Fence'].fillna('None')



X_full['MiscFeature']=X_full['MiscFeature'].fillna('None')



X_full["LotFrontage"] = X_full.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



X_full["Functional"] = X_full["Functional"].fillna("Typ")



for col in ('Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):

    X_full[col] = X_full[col].fillna(X_full[col].mode()[0])

    



X_full['MSSubClass'] = X_full['MSSubClass'].fillna("None")



#MSSubClass=The building class

X_full['MSSubClass'] = X_full['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

X_full['OverallCond'] = X_full['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

X_full['YrSold'] = X_full['YrSold'].astype(str)

X_full['MoSold'] = X_full['MoSold'].astype(str)



X_full = X_full.drop(['Utilities'], axis=1)





OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

categoricals_values=[col for col in X_full.columns if X_full[col].dtype=="object" and X_full[col].nunique()< 10]

OH_cols_full = pd.DataFrame(OH_encoder.fit_transform(X_full[categoricals_values]))

OH_cols_full.index = X_full[categoricals_values].index

num_col= [col for col in X_full.columns if X_full[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']]

OH_X = pd.concat([X_full[num_col], OH_cols_full],axis=1)

OH_X.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_squared_log_error





n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(OH_X.values)

    rmse= np.sqrt(-cross_val_score(model, OH_X.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)





model_2=RandomForestRegressor(n_estimators=1500,max_depth=4, random_state=0)

score2=rmsle_cv(model_2)

print("randomforest score: {:.4f} ({:.4f})\n".format(score2.mean(), score2.std()))







model_1 = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=4)

score1=rmsle_cv(model_1)

print("xgboost score{:.4f} ({:.4f})\n ".format(score1.mean(), score1.std()))













Xgb=model_1.fit(OH_X.values,y)

rf=model_2.fit(OH_X.values,y)

import numpy as np # linear algebra

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer







######preprocessing data





X_test_full=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')







X_test_full['Alley']=X_test_full.fillna('None')



for col in ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):

    X_test_full[col]=X_test_full[col].fillna('None')

    

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    X_test_full[col] = X_test_full[col].fillna(0)



X_test_full['MasVnrType']=X_test_full['MasVnrType'].fillna('None')

X_test_full["MasVnrArea"] = X_test_full["MasVnrArea"].fillna(0)



X_test_full['MSZoning']=X_test_full['MSZoning'].fillna(X_test_full['MSZoning'].mode()[0])

    

X_test_full['FireplaceQu']=X_test_full['FireplaceQu'].fillna('None')



for cols in ('GarageType','GarageFinish','GarageQual','GarageCond'):

    X_test_full[cols]=X_test_full[cols].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    X_test_full[col] = X_test_full[col].fillna(0)

    

X_test_full['PoolQC']=X_test_full['PoolQC'].fillna('None')



X_test_full['Fence']=X_test_full['Fence'].fillna('None')



X_test_full['MiscFeature']=X_test_full['MiscFeature'].fillna('None')



X_test_full["LotFrontage"] = X_test_full.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



X_test_full["Functional"] = X_test_full["Functional"].fillna("Typ")



for col in ('Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):

    X_test_full[col] = X_test_full[col].fillna(X_test_full[col].mode()[0])

    



X_test_full['MSSubClass'] = X_test_full['MSSubClass'].fillna("None")



#MSSubClass=The building class

X_test_full['MSSubClass'] = X_test_full['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

X_test_full['OverallCond'] = X_test_full['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

X_test_full['YrSold'] = X_test_full['YrSold'].astype(str)

X_test_full['MoSold'] = X_test_full['MoSold'].astype(str)



X_test_full = X_test_full.drop(['Utilities'], axis=1)





OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_full_test = pd.DataFrame(OH_encoder.fit_transform(X_test_full[categoricals_values]))

OH_cols_full_test.index = X_test_full.index



num_col= [col for col in X_test_full.columns if X_test_full[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']]

OH_X_test = pd.concat([X_test_full[num_col], OH_cols_full_test],axis=1)

missing_cols=set(OH_X.columns)-set(OH_X_test.columns)

for col in missing_cols:

    OH_X_test[col]=0

OH_X_test=OH_X_test[OH_X.columns]



preds = 0.45*np.expm1(Xgb.predict(OH_X_test.values)) + 0.55*np.expm1(rf.predict(OH_X_test.values)) 





sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')



output = pd.DataFrame({'Id': sample_submission.Id,

'SalePrice': preds})

output.to_csv('submission.csv', index=False)
