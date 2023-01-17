import pandas as pd

import sklearn as sc

import numpy as np

import matplotlib as mp

from scipy.stats import skew

import csv

import math

from sklearn import linear_model

import xgboost as xgb



from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,LassoCV,LassoLarsCV

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestRegressor as rfr



data_train=[]

data_test=[]

data_final=[]

price=[]





def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, data_train, price, scoring="mean_squared_error", cv = 5))

    return(rmse)

test=pd.read_csv('../input/test.csv')	

train=pd.read_csv('../input/train.csv')





data_final = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))



#dropping columns 'Exterior2nd', 'EnclosedPorch', 'RoofMatl', 'PoolQC', 'BsmtHalfBath', 'RoofStyle', 'PoolArea', 'MoSold', 'Alley', 'Fence', 'LandContour', 'MasVnrType', '3SsnPorch', 'LandSlope'

data_final.drop('MasVnrType',axis=1)

data_final.drop('PoolArea',axis=1)

data_final.drop('PoolQC',axis=1)

data_final.drop('3SsnPorch',axis=1)

data_final.drop('ScreenPorch',axis=1)

data_final.drop('EnclosedPorch',axis=1)

data_final.drop('KitchenAbvGr',axis=1)

data_final.drop('BsmtHalfBath',axis=1) 

data_final.drop('YearBuilt',axis=1) 

data_final.drop('Exterior2nd',axis=1)

data_final.drop('RoofMatl',axis=1)

data_final.drop('RoofStyle',axis=1)

data_final.drop('Alley',axis=1)

data_final.drop('Fence',axis=1)

data_final.drop('LandContour',axis=1)

data_final.drop('LandSlope',axis=1)

data_final.drop('MoSold',axis=1)





data_final.loc[data_final.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'

data_final.loc[data_final.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'

data_final.loc[data_final.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'

data_final.loc[data_final.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'

data_final.loc[data_final.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'

data_final.loc[data_final.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0

data_final.loc[data_final.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0

data_final.loc[data_final.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = data_final.BsmtFinSF1.median()

data_final.loc[data_final.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0

data_final.loc[data_final.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = data_final.BsmtUnfSF.median()

data_final.loc[data_final.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0

data_final.loc[data_final.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'

data_final.loc[data_final.GarageType.isnull(), 'GarageType'] = 'NoGarage'

data_final.loc[data_final.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'

data_final.loc[data_final.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'

data_final.loc[data_final.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'

data_final.loc[data_final.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0

data_final.loc[data_final.KitchenQual.isnull(), 'KitchenQual'] = 'TA'

data_final.loc[data_final.MSZoning.isnull(), 'MSZoning'] = 'RL'

data_final.loc[data_final.Utilities.isnull(), 'Utilities'] = 'AllPub'

data_final.loc[data_final.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'

data_final.loc[data_final.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'

data_final.loc[data_final.Functional.isnull(), 'Functional'] = 'Typ'

data_final.loc[data_final.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'

data_final.loc[data_final.SaleCondition.isnull(), 'SaleType'] = 'WD'

data_final.loc[data_final['PoolQC'].isnull(), 'PoolQC'] = 'NoPool'

data_final.loc[data_final['MiscFeature'].isnull(), 'MiscFeature'] = 'None'

data_final.loc[data_final['Electrical'].isnull(), 'Electrical'] = 'SBrkr'

# only one is null and it has type Detchd

data_final.loc[data_final['GarageArea'].isnull(), 'GarageArea'] = data_final.loc[data_final['GarageType']=='Detchd', 'GarageArea'].mean()

data_final.loc[data_final['GarageCars'].isnull(), 'GarageCars'] = data_final.loc[data_final['GarageType']=='Detchd', 'GarageCars'].median()



#'Exterior2nd', 'EnclosedPorch', 'RoofMatl', 'PoolQC', 'BsmtHalfBath', 'RoofStyle', 'PoolArea', 'MoSold', 'Alley', 'Fence', 'LandContour', 'MasVnrType', '3SsnPorch', 'LandSlope'

#data_final.drop(data_final.columns[data_final.apply(lambda col: col.uniq().sum() <= 3)], axis=1)



for col in data_final.columns:

	    l=len(data_final[col].value_counts())

	    if l<=3:

	        data_final.drop(col,axis=1,inplace=True)







#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



# #log transform skewed numeric features:

numeric_feats = data_final.dtypes[data_final.dtypes != "object"].index

# #categorical_feats= data_final.dtypes[data_final.dtypes == "object"].index



# # print numeric_feats

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

#numeric_feats=numeric_feats.index



# #print skewed_feats



data_final[skewed_feats] = np.log1p(data_final[skewed_feats])



# #print 1,data_final.shape



data_final = pd.get_dummies(data_final)



# #filling NA's with the mean of the column:

data_final = data_final.fillna(data_final[:train.shape[0]].mean())







# #creating matrices for sklearn:

data_train = data_final[:train.shape[0]]

data_test = data_final[train.shape[0]:]

price = train.SalePrice





rf_model=rfr(n_estimators=100)

rf_model.fit(data_train,price)



rmse1=np.sqrt(-cross_val_score(rf_model,data_train,price,scoring="mean_squared_error",cv=5))



# #print "Root Mean Square Error"



# #print rmse1.mean()



lasso_model = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(data_train, price)



rmse2=rmse_cv(lasso_model)



# #print rmse2.mean()



dtrain = xgb.DMatrix(data_train, label = price)

dtest = xgb.DMatrix(data_test)



params = {"max_depth":2, "eta":0.08}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)



model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()







xgb_model = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.08) #the params were tuned using xgb.cv

xgb_model.fit(data_train, price)







rf_preds = np.expm1(rf_model.predict(data_test))

lasso_preds = np.expm1(lasso_model.predict(data_test))

xgb_preds=np.expm1(xgb_model.predict(data_test))



final_result=lasso_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":final_result})

solution.to_csv("kaggle_bumble82.123_test_sol4_final1.csv", index = False)