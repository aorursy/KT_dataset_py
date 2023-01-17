import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

Y = train['SalePrice']

X = train.drop(['SalePrice','Id'],axis=1)



test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test = test.drop(['Id'],axis=1)
X['Alley'] = X['Alley'].fillna('na')

X['PoolQC'] = X['PoolQC'].fillna('na')

X['Fence'] = X['Fence'].fillna('na')

X['MiscFeature'] = X['MiscFeature'].fillna('na')

X['MasVnrType'] = X['MasVnrType'].fillna('na')

X['MasVnrArea'] = X['MasVnrArea'].fillna(0)



X['BsmtQual'] = X['BsmtQual'].fillna('na')

X['BsmtCond'] = X['BsmtCond'].fillna('na')

X['BsmtExposure'] = X['BsmtExposure'].fillna('na')

X['BsmtFinType1'] = X['BsmtFinType1'].fillna('na')

X['BsmtFinType2'] = X['BsmtFinType2'].fillna('na')

X['Electrical'] = X['Electrical'].fillna('na')

X['FireplaceQu'] = X['FireplaceQu'].fillna('na')

X['GarageType'] = X['GarageType'].fillna('na')

X['GarageYrBlt'] = X['GarageYrBlt'].fillna(X['GarageYrBlt'].median())

X['GarageFinish'] = X['GarageFinish'].fillna('na')

X['GarageQual'] = X['GarageQual'].fillna('na')

X['GarageCond'] = X['GarageCond'].fillna('na')

X['LotFrontage'] = X['LotFrontage'].fillna(0)

pd.set_option('display.max_columns', 500)

pd.DataFrame(train).head()
idx=72

train.columns[idx]
groupby_mean = pd.DataFrame(Y.groupby(X[train.columns[idx]]).mean()).sort_values('SalePrice')

groupby_mean['index'] = pd.DataFrame(Y.groupby(X[train.columns[idx]]).mean()).sort_values('SalePrice').index



groupby_median = pd.DataFrame(Y.groupby(X[train.columns[idx]]).median()).sort_values('SalePrice')

groupby_median['index'] = pd.DataFrame(Y.groupby(X[train.columns[idx]]).median()).sort_values('SalePrice').index



groupby_std = pd.DataFrame(Y.groupby(X[train.columns[idx]]).std()).sort_values('SalePrice')

groupby_std['index'] = pd.DataFrame(Y.groupby(X[train.columns[idx]]).std()).sort_values('SalePrice').index



valuecounts = pd.DataFrame(X[train.columns[idx]].value_counts())

valuecounts['index'] = pd.DataFrame(X[train.columns[idx]].value_counts()).index



pd.merge(pd.merge(pd.merge(groupby_mean, groupby_median, on='index'), groupby_std, on='index'), valuecounts, on='index')
test[train.columns[idx]].value_counts()
pd.merge(pd.merge(pd.merge(groupby_mean, groupby_median, on='index'), groupby_std, on='index'), valuecounts, on='index').corr()
print(pd.DataFrame(pd.concat([Y, 

                       X[train.columns[idx]]], axis=1)).corr())

sns.scatterplot(Y, X[train.columns[idx]])
print(train.columns[idx])

sns.boxplot(Y, X[train.columns[idx]])
X['MSZoning'] = X['MSZoning'].replace({"C (all)":1,"RM":2,"RH":3,"RL":4,"FV":5 })



X['Condition2'] = X['Condition2'].replace({"Norm":1,"RRNn":0,"Artery":0,"Feedr":0,

                                           "RRAn":0,"RRAe":0, "PosN":0, "PosA":0 })

X['RoofMatl'] = (X['RoofMatl']=='CompShg')*1

X['HouseStyle'] = X['HouseStyle'].replace({"2.5Fin":2.5,"2.5Unf":2.5})

X['Exterior1st'] = X['Exterior1st'].replace({"BrkComm":0,"AsphShn":0,"CBlock":0,"Stone":0,

                                             "ImStucc":0 })

X['Exterior2nd'] = X['Exterior2nd'].replace({"CBlock":0,"Brk Cmn":0,"AsphShn":0,"Stone":0,

                                             "ImStucc":0, 'Other':0 })

X['Electrical'] = X['Electrical'].replace({"na":0,"Mix":0,"FuseP":0,"FuseF":0})



X['Heating'] = X['Heating'].replace({"Floor":0,"Grav":0,"Wall":0,"OthW":0})

X['CentralAir'] = (X['CentralAir']=='Y')*1

X['MiscFeature'] = X['MiscFeature'].replace({"Othr":0,"Gar2":0,"TenC":0})

X['SaleType'] = X['SaleType'].replace({"Oth":0,"ConLD":0,"ConLw":0,"ConLI":0,

                                      "CWD":0, "Con":0})

X['SaleCondition'] = X['SaleCondition'].replace({"AdjLand":0,"Alloca":0,"Family":0})





X['ExterQual'] = X['ExterQual'].replace({"Fa":1,"TA":2,"Gd":3,"Ex":4})

X['ExterCond'] = X['ExterCond'].replace({"Po":1, "Fa":2,"TA":3,"Gd":4,"Ex":5})

X['BsmtQual'] = X['BsmtQual'].replace({"na":0, "Fa":1,"TA":2,"Gd":3,"Ex":4})

X['BsmtCond'] = X['BsmtCond'].replace({"Po":0, "na":1,"Fa":2,"TA":3,"Gd":4})

X['BsmtExposure'] = X['BsmtExposure'].replace({"na":0, "No":1,"Mn":2,"Av":3,"Gd":4})

X['HeatingQC'] = X['HeatingQC'].replace({"Po":0, "Fa":1,"TA":2,"Gd":3,"Ex":4})

X['KitchenQual'] = X['KitchenQual'].replace({"Fa":1,"TA":2,"Gd":3,"Ex":4})

X['FireplaceQu'] = X['FireplaceQu'].replace({"na":0, "Na":0, "Po":1, "Fa":2,"TA":3,"Gd":4,

                                             "Ex":5})

X['GarageFinish'] = X['GarageFinish'].replace({"Na":0, "Po":1, "Fa":2,"TA":3,"Gd":4,

                                             "Ex":5})

X['GarageQual'] = X['GarageQual'].replace({"na":0, "Po":1, "Fa":2,"TA":3,"Gd":4,

                                             "Ex":5})

X['GarageCond'] = X['GarageCond'].replace({"na":0, "Po":1, "Fa":2,"TA":3,"Gd":4,

                                             "Ex":5})

X['PoolQC'] = X['PoolQC'].replace({"na":0, "Po":1, "Fa":2,"Ta":3,"Gd":4,

                                             "Ex":5})







#X = pd.concat([X, pd.get_dummies(X['Street'], prefix='Street')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Alley'], prefix='Alley')], axis=1)

X = pd.concat([X, pd.get_dummies(X['LotShape'], prefix='LotShape')], axis=1)

X = pd.concat([X, pd.get_dummies(X['LandContour'], prefix='LandContour')], axis=1)

#X = pd.concat([X, pd.get_dummies(X['Utilities'], prefix='Utilities')], axis=1)

X = pd.concat([X, pd.get_dummies(X['LotConfig'], prefix='LotConfig')], axis=1)

X = pd.concat([X, pd.get_dummies(X['LandSlope'], prefix='LandSlope')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Neighborhood'], prefix='Neighborhood')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Condition1'], prefix='Condition1')], axis=1)

X = pd.concat([X, pd.get_dummies(X['BldgType'], prefix='BldgType')], axis=1)

X = pd.concat([X, pd.get_dummies(X['HouseStyle'], prefix='HouseStyle')], axis=1)

X = pd.concat([X, pd.get_dummies(X['RoofStyle'], prefix='RoofStyle')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Exterior1st'], prefix='Exterior1st')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Exterior2nd'], prefix='Exterior2nd')], axis=1)

X = pd.concat([X, pd.get_dummies(X['MasVnrType'], prefix='MasVnrType')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Foundation'], prefix='Foundation')], axis=1)

X = pd.concat([X, pd.get_dummies(X['BsmtFinType1'], prefix='BsmtFinType1')], axis=1)

X = pd.concat([X, pd.get_dummies(X['BsmtFinType2'], prefix='BsmtFinType2')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Heating'], prefix='Heating')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Electrical'], prefix='Electrical')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Functional'], prefix='Functional')], axis=1)

X = pd.concat([X, pd.get_dummies(X['FireplaceQu'], prefix='FireplaceQu')], axis=1)

X = pd.concat([X, pd.get_dummies(X['GarageType'], prefix='GarageType')], axis=1)

X = pd.concat([X, pd.get_dummies(X['GarageFinish'], prefix='GarageFinish')], axis=1)

X = pd.concat([X, pd.get_dummies(X['PavedDrive'], prefix='PavedDrive')], axis=1)

X = pd.concat([X, pd.get_dummies(X['Fence'], prefix='Fence')], axis=1)

X = pd.concat([X, pd.get_dummies(X['MiscFeature'], prefix='MiscFeature')], axis=1)

X = pd.concat([X, pd.get_dummies(X['MoSold'], prefix='MoSold')], axis=1)

X = pd.concat([X, pd.get_dummies(X['YrSold'], prefix='YrSold')], axis=1)

X = pd.concat([X, pd.get_dummies(X['SaleType'], prefix='SaleType')], axis=1)

X = pd.concat([X, pd.get_dummies(X['SaleCondition'], prefix='SaleCondition')], axis=1)



X = X.drop(['Street','Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 

            'Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','Exterior1st',

            'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'BsmtFinType2','Heating',

            'Electrical','Functional','FireplaceQu', 'GarageType', 'GarageFinish',

           'PavedDrive', 'Fence','MiscFeature','MoSold','YrSold','SaleType',

           'SaleCondition'],axis=1)
X.describe()
import random

valid=random.sample(list(X.index),150)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, max_depth=30, max_features=None)

rf.fit(X.loc[set(list(X.index))-set(valid)],Y.loc[set(list(X.index))-set(valid)])



(np.abs(rf.predict(X.loc[valid])-Y[valid]).sum()/100)/(Y[valid].sum()/100)
import xgboost as xgb

xgb = xgb.XGBRegressor(n_estimators=200, max_depth=11,

                       objective ='reg:squarederror')



xgb.fit(X.loc[set(list(X.index))-set(valid)],Y.loc[set(list(X.index))-set(valid)])



(np.abs(xgb.predict(X.loc[valid])-Y[valid]).sum()/100)/(Y[valid].sum()/100)
import lightgbm as lgb

lgb=lgb.LGBMRegressor(n_estimators=300, num_leaves=9)



lgb.fit(X.loc[set(list(X.index))-set(valid)],Y.loc[set(list(X.index))-set(valid)])



(np.abs(lgb.predict(X.loc[valid])-Y[valid]).sum()/100)/(Y[valid].sum()/100)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha= 1, normalize=True)

ridge.fit(X.loc[set(list(X.index))-set(valid)],Y.loc[set(list(X.index))-set(valid)])



(np.abs(ridge.predict(X.loc[valid])-Y[valid]).sum()/100)/(Y[valid].sum()/100)

from sklearn.linear_model import Lasso

lasso = Lasso(alpha= 20, normalize=True)

lasso.fit(X.loc[set(list(X.index))-set(valid)],Y.loc[set(list(X.index))-set(valid)])



(np.abs(lasso.predict(X.loc[valid])-Y[valid]).sum()/100)/(Y[valid].sum()/100)

test['Alley'] = test['Alley'].fillna('na')

test['PoolQC'] = test['PoolQC'].fillna('na')

test['Fence'] = test['Fence'].fillna('na')

test['MiscFeature'] = test['MiscFeature'].fillna('na')

test['MasVnrType'] = test['MasVnrType'].fillna('na')

test['MasVnrArea'] = test['MasVnrArea'].fillna(0)



test['BsmtQual'] = test['BsmtQual'].fillna('na')

test['BsmtCond'] = test['BsmtCond'].fillna('na')

test['BsmtExposure'] = test['BsmtExposure'].fillna('na')

test['BsmtFinType1'] = test['BsmtFinType1'].fillna('na')

test['BsmtFinType2'] = test['BsmtFinType2'].fillna('na')

test['Electrical'] = test['Electrical'].fillna('na')

test['FireplaceQu'] = test['FireplaceQu'].fillna('na')

test['GarageType'] = test['GarageType'].fillna('na')

test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].median())

test['GarageFinish'] = test['GarageFinish'].fillna('na')

test['GarageQual'] = test['GarageQual'].fillna('na')

test['GarageCond'] = test['GarageCond'].fillna('na')

test['LotFrontage'] = test['LotFrontage'].fillna(0)





test['MSZoning'] = test['MSZoning'].replace({"C (all)":1,"RM":2,"RH":3,"RL":4,"FV":5 })

test['Condition2'] = test['Condition2'].replace({"Norm":1,"RRNn":0,"Artery":0,"Feedr":0,

                                           "RRAn":0,"RRAe":0, "PosN":0, "PosA":0 })

test['HouseStyle'] = test['HouseStyle'].replace({"2.5Fin":2.5,"2.5Unf":2.5})

test['RoofMatl'] = (test['RoofMatl']=='CompShg')*1

test['Exterior1st'] = test['Exterior1st'].replace({"BrkComm":0,"AsphShn":0,"CBlock":0,"Stone":0,

                                             "ImStucc":0 })

test['Exterior2nd'] = test['Exterior2nd'].replace({"CBlock":0,"Brk Cmn":0,"AsphShn":0,"Stone":0,

                                             "ImStucc":0, 'Other':0 })

test['Electrical'] = test['Electrical'].replace({"na":0,"Mix":0,"FuseP":0,"FuseF":0})





test['Heating'] = test['Heating'].replace({"Floor":0,"Grav":0,"Wall":0,"OthW":0})

test['CentralAir'] = (test['CentralAir']=='Y')*1

test['MiscFeature'] = test['MiscFeature'].replace({"Othr":0,"Gar2":0,"TenC":0})

test['SaleType'] = test['SaleType'].replace({"Oth":0,"ConLD":0,"ConLw":0,"ConLI":0,

                                      "CWD":0, "Con":0})

test['SaleCondition'] = test['SaleCondition'].replace({"AdjLand":0,"Alloca":0,"Family":0})





test['ExterQual'] = test['ExterQual'].replace({"Fa":1,"TA":2,"Gd":3,"Ex":4})

test['ExterCond'] = test['ExterCond'].replace({"Po":1, "Fa":2,"TA":3,"Gd":4,"Ex":5})

test['BsmtQual'] = test['BsmtQual'].replace({"na":0, "Fa":1,"TA":2,"Gd":3,"Ex":4})

test['BsmtCond'] = test['BsmtCond'].replace({"Po":0, "na":1,"Fa":2,"TA":3,"Gd":4})

test['BsmtExposure'] = test['BsmtExposure'].replace({"na":0, "No":1,"Mn":2,"Av":3,"Gd":4})

test['HeatingQC'] = test['HeatingQC'].replace({"Po":0, "Fa":1,"TA":2,"Gd":3,"Ex":4})

test['KitchenQual'] = test['KitchenQual'].replace({"Fa":1,"TA":2,"Gd":3,"Ex":4})

test['FireplaceQu'] = test['FireplaceQu'].replace({"na":0, "Na":0, "Po":1, "Fa":2,"TA":3,"Gd":4,

                                             "Ex":5})

test['GarageFinish'] = test['GarageFinish'].replace({"Na":0, "Po":1, "Fa":2,"TA":3,"Gd":4,

                                             "Ex":5})

test['GarageQual'] = test['GarageQual'].replace({"na":0, "Po":1, "Fa":2,"TA":3,"Gd":4,

                                             "Ex":5})

test['GarageCond'] = test['GarageCond'].replace({"na":0, "Po":1, "Fa":2,"TA":3,"Gd":4,

                                             "Ex":5})

test['PoolQC'] = test['PoolQC'].replace({"na":0, "Po":1, "Fa":2,"Ta":3,"Gd":4,

                                             "Ex":5})



test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].median())

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median())

test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].median())

test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].median())

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median())

test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].median())

test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].median())

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].median())

test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].median())

test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].median())





#test = pd.concat([test, pd.get_dummies(test['Street'], prefix='Street')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Alley'], prefix='Alley')], axis=1)

test = pd.concat([test, pd.get_dummies(test['LotShape'], prefix='LotShape')], axis=1)

test = pd.concat([test, pd.get_dummies(test['LandContour'], prefix='LandContour')], axis=1)

#test = pd.concat([test, pd.get_dummies(test['Utilities'], prefix='Utilities')], axis=1)

test = pd.concat([test, pd.get_dummies(test['LotConfig'], prefix='LotConfig')], axis=1)

test = pd.concat([test, pd.get_dummies(test['LandSlope'], prefix='LandSlope')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Neighborhood'], prefix='Neighborhood')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Condition1'], prefix='Condition1')], axis=1)

test = pd.concat([test, pd.get_dummies(test['BldgType'], prefix='BldgType')], axis=1)

test = pd.concat([test, pd.get_dummies(test['HouseStyle'], prefix='HouseStyle')], axis=1)

test = pd.concat([test, pd.get_dummies(test['RoofStyle'], prefix='RoofStyle')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Exterior1st'], prefix='Exterior1st')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Exterior2nd'], prefix='Exterior2nd')], axis=1)

test = pd.concat([test, pd.get_dummies(test['MasVnrType'], prefix='MasVnrType')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Foundation'], prefix='Foundation')], axis=1)

test = pd.concat([test, pd.get_dummies(test['BsmtFinType1'], prefix='BsmtFinType1')], axis=1)

test = pd.concat([test, pd.get_dummies(test['BsmtFinType2'], prefix='BsmtFinType2')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Heating'], prefix='Heating')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Electrical'], prefix='Electrical')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Functional'], prefix='Functional')], axis=1)

test = pd.concat([test, pd.get_dummies(test['FireplaceQu'], prefix='FireplaceQu')], axis=1)

test = pd.concat([test, pd.get_dummies(test['GarageType'], prefix='GarageType')], axis=1)

test = pd.concat([test, pd.get_dummies(test['GarageFinish'], prefix='GarageFinish')], axis=1)

test = pd.concat([test, pd.get_dummies(test['PavedDrive'], prefix='PavedDrive')], axis=1)

test = pd.concat([test, pd.get_dummies(test['Fence'], prefix='Fence')], axis=1)

test = pd.concat([test, pd.get_dummies(test['MiscFeature'], prefix='MiscFeature')], axis=1)

test = pd.concat([test, pd.get_dummies(test['MoSold'], prefix='MoSold')], axis=1)

test = pd.concat([test, pd.get_dummies(test['YrSold'], prefix='YrSold')], axis=1)

test = pd.concat([test, pd.get_dummies(test['SaleType'], prefix='SaleType')], axis=1)

test = pd.concat([test, pd.get_dummies(test['SaleCondition'], prefix='SaleCondition')], axis=1)



test = test.drop(['Street','Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 

            'Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','Exterior1st',

            'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtFinType1', 'BsmtFinType2','Heating',

            'Electrical','Functional','FireplaceQu', 'GarageType', 'GarageFinish',

           'PavedDrive', 'Fence','MiscFeature','MoSold','YrSold','SaleType',

           'SaleCondition'],axis=1)
for i in set(X.columns) - set(test.columns):

    test[i]=0
test['GarageQual'] = test['GarageQual'].astype(np.int8)
test.describe()
sub = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

rf.fit(X,Y)

xgb.fit(X,Y)

lgb.fit(X,Y)

ridge.fit(X,Y)

lasso.fit(X,Y)
sub['SalePrice']= (rf.predict(test)*0.1) + (xgb.predict(test)*0.3) 

+ (lgb.predict(test)*0.6) + (ridge.predict(test)*0) + (lasso.predict(test)*0)





sub.to_csv("/kaggle/working/submission.csv", index=False)