import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
train = pd.read_csv("../input/train.csv")

train.info()
test=pd.read_csv("../input/test.csv")

test.info()
trainnum = train._get_numeric_data()
traincat = train.drop(trainnum.columns, axis=1)
for x in trainnum.drop('SalePrice', axis=1).columns:

    sns.regplot(x, "SalePrice", data=trainnum)

    plt.xlabel(x)

    plt.ylabel("SalePrice")

    plt.show()
for x in traincat.columns:

    sns.boxplot(x, np.log(train.SalePrice), data=traincat)

    plt.xlabel(x)

    plt.ylabel("SalePrice")

    plt.show()
total = pd.concat([train, test], sort=False).set_index('Id')
total.isnull().sum()
for col in trainnum.drop(['Id', 'SalePrice'],axis=1).columns :

    if (abs(train.SalePrice.corr(trainnum[col]))>0.2):

        print('Correlation btw SalePrice vs',trainnum[col].name, ':', round(train.SalePrice.corr(trainnum[col]),2))
train.drop(train[(train.GrLivArea>4000) & (train.SalePrice<400000)].index, inplace=True)
train.drop(train[(train.TotRmsAbvGrd>12) & (train.SalePrice<250000)].index, inplace=True)
train.drop(train.loc[(train.TotalBsmtSF>6000) & (train.SalePrice<200000)].index, inplace=True)
train.drop(train.loc[(train.BsmtFinSF1>5000) & (train.SalePrice<200000)].index, inplace=True)
train.drop(train.loc[(train.PoolArea>500) & (train.SalePrice>700000)].index, inplace=True)
train.drop(train.loc[(train.MiscVal>3000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)
total = pd.concat([train, test], sort=False).set_index('Id')
total.Alley.unique()
total.Alley.isnull().sum()
total.Alley[total.Alley=='Pave'].count()
total2 = total[:]
total2.Alley = total2.Alley.replace(total2.Alley.loc[(total2.Alley.isnull()==True)],0).replace(total2.Alley.loc[(total2.Alley!=0)],1)
total2.Alley.isnull().sum()
total3 = total2[:]
total3.groupby("MiscFeature").MiscFeature.count()
total3.MiscFeature = total3.MiscFeature.replace(total3.MiscFeature.loc[(total3.MiscFeature.isnull()==True)],0).replace(total3.MiscFeature.loc[(total3.MiscFeature!=0)],1)
total3.CentralAir = total3.CentralAir.replace(total3.CentralAir.loc[(total3.CentralAir=="Y")],1).replace(total3.CentralAir.loc[(total3.CentralAir=="N")],0)
total3.columns
for x in ('MSZoning', 'Exterior1st','Exterior2nd','Electrical','BsmtFullBath','BsmtHalfBath','KitchenQual','SaleType') :

    total3[x] = total3[x].fillna(total3[x].mode()[0])
for x in ('MasVnrType', 'BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','FireplaceQu', 'GarageType', 'GarageYrBlt',

       'GarageFinish', 'GarageQual', 'GarageCond','PoolQC','Fence') :

    total3[x] = total3[x].fillna("None")
total3[["BsmtFinSF1","BsmtUnfSF","TotalBsmtSF","BsmtFinSF2"]].loc[total3.BsmtFinSF2.isnull()==True]
for x in ('MasVnrArea', "BsmtFinSF1","BsmtUnfSF","TotalBsmtSF","BsmtFinSF2","BsmtFinType2",'GarageCars', 'GarageArea') :

    total3[x] = total3[x].fillna(0)
total3.groupby("Utilities").Utilities.count()
total3 = total3.drop(["Utilities"], axis=1)
train.groupby(["Neighborhood","LotFrontage"]).SalePrice.mean().head(15)
total3.LotFrontage = total3.LotFrontage.fillna(total3.groupby(["Neighborhood"]).LotFrontage.mean()[0])
total3.Functional=total3.Functional.fillna(method='ffill')
total3 = total3.drop(["YearBuilt"], axis=1)
total3.loc[total3.GarageYrBlt=='None'].GarageType.unique()
total3.loc[(total3.GarageYrBlt=='None')&(total3.GarageType=='Detchd')].YearRemodAdd
total3.loc[total3.GarageYrBlt=='None', 'GarageYrBlt']=total3.YearRemodAdd
total3.loc[total3.GarageYrBlt==2207].YearRemodAdd
total3.loc[total3.GarageYrBlt==2207, 'GarageYrBlt']=total3.YearRemodAdd
min(total.GarageYrBlt)
total3.GarageYrBlt.replace("None",1800, inplace=True)
total3.GarageYrBlt = 2011 - total3.GarageYrBlt
total3=total3.rename(columns = {'GarageYrBlt':'GrAge'})
min(total3.GrAge)
max(total3.YearRemodAdd)
total3.YearRemodAdd = 2011 - total3.YearRemodAdd
total3=total3.rename(columns = {'YearRemodAdd':'HAge'})
min(total3.HAge)
max(total3.YrSold)
total3.YrSold = 2011 - total3.YrSold
total3=total3.rename(columns = {'YrSold':'SoldAge'})
min(total3.SoldAge)
total3.isnull().any().sum()
plt.figure(figsize=[10,6])

sns.distplot(np.log(total3.loc[total3.SalePrice.isnull()!=True].SalePrice))
total3 = total3.replace({'Street': {'Pave': 1, 'Grvl': 0 },

                             'FireplaceQu': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'None': 0 

                                            },

                             'Fence':      {'GdPrv': 4, 

                                            'GdWo': 3, 

                                            'MnPrv': 2, 

                                            'MnWw': 1,

                                            'None': 0},

                             'ExterQual':  {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                             'ExterCond':  {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                             'BsmtQual':   {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'None': 0},

                            'BsmtExposure':{'Gd': 3, 

                                            'Av': 2, 

                                            'Mn': 1,

                                            'No': 0,

                                            'None': 0},

                             'BsmtCond':   {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'None': 0},

                             'GarageQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'None': 0},

                             'GarageCond': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1,

                                            'None': 0},

                            'KitchenQual': {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1},

                             'Functional': {'Typ': 7,

                                            'Min1':6,

                                            'Min2':5,

                                            'Mod': 4,

                                            'Maj1':3,

                                            'Maj2':2,

                                            'Sev': 1,

                                            'Sal': 0},

                             'BsmtFinType1': {'GLQ' : 6,

                                              'ALQ' : 5,

                                              'BLQ' : 4,

                                              'Rec' : 3,

                                              'LwQ' : 2,

                                              'Unf' : 1,

                                              'None': 0},

                            'BsmtFinType2': {'GLQ' : 6,

                                              'ALQ' : 5,

                                              'BLQ' : 4,

                                              'Rec' : 3,

                                              'LwQ' : 2,

                                              'Unf' : 1,

                                              'None': 0},

                             'HeatingQC':   {'Ex': 5, 

                                            'Gd': 4, 

                                            'TA': 3, 

                                            'Fa': 2,

                                            'Po': 1

                                            },

                            'PoolQC' :    {'Ex': 4, 

                                            'Gd': 3, 

                                            'Fa': 2,

                                            'None': 1

                                            }

                            })
def NormVariable(x):

    x= (x - np.mean(x))/(max(x)-min(x))

    return(x)
def ScaleVariable(x):

    x = x/max(x)

    return(x)
total4 = total3[:]
total4.columns
for col in total4[['MSSubClass', 'MSZoning', 'Street','LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',

    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd', 'MiscFeature',

    'MasVnrType','Foundation',  'Heating', 'Electrical','GarageType','GarageFinish', 'PavedDrive','MoSold', 'SaleType',

                   'SaleCondition']]:

    col = pd.get_dummies(total4[col])

    total4 = total4.merge(col, left_index=True, right_index=True)
total4 = total4.drop(['MSSubClass', 'MSZoning', 'Street','LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood',

    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd', 

    'MasVnrType','Foundation',  'Heating', 'Electrical','GarageType','GarageFinish', 'PavedDrive','MoSold', 'SaleType',

                      'SaleCondition'], axis=1)
for col in total4[[]]:

    total4[col] = NormVariable(pd.to_numeric(total4[col]))
for col in total4[['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

                  'PoolQC','KitchenQual','GarageQual','GarageCond','Fence','OverallQual','OverallCond','Functional',

                   'FireplaceQu','HeatingQC', 'TotRmsAbvGrd','Fireplaces','GarageCars','Fireplaces', 'BedroomAbvGr']]:

    total4[col] = ScaleVariable(pd.to_numeric(total4[col]))
for col in total4[['SalePrice','SoldAge','GrAge','HAge','MiscVal','LotFrontage', 'LotArea','MasVnrArea','PoolArea',

                   'GrLivArea','GarageArea','WoodDeckSF', 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch',

                   'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF','1stFlrSF','2ndFlrSF', 'LowQualFinSF']]:

    total4[col] = np.log(total4[col]+1)
test2 = total4.loc[total4.SalePrice.isnull()==True]
train2 = total4.loc[total4.SalePrice.isnull()==False]
from sklearn.linear_model import RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV,LinearRegression

from sklearn.ensemble import  GradientBoostingRegressor

from sklearn.model_selection import KFold,cross_val_score
import xgboost as xgb
ytrain=train2.SalePrice
train3 = train2[:]
train3 = train3.drop('SalePrice',1)
test3 = test2[:]
test3 = test3.drop('SalePrice',1)
kfolds = KFold(n_splits=10, shuffle=True, random_state=35)
def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, train3, ytrain, scoring="neg_mean_squared_error", cv = kfolds))

    return(rmse)
model_lasso = LassoCV(alphas = [1, 0.1, 0.05, 0.001, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009], cv=kfolds).fit(train3, ytrain)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = train3.columns)
print("Lasso picked " + str(sum(coef != 0)) + "variables at total")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients by Importance")
lasso_preds = np.expm1(model_lasso.predict(test3))-1
solution = pd.DataFrame({"id":test.Id, "SalePrice":lasso_preds})

solution.to_csv("submit_lasso.csv", index = False)
Ridge = RidgeCV(alphas = [0.01,0.1,0.5,1,3,5,10,20,30,50,100]).fit(train3, ytrain)
rmse_cv(Ridge).mean()
ridge_preds = np.expm1(Ridge.predict(test3))-1
solution = pd.DataFrame({"id":test.Id, "SalePrice":ridge_preds})

solution.to_csv("submit_ridge.csv", index = False)
Enet = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000, cv=kfolds).fit(train3, ytrain)
rmse_cv(Enet).mean()
enet_preds = np.expm1(Enet.predict(test3))-1
solution = pd.DataFrame({"id":test.Id, "SalePrice":enet_preds})

solution.to_csv("submit_enet.csv", index = False)
gbmodel = GradientBoostingRegressor(learning_rate=0.05, loss='huber', min_samples_split=10, n_estimators=3000,

                                       random_state=35).fit(train3, ytrain)
rmse_cv(gbmodel).mean()
gb_preds = np.expm1(gbmodel.predict(test3))-1
def duplicate_columns(frame):

    groups = frame.columns.to_series().groupby(frame.dtypes).groups

    dups = []



    for t, v in groups.items():



        cs = frame[v].columns

        vs = frame[v]

        lcs = len(cs)



        for i in range(lcs):

            iv = vs.iloc[:,i].tolist()

            for j in range(i+1, lcs):

                jv = vs.iloc[:,j].tolist()

                if iv == jv:

                    dups.append(cs[i])

                    break



    return dups
duplicate_columns(total4)
total4.columns.values
total5 = total4[:]
total5.columns = (['LotFrontage', 'LotArea', 'Alley', 'OverallQual', 'OverallCond',

       'HAge', 'MasVnrArea', 'ExterQual', 'ExterCond', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',

       'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GrAge',

       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',

       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',

       'MiscVal', 'SoldAge', 'SalePrice', 20, 30, 40, 45, 50, 60, 70, 75,

       80, 85, 90, 120, 150, 160, 180, 190, 'C (all)', 'FV', 'RH', 'RL',

       'RM', '0_x', '1_x', 'IR1', 'IR2', 'IR3', 'Reg', 'Bnk', 'HLS',

       'Low', 'Lvl', 'Corner', 'CulDSac', 'FR2', 'FR3', 'Inside', 'Gtl',

       'Mod', 'Sev', 'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr',

       'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV',

       'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt',

       'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr',

       'Timber', 'Veenker', 'Artery_x', 'Feedr_x', 'Norm_x', 'PosA_x',

       'PosN_x', 'RRAe', 'RRAn_x', 'RRNe', 'RRNn_x', 'Artery_y',

       'Feedr_y', 'Norm_y', 'PosA_y', 'PosN_y', 'RRAn_y', 'RRNn_y',

       '1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE', '1.5Fin', '1.5Unf',

       '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl', 'Flat',

       'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed', 'CompShg', 'Membran',

       'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl', 'AsbShng_x',

       'AsphShn_x', 'BrkComm', 'BrkFace_x', 'CBlock_x', 'CemntBd',

       'HdBoard_x', 'ImStucc_x', 'MetalSd_x', 'Plywood_x', 'Stone_x1',

       'Stucco_x', 'VinylSd_x', 'Wd Sdng_x', 'WdShing', 'AsbShng_y',

       'AsphShn_y', 'Brk Cmn', 'BrkFace_y', 'CBlock_y', 'CmentBd',

       'HdBoard_y', 'ImStucc_y', 'MetalSd_y', 'Other', 'Plywood_y',

       'Stone_y1', 'Stucco_y', 'VinylSd_y', 'Wd Sdng_y', 'Wd Shng', '0_y',

       '1_y', 'BrkCmn', 'BrkFace', 'None_x', 'Stone_x', 'BrkTil',

       'CBlock', 'PConc', 'Slab', 'Stone_y', 'Wood', 'Floor', 'GasA',

       'GasW', 'Grav', 'OthW', 'Wall', 'FuseA', 'FuseF', 'FuseP', 'Mix',

       'SBrkr', '2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort',

       'Detchd', 'None_y', 'Fin', 'None', 'RFn', 'Unf', 'N', 'P', 'Y', 1,

       2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'COD', 'CWD', 'Con', 'ConLD',

       'ConLI', 'ConLw', 'New', 'Oth', 'WD', 'Abnorml', 'AdjLand',

       'Alloca', 'Family', 'Normal', 'Partial'])
testxgb = total5.loc[total5.SalePrice.isnull()==True]
testxgb.drop('SalePrice',1, inplace=True)
trainxgb = total5.loc[total5.SalePrice.isnull()==False]
trainxgb.drop('SalePrice',1, inplace=True)
def rmse_cv1(model):

    rmse= np.sqrt(-cross_val_score(model, trainxgb, ytrain, scoring="neg_mean_squared_error", cv = kfolds))

    return(rmse)
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1).fit(trainxgb, ytrain)
rmse_cv1(model_xgb).mean()
xgb_preds = np.expm1(model_xgb.predict(testxgb))-1
pred = (lasso_preds + ridge_preds + enet_preds + xgb_preds + gb_preds)/5
solution = pd.DataFrame({"id":test.Id, "SalePrice":pred})

solution.to_csv("submit_comb.csv", index = False)