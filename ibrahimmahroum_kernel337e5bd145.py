# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

data.head()
data.drop_duplicates(subset=None, keep='first', inplace=False)

missing_features = data.columns[data.isnull().any()]

print(data[missing_features].isnull().sum())
data.Condition1.isnull().sum()


data.drop(['LotFrontage','GarageArea','MoSold','GarageYrBlt'],axis=1,inplace=True)



data['Porch'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch'] + data['WoodDeckSF']

data.drop(['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','WoodDeckSF'],axis=1,inplace=True)

ax= data.plot.scatter(x='Porch', y='SalePrice',c='DarkBlue')





data['Year'] = 2019 - data['YearBuilt']

data['Year'] = data['Year'].astype('int64')

data.drop(['YearBuilt'],axis=1,inplace=True)

ax= data.plot.scatter(x='Year', y='SalePrice',c='DarkBlue')





dict = {"Ex": 5, "Gd": 4, "TA": 3, "Fa":2,"Po" : 1, "NA": 0 , "Grvl" : 1 , "Pave" : 1}



data['Bath'] = data['BsmtFullBath']+data['FullBath']+((data['HalfBath']+data['BsmtHalfBath'])*0.5)

data.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],axis=1,inplace=True)





dict2 = {"AllPub": 4,"NoSewr" : 3,"NoSeWa":2,"ELO":1}

data = data.replace({"Utilities" : dict2})

data['Utilities'] = data['Utilities'].astype('int64')





data.drop(['Id'],axis=1,inplace=True)



# data.drop(['MasVnrType'],axis=1,inplace=True)

# data = data.fillna(data.mode().iloc[0])



data = data.replace({"ExterCond": dict})

data = data.replace({"ExterQual": dict})

data['Exter'] = data['ExterCond'] * data['ExterQual']

data.drop(['ExterCond','ExterQual'],axis=1,inplace=True)

data['Exter'] = data['Exter'].astype('int64')

ax1 = data.plot.scatter(x='Exter', y='SalePrice',c='DarkBlue')









data['SF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'] 

data['SF'] = data['SF'].astype('float64')

data.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)





data['Alley'].fillna('None')



dictLotShape =  {"Reg": 4, "IR1": 3, "IR2":2,"IR3" : 1}

data = data.replace({"LotShape" : dictLotShape})

data['LotShape'] = data['LotShape'].astype('int64')

ax1 = data.plot.scatter(x='LotShape', y='SalePrice',c='DarkBlue')





dictBsmt =  {"Ex": 100, "Gd": 95, "TA": 85, "Fa":75,"Po" : 35}

data['BsmtQual'].fillna(0)

data = data.replace({"BsmtQual" : dictBsmt})

data['BsmtQual'] = data['BsmtQual'].astype('float64')

ax1 = data.plot.scatter(x='BsmtQual', y='SalePrice',c='DarkBlue')





dictBsmt =  {"Ex": 5, "Gd": 4, "TA": 3, "Fa":2,"Po" : 1}

data['BsmtCond'].fillna(0)

data = data.replace({"BsmtCond" : dictBsmt})

data['BsmtCond'] = data['BsmtCond'].astype('float64')

ax1 = data.plot.scatter(x='BsmtCond', y='SalePrice',c='DarkBlue')





data['BSMT'] = data['BsmtQual'] * data['BsmtCond']

ax1 = data.plot.scatter(x='BSMT', y='SalePrice',c='DarkBlue')

data.drop(['BsmtQual' , 'BsmtCond'] , axis=1 , inplace=True)



dictMSZ = {"RH"  : "R"  ,"RL"  : "R"  , "RP"  : "R"  , "RM"  : "R" }

data = data.replace({"MSZoning" : dictMSZ})





dictBsmtExpo =  {"Gd": 3, "Av": 2, "Mn": 1, "No":0}

data['BsmtExposure'].fillna(0)

data = data.replace({"BsmtExposure" : dictBsmtExpo})

data['BsmtExposure'] = data['BsmtExposure'].astype('float64')

ax1 = data.plot.scatter(x='BsmtExposure', y='SalePrice',c='DarkBlue')





data = data.replace({"HeatingQC" : dict})

data['HeatingQC'] = data['HeatingQC'].astype('int64')

ax1 = data.plot.scatter(x='HeatingQC', y='SalePrice',c='DarkBlue')



dictAir =  {"Y": '1', "N":'0'}

data = data.replace({"CentralAir" : dictAir})

data['CentralAir'] = data['CentralAir'].astype('int64')

ax1 = data.plot.scatter(x='CentralAir', y='SalePrice',c='DarkBlue')







data['GarageType'].fillna('None')





data = data.replace({"FireplaceQu" : dict})

data['FireplaceQu'].fillna(0)

data['FireplaceQu']= data['FireplaceQu'].astype('float64')

ax1 = data.plot.scatter(x='FireplaceQu', y='SalePrice',c='DarkBlue')



data['Fire'] = data['Fireplaces'] * data['FireplaceQu']

ax1 = data.plot.scatter(x='Fire', y='SalePrice',c='DarkBlue')

data['Fire'].fillna(0)

data.drop(['FireplaceQu' , 'Fireplaces'] , axis=1 , inplace=True)

dictGarage = {"Ex": 3, "Gd": 4, "TA": 5, "Fa":2,"Po" : 1, "NA": 0 , "Grvl" : 1 , "Pave" : 1}

data = data.replace({"GarageQual" : dictGarage})

data['GarageQual']= data['GarageQual'].astype('float64')

data['GarageQual'].fillna(0)

ax1 = data.plot.scatter(x='GarageQual', y='SalePrice',c='DarkBlue')

data = data.replace({"GarageCond" : dictGarage})

data['GarageCond']= data['GarageCond'].astype('float64')

data['GarageCond'].fillna(0)

ax1 = data.plot.scatter(x='GarageCond', y='SalePrice',c='DarkBlue')



data['Garage'] = data['GarageCars'] * data['GarageQual'] * data['GarageCond']

data.drop(['GarageCars','GarageQual','GarageCond'] , axis=1 , inplace=True)





data = data.replace({"PoolQC" : dict})

data['PoolQC'].fillna('0')

data['PoolQC']= data['PoolQC'].astype('float64')

ax1 = data.plot.scatter(x='PoolQC', y='SalePrice',c='DarkBlue')



data['Pool'] = data['PoolArea'].apply(lambda x:1 if x>0 else 0)

data['Pool']= data['Pool'].astype('float64')

ax1 = data.plot.scatter(x='Pool', y='SalePrice',c='DarkBlue')

data.drop(['PoolQC','PoolArea'] , axis=1 , inplace=True)





dataFence = {"GdPrv"  : 5 , "MnPrv" : 4 , "GdWo"  : 3 , "MnWw" :  2}

data = data.replace({"Fence" : dataFence})

data['Fence'].fillna('0')

data['Fence']= data['Fence'].astype('float64')

ax1 = data.plot.scatter(x='Fence', y='SalePrice',c='DarkBlue')





data['MiscFeature'].fillna('None')











data = data.replace({"KitchenQual": dict})

data['Kitch'] = data['KitchenAbvGr'] * data['KitchenQual']

data['Kitch'] = data['Kitch'].astype('float64')

data.drop(['KitchenAbvGr','KitchenQual'],axis=1,inplace=True)



ax= data.plot.scatter(x='SF', y='SalePrice',c='DarkBlue')



null_columns=data.columns[data.isnull().any()]



data[null_columns].isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
plt.figure(figsize=(20,20))

cor = data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
data.info()


# data.corr(method='pearson')

# plt.figure(figsize=(20,20))

# cor = data2.corr()

# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

# plt.show()







# data.info()
data = pd.get_dummies(data)
# data.info()
# data.corr(method='spearman')
# data.drop(['SaleCondition'],axis=1,inplace=True)

# plt.figure(figsize=(20,20))

# cor = data.corr()

# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

# plt.show()
# data.drop(['HouseStyle','Functional','BsmtFinType1','BsmtFinType2','RoofStyle','RoofMatl','Exterior1st',

# 'Exterior2nd','ExterCond','CentralAir'],axis=1,inplace=True)
# y = data[data.columns[1:]].corr()['SalePrice'][:]

# y.to_csv('Correlation.csv' , index = True)



# print(y)




from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.cross_decomposition import PLSRegression

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

def rmsle(predictions, yvalid) : 

    assert len(yvalid) == len(predictions)

    return np.sqrt(np.mean((np.log(1+predictions) - np.log(1+yvalid))**2))

Y=data['SalePrice']

X=data.loc[:, ~data.columns.isin(['SalePrice'])]









# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# X = scaler.fit_transform(X)



from sklearn.model_selection import train_test_split

xtrain , xvalid , ytrain , yvalid = train_test_split(X,Y,test_size =0.1)

seed = 44

lr = LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=640,

                              max_bin =55, bagging_fraction = 0.7,

                              bagging_freq = 5, feature_fraction = 0.55,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =15, min_sum_hessian_in_leaf = 15)













lr.fit(xtrain,ytrain)



preds = lr.predict(xvalid)

print(rmsle(preds,yvalid))





n_folds = 20



def rmsle_cv(modelx):

    kf = KFold(n_folds, shuffle=True, random_state=30).get_n_splits(xtrain)

    rmse= np.sqrt(-cross_val_score(modelx,xtrain, ytrain, scoring="neg_mean_squared_log_error", cv = kf))

    return(rmse)



print(rmsle_cv(lr).mean())

print(rmsle_cv(lr).min())

print(rmsle_cv(lr).mean())

print(rmsle_cv(lr).max())
from mlxtend.regressor import StackingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.svm import SVR

import matplotlib.pyplot as plt

import numpy as np

import warnings

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestRegressor

import warnings

warnings.filterwarnings("ignore")



# print('RandomForest')

# trainingmodel = RandomForestRegressor()

# print(rmsle_cv(trainingmodel).mean())

# print('GradientBoost')

# trainingmodel =  GradientBoostingRegressor()

# print(rmsle_cv(trainingmodel).mean())





lr1 = LinearRegression()

ridge = Ridge(random_state=1)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



GBoost = GradientBoostingRegressor(n_estimators=600, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=15, 

                                   loss='huber', random_state =3)

model_xgb = XGBRegressor(colsample_bytree= 0.4,gamma= 0.0,learning_rate= 0.01,max_depth= 4,min_child_weight= 2, n_estimators= 4000, seed= 36,subsample= 0.5, verbosity= 0)

model_lgb =LGBMRegressor(objective='regression',num_leaves=10,

                              learning_rate=0.05, n_estimators=640,

                              max_bin =55, bagging_fraction = 0.7,

                              bagging_freq = 5, feature_fraction = 0.55,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =15, min_sum_hessian_in_leaf = 15)

# print('GradientBoost with parameters')

# print(rmsle_cv(GBoost).mean())

# print('LGB')

# print(rmsle_cv(model_lgb).mean())

# print('XGB')

# print(rmsle_cv(model_xgb).mean())



















































from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import GridSearchCV



parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['reg:linear'],

              'learning_rate': [0.01 , .03, 0.05, .07 , 0.09], #so called `eta` value

              'max_depth': [4],

              'min_child_weight': [4],

              'silent': [0],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [700]}

xgb1 = XGBRegressor()

xgb_grid = GridSearchCV(xgb1,

                        parameters,

                        cv = 2,

                        n_jobs = 1,

                        verbose=True)



xgb_grid.fit(xtrain, ytrain)



# xgb_best = xgb_grid.best_estimator_

predsparaXGB = xgb_grid.predict(xvalid)

print(rmsle(predsparaXGB, yvalid))
xgb_grid.best_params_
xgb2 = XGBRegressor(**xgb_grid.best_params_)



xgb2.fit(xtrain,ytrain)

xgb2pred = xgb2.predict(xvalid)

print(rmsle(xgb2pred, yvalid))
testdata = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

testdata.info()

testdata.drop_duplicates(subset=None, keep='first', inplace=False)

idData = pd.DataFrame()

idData['Id'] = testdata['Id']







testdata.drop(['LotFrontage','GarageArea','MoSold','GarageYrBlt'],axis=1,inplace=True)



testdata['Porch'] = testdata['OpenPorchSF'] + testdata['EnclosedPorch'] + testdata['3SsnPorch'] + testdata['ScreenPorch'] + testdata['WoodDeckSF']

testdata.drop(['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','WoodDeckSF'],axis=1,inplace=True)







testdata['Year'] = 2019 - testdata['YearBuilt']

testdata['Year'] = testdata['Year'].astype('float64')

testdata.drop(['YearBuilt'],axis=1,inplace=True)







dict = {"Ex": 5, "Gd": 4, "TA": 3, "Fa":2,"Po" : 1, "NA": 0 , "Grvl" : 1 , "Pave" : 1}



testdata['Bath'] = testdata['BsmtFullBath']+testdata['FullBath']+((testdata['HalfBath']+testdata['BsmtHalfBath'])*0.5)

testdata.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],axis=1,inplace=True)





dict2 = {"AllPub": 4,"NoSewr" : 3,"NoSeWa":2,"ELO":1}

testdata = testdata.replace({"Utilities" : dict2})

testdata['Utilities'] = testdata['Utilities'].astype('float64')





testdata.drop(['Id'],axis=1,inplace=True)



# testdata.drop(['MasVnrType'],axis=1,inplace=True)

# testdata = testdata.fillna(testdata.mode().iloc[0])



testdata = testdata.replace({"ExterCond": dict})

testdata = testdata.replace({"ExterQual": dict})

testdata['Exter'] = testdata['ExterCond'] * testdata['ExterQual']

testdata.drop(['ExterCond','ExterQual'],axis=1,inplace=True)

testdata['Exter'] = testdata['Exter'].astype('float64')











testdata['SF'] = testdata['TotalBsmtSF'] + testdata['1stFlrSF'] + testdata['2ndFlrSF'] 

testdata['SF'] = testdata['SF'].astype('float64')

testdata.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)





testdata['Alley'].fillna('None')



dictLotShape =  {"Reg": 4, "IR1": 3, "IR2":2,"IR3" : 1}

testdata = testdata.replace({"LotShape" : dictLotShape})

testdata['LotShape'] = testdata['LotShape'].astype('float64')







dictBsmt =  {"Ex": 100, "Gd": 95, "TA": 85, "Fa":75,"Po" : 35}

testdata['BsmtQual'].fillna(0)

testdata = testdata.replace({"BsmtQual" : dictBsmt})

testdata['BsmtQual'] = testdata['BsmtQual'].astype('float64')







dictBsmt =  {"Ex": 5, "Gd": 4, "TA": 3, "Fa":2,"Po" : 1}

testdata['BsmtCond'].fillna(0)

testdata = testdata.replace({"BsmtCond" : dictBsmt})

testdata['BsmtCond'] = testdata['BsmtCond'].astype('float64')







testdata['BSMT'] = testdata['BsmtQual'] * testdata['BsmtCond']



testdata.drop(['BsmtQual' , 'BsmtCond'] , axis=1 , inplace=True)



dictMSZ = {"RH"  : "R"  ,"RL"  : "R"  , "RP"  : "R"  , "RM"  : "R" }

testdata = testdata.replace({"MSZoning" : dictMSZ})





dictBsmtExpo =  {"Gd": 3, "Av": 2, "Mn": 1, "No":0}

testdata['BsmtExposure'].fillna(0)

testdata = testdata.replace({"BsmtExposure" : dictBsmtExpo})

testdata['BsmtExposure'] = testdata['BsmtExposure'].astype('float64')







testdata = testdata.replace({"HeatingQC" : dict})

testdata['HeatingQC'] = testdata['HeatingQC'].astype('int64')





dictAir =  {"Y": '1', "N":'0'}

testdata = testdata.replace({"CentralAir" : dictAir})

testdata['CentralAir'] = testdata['CentralAir'].astype('int64')









testdata['GarageType'].fillna('None')





testdata = testdata.replace({"FireplaceQu" : dict})

testdata['FireplaceQu'].fillna(0)

testdata['FireplaceQu']= testdata['FireplaceQu'].astype('float64')





testdata['Fire'] = testdata['Fireplaces'] * testdata['FireplaceQu']



testdata.drop(['FireplaceQu' , 'Fireplaces'] , axis=1 , inplace=True)

dictGarage = {"Ex": 3, "Gd": 4, "TA": 5, "Fa":2,"Po" : 1, "NA": 0 , "Grvl" : 1 , "Pave" : 1}

testdata = testdata.replace({"GarageQual" : dictGarage})

testdata['GarageQual']= testdata['GarageQual'].astype('float64')

testdata['GarageQual'].fillna(0)



testdata = testdata.replace({"GarageCond" : dictGarage})

testdata['GarageCond']= testdata['GarageCond'].astype('float64')

testdata['GarageCond'].fillna(0)





testdata['Garage'] = testdata['GarageCars'] * testdata['GarageQual'] * testdata['GarageCond']

testdata.drop(['GarageCars','GarageQual','GarageCond'] , axis=1 , inplace=True)





testdata = testdata.replace({"PoolQC" : dict})

testdata['PoolQC'].fillna('0')

testdata['PoolQC']= testdata['PoolQC'].astype('float64')





testdata['Pool'] = testdata['PoolArea'].apply(lambda x:1 if x>0 else 0)

testdata['Pool']= testdata['Pool'].astype('float64')



testdata.drop(['PoolQC','PoolArea'] , axis=1 , inplace=True)





testdataFence = {"GdPrv"  : 5 , "MnPrv" : 4 , "GdWo"  : 3 , "MnWw" :  2}

testdata = testdata.replace({"Fence" : testdataFence})

testdata['Fence'].fillna('0')

testdata['Fence']= testdata['Fence'].astype('float64')







testdata['MiscFeature'].fillna('None')











testdata = testdata.replace({"KitchenQual": dict})

testdata['Kitch'] = testdata['KitchenAbvGr'] * testdata['KitchenQual']

testdata['Kitch'] = testdata['Kitch'].astype('float64')

testdata.drop(['KitchenAbvGr','KitchenQual'],axis=1,inplace=True)

















testdata.info()

testdata = pd.get_dummies(testdata)

testdata.info()

xyz=testdata.loc[:, ~testdata.columns.isin([])]
xyz.head()
missing_cols = set( xtrain.columns ) - set( xyz.columns )

for c in missing_cols:

    xyz[c] = 0

xyz = xyz[xtrain.columns]


preds2 = lr.predict(xyz)

print(preds2)
results = pd.DataFrame()

results['Id'] = idData['Id']

results['SalePrice'] = preds2

results.head()

results.info()
results.to_csv('submission2.csv',index=False)
xgb2.fit(xtrain,ytrain)

predsx  =xgb2.predict(xyz)

results['SalePrice'] = (predsx + preds2)/2

results.to_csv('stackingSubmissin1.csv',index=False)
results