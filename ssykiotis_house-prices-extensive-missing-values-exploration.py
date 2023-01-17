import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from pathlib import Path

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
color = sns.color_palette()

sns.set_style('darkgrid')
path = '/kaggle/input/house-prices-advanced-regression-techniques'

path = Path(path)

train_raw = pd.read_csv(path/'train.csv')

test_raw = pd.read_csv(path/'test.csv')



train = train_raw.copy(deep=True)

test = test_raw.copy(deep=True)

data_clean = [train_raw,test_raw]
print("Dataset dimensions: ")

print("Training Set: " ,train.shape)

print("Test Set: " ,test.shape)
train.head(n=10)
test.head(n=10)
print("Variables with missing values in the dataset:")

print("Training set: " ,train.isnull().any().sum())

print("Test set: ", test.isnull().any().sum())
y_train = train['SalePrice']

x_train = train.drop('SalePrice',axis = 1)



data = pd.concat([x_train,test],ignore_index= True, verify_integrity = True,copy = True)

print(data.shape)
train.plot.scatter(x = 'LotArea',y = 'SalePrice')

train.plot.scatter(x = 'GrLivArea',y = 'SalePrice')
idx_outliers =train[['GrLivArea','SalePrice']][(train['GrLivArea']>4000) & (train['SalePrice']<300000)]
pd.options.display.max_rows = 50

price_corr = train[train.notnull()].corr(method='pearson')['SalePrice'].abs()

price_corr = pd.DataFrame(price_corr)

price_corr.sort_values(by = 'SalePrice',ascending = False)
mv = data.isnull().sum()/data.shape[0]*100

mv = mv[mv>0]

mv = mv.sort_values(axis = 0,ascending = False)
f, ax = plt.subplots(figsize=(15, 10))

plt.xticks(rotation='90')

sns.barplot(mv.index,mv)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

plt.show()
garage = ['GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt']
idx = data['GarageType'].isnull()

print("Total properties without a garage: ",idx.sum())

print("Non-NaN entries in properties without a garage: ")

print(data[garage][idx].notnull().sum())
idx = data['GarageType'].isnull() & data['GarageArea'].notnull() & data['GarageCars'].notnull()

print("Total Garage Area and Cards in properties without a garage:")

print(data[['GarageCars','GarageArea']][idx].sum())
idx = data['GarageArea'].isnull()

data[['GarageArea','GarageCars','GarageQual']][idx]
bsmt = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']

idx = data['BsmtQual'].isnull()

data[bsmt][idx].notnull().sum()
idx = data['BsmtQual'].isnull() & data['BsmtCond'].notnull()

data[bsmt][idx]
idx = data['BsmtQual'].isnull() & data['BsmtFinSF1'].notnull()

data[bsmt][idx]
print(data[['BsmtFinSF1','BsmtFinSF2']][idx].sum())
idx = data['BsmtFinSF2'].isnull()

data[['BsmtCond','BsmtFinSF1','BsmtFinSF2']][idx]
idx = data['BsmtQual'].isnull() & data['BsmtFullBath'].notnull()

data[['BsmtFullBath','BsmtHalfBath']][idx].sum()
idx = data['BsmtQual'].isnull() & data['BsmtUnfSF'].notnull()

data[['BsmtUnfSF','TotalBsmtSF']][idx].sum()
idx = data['BsmtQual'].isnull() & data['BsmtUnfSF']>0

data[['BsmtUnfSF','TotalBsmtSF']][idx]
pd.options.display.max_rows = 15

idx = data['BsmtExposure'].isnull() & data['BsmtCond'].notnull()

data[bsmt][idx]
idx = data['BsmtCond'].notnull() & data['BsmtFinType1'].isnull()

data[bsmt][idx]
idx = data['BsmtCond'].notnull() & data['BsmtFinType2'].isnull()

data[bsmt][idx]
idx = data['BsmtCond'].notnull() & data['BsmtFullBath'].isnull()

print(data[bsmt][idx])

idx = data['BsmtCond'].notnull() & data['BsmtHalfBath'].isnull()

print(data[bsmt][idx])
idx = data['BsmtCond'].notnull() & data['BsmtUnfSF'].isnull()

data[bsmt][idx]
idx = data['BsmtCond'].notnull() & data['TotalBsmtSF'].isnull()

data[bsmt][idx]
dict_ext = {'Brk Cmn':'BrkComm',

            'CmentBd':'CemntBd',

            'Wd Shng': 'WdShing'

           }

data['Exterior2nd'] = data['Exterior2nd'].replace(dict_ext)
v = ['Exterior1st','Exterior2nd']

idx = data['Exterior1st']!=data['Exterior2nd']

data[v][idx]
pd.options.display.max_columns = 100



idx = data['Exterior1st'].isnull()

data[idx]
idx = data['MasVnrType'].isnull()

data[['MasVnrArea','MasVnrType']][idx]
data[data['MSZoning'].isnull()]
data[data['SaleType'].isnull()]
train = train.drop(idx_outliers.index, axis = 0)
train.plot.scatter(x = 'GrLivArea',y = 'SalePrice')
y_train = train['SalePrice']

x_train = train.drop('SalePrice',axis = 1)



del data

data = pd.concat([x_train,test],ignore_index= True, verify_integrity = True,copy=True)
dict_ext = {'Brk Cmn':'BrkComm',

            'CmentBd':'CemntBd',

            'Wd Shng': 'WdShing'

           }

data['Exterior2nd'] = data['Exterior2nd'].replace(dict_ext)
def set_value(df,value, variables):

    assert type(variables)==list,"variables must be passed on as list"

    var0,var1 = variables

    idx = df[var0].notnull() & data[var1].isnull()

    loc = df[var1][idx].index[0]

    df.at[loc,var1] = value

    return df



def findStringMostCommon(d,target,conds, tvals=None):

    assert type(conds) == list, "Targetvars must be passed on as a list"

    assert len(conds)<3

    if tvals:

        if len(conds)>1:

            cond0,cond1 = conds

            tval0,tval1 = tvals    

            selected_data = d[target][(d[cond0]==tval0) & (d[cond1]==tval1)].sort_values()

        else:

            selected_data = d[target][d[conds]==tvals].sort_values()

    else:

        conds = conds[0]

        selected_data = d[target].groupby(conds).value_counts()

    

    return selected_data.value_counts().index[0]



def set_conditional(df,target,cond):

    idx = df[target].isnull()

    for i in idx.index:

        if idx[i]:

            stats = df[target][df[cond]==df.loc[i,cond]].value_counts()

            df.at[i,target] = stats.index[0]

    return df
def fill_and_fix(df):

    df_cp = df.copy(deep= True)

    vars_cat = ['Alley','BsmtCond','BsmtFinType1','BsmtFinType2','Fence','FireplaceQu','GarageCond','GarageFinish','GarageQual','GarageType','MiscFeature','PoolQC','BsmtQual']

    

    vars_num = ['BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','GarageArea','GarageCars','LotFrontage', 'TotalBsmtSF','MasVnrArea','GarageYrBlt']

    df_cp = set_value(df_cp,'Unf',['BsmtQual','BsmtFinType2'])

    df_cp = set_value(df_cp,findStringMostCommon(df_cp,'BsmtQual',['BsmtExposure','BsmtFinType1'],['No','Unf']),['BsmtCond','BsmtQual'])

    

    for v in ['Exterior1st','Exterior2nd','MSZoning','Utilities']:

        df_cp = set_conditional(df_cp,v,'Neighborhood')

        

    for v in ['Electrical','KitchenQual']:

        df_cp = set_conditional(df_cp,v,'MSZoning')

     

    idx = df_cp['MasVnrType'].isnull() & df_cp['MasVnrArea'].notnull()

    df_cp.at[idx[idx==True].index[0],'MasVnrArea']= 0 

    

    for var in vars_cat:

        df_cp[var].fillna(value = '0',inplace = True)

        

    for var in vars_num:

        df_cp[var].fillna(value = 0,inplace = True)

        

    df_cp['BsmtExposure'].fillna(value = '0',inplace = True)

    df_cp['Functional'].fillna(value = 'Typ',inplace = True) 

    df_cp['MasVnrType'].fillna(value = 'None',inplace = True)     

    

    return df_cp
data  = fill_and_fix(data)

print("Missing values after cleaning: ",data.isnull().sum().sum())
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['hasGarage'] =  np.where(data['GarageQual']!='0', 1, 0)

data['hasBsmt'] =  np.where(data['BsmtCond']!='0', 1, 0)



data['GarageAge'] = data['YrSold']-data['GarageYrBlt']

data['GarageAge'] = np.where(data['GarageAge']<0,100,0)

data['HouseAge'] = data['YrSold']-data['YearBuilt']

data['RemodAge'] = data['YrSold']-data['YearRemodAdd']



data_clean = data.drop(labels = ['SaleType','GarageYrBlt','YearBuilt','YearRemodAdd'],axis = 1)
ordinal = ['Alley','BldgType','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual','CentralAir','Condition1','Condition2','Electrical','ExterCond','Exterior1st','Exterior2nd',\

           'ExterQual', 'Fence','FireplaceQu','Foundation','Functional','GarageCond','GarageFinish','GarageQual','GarageType','Heating','HeatingQC','HouseStyle','KitchenQual','LandContour',\

           'LandSlope','LotConfig','LotShape','MasVnrArea','MasVnrType','MiscFeature','MSSubClass','MSZoning','Neighborhood','OverallCond','OverallQual','PavedDrive','PoolQC','RoofMatl',\

           'RoofStyle','SaleCondition','Street','Utilities']
def categorify(df, var, d = None):

    df_cp = df.copy(deep=True)

    codebook = d if d else dict()  

    for v in var:

        if v not in codebook.keys():

            df_cp[v] = df_cp[v].astype('category')

            keys = np.sort(df[v].unique())

            if np.array_equal(keys,np.arange(len(keys))):

                assert bool(codebook),"No dictionary provided. Please provide a dictionary to avoid overwriting values"

            else:

                df_cp[v] = df_cp[v].cat.reorder_categories(keys,ordered=True)

                values = df_cp[v].cat.codes

                df_cp[v] = values

                codebook[v] = list(zip(keys,np.arange(len(keys))))

    return df_cp,codebook

data,codebook = categorify(data_clean,ordinal)


need_norm = ['1stFlrSF','2ndFlrSF','3SsnPorch','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','EnclosedPorch','GarageArea','MasVnrArea','OpenPorchSF','PoolArea','ScreenPorch','TotalBsmtSF','WoodDeckSF','LotArea','LotFrontage','TotalSF']
def normalize(df,need_norm):

    df = df.astype('float64')

    for v in need_norm:

        df[v] = (df[v]-df[v].mean())/df[v].std()

    return df

data_final = normalize(data,need_norm)

data_final
sns.distplot(y_train)
y_train_log = np.log(y_train)

sns.distplot(y_train_log)
data_final = data

m = train_raw.shape[0]-2

x_train = data_final.loc[:(m-1),:]

x_test = data_final.loc[m:,:]
xtrain  = x_train.drop(labels = 'Id', axis = 1)

xtest = x_test.drop(labels = 'Id',axis = 1)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,Ridge

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(xtrain)

    rmse= np.sqrt(-cross_val_score(model, xtrain, y_train_log, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
RR = Ridge(alpha=0.8)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(RR)

print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (ENet, GBoost, RR, lasso,model_xgb,model_lgb))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
%%capture

averaged_models.fit(xtrain, y_train_log)
train_pred = np.exp(averaged_models.predict(xtrain))

test_pred = np.exp(averaged_models.predict(xtest))
submission = pd.DataFrame()

submission['Id'] = x_test['Id'].astype('int32')

submission['SalePrice'] = test_pred

submission.to_csv('submission.csv',index=False)