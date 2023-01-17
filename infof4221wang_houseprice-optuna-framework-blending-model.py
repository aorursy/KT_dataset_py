# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# preprocessing

from scipy.stats import skew, kurtosis

from scipy.stats import probplot

from scipy.stats import norm

from sklearn.preprocessing import OrdinalEncoder

from scipy.special import boxcox1p

# model data

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

import optuna

# Linear regressor

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, BayesianRidge

# Kernel regressor

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

# Neural Network

from sklearn.neural_network import MLPRegressor

# Tree based ensemble regressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

from xgboost.sklearn import XGBRegressor

from lightgbm import LGBMRegressor

# Stack regressor

from sklearn.ensemble import StackingRegressor

# Neural Network

import tensorflow as tf

from tensorflow import keras
# load dataset

raw_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

raw_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')

raw_train.info()

raw_test.info()
raw_train.head()
train = raw_train.copy()

test = raw_test.copy()

train_test = pd.concat([train.drop(columns=['SalePrice']),test])
# feature correlation with 'SalePrice'

train.corr()['SalePrice'].sort_values(key=abs,ascending=False)
sns.scatterplot(x='GrLivArea',y='SalePrice',data = train)
train_test = train_test.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
# show number of missing values in each feature

train.isnull().sum().sort_values(ascending=False).head(20)
test.isnull().sum().sort_values(ascending=False).head(30)
# NA literally means None

for col in ['PoolQC','MiscFeature','Alley','Fence','BsmtQual','BsmtCond','BsmtExposure',

            'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish',

            'GarageQual','GarageCond','MasVnrType']:

    train_test[col].fillna('None',inplace=True)

# features with small num missing, use mode to cover

for col in ['Functional','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType','MSZoning','Utilities']:

    train_test[col].fillna(train_test[col].mode()[0],inplace=True)

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1',

            'BsmtUnfSF','BsmtFinSF2','BsmtHalfBath','BsmtHalfBath','BsmtFullBath','TotalBsmtSF']:

    train_test[col].fillna(0,inplace=True)
train.corr()['LotFrontage'].sort_values(key=abs, ascending=False)
# Use random forest regressor to predict 'LotFrontage'

from sklearn.ensemble import RandomForestRegressor



#choose related data to predict Lotfrontage on training set

LF_df =train_test[['LotFrontage','1stFlrSF','LotArea','GrLivArea', 'TotalBsmtSF', 'MSSubClass', 'TotRmsAbvGrd','GarageArea','GarageCars']]

LF_df_notnull = LF_df.loc[train_test['LotFrontage'].notnull()]

LF_df_isnull = LF_df.loc[(train_test['LotFrontage'].isnull())]

Xtr_LF = pd.get_dummies(LF_df_notnull.drop(columns=['LotFrontage']))

Xte_LF = pd.get_dummies(LF_df_isnull.drop(columns=['LotFrontage']))

Y_LF = LF_df_notnull.LotFrontage

# use RandomForestRegression to train data

RFR = RandomForestRegressor(n_estimators=60, n_jobs=-1)

RFR.fit(Xtr_LF,Y_LF)

predict = RFR.predict(Xte_LF)

train_test.loc[train_test['LotFrontage'].isnull(), ['LotFrontage']]= predict

RFR.score(Xtr_LF,Y_LF)
# # Use the same model on test set to avoid data leakage

# LF_df =test[['LotFrontage','1stFlrSF','LotArea','GrLivArea', 'TotalBsmtSF', 'MSSubClass', 'TotRmsAbvGrd','GarageArea','GarageCars']]

# LF_df_notnull = LF_df.loc[test['LotFrontage'].notnull()]

# LF_df_isnull = LF_df.loc[(test['LotFrontage'].isnull())]

# Xte_LF = pd.get_dummies(LF_df_isnull.drop(columns=['LotFrontage']))

# Y_LF = LF_df_notnull.LotFrontage

# predict = RFR.predict(Xte_LF)

# test.loc[test['LotFrontage'].isnull(), ['LotFrontage']]= predict
train_test.isnull().sum().sort_values(ascending=False).head(30)
train[['Neighborhood','SalePrice']].groupby('Neighborhood').mean().sort_values('SalePrice')
train_test["oNeighborhood"] = train_test.Neighborhood.map({'MeadowV':1,

                                                'IDOTRR':2, 'BrDale':2,

                                               'OldTown':3, 'Edwards':3, 'BrkSide':3,

                                               'Sawyer':4, 'Blueste':4, 'SWISU':4, 'NAmes':4,

                                               'NPkVill':5, 'Mitchel':5,

                                               'SawyerW':6, 'Gilbert':6, 'NWAmes':6,

                                               'Blmngtn':7, 'CollgCr':7, 'ClearCr':7, 'Crawfor':7,

                                               'Veenker':8, 'Somerst':8, 'Timber':8,

                                               'StoneBr':9,

                                               'NoRidge':10, 'NridgHt':10})
# create simple boolean features

train_test['Has_pool'] = train_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

train_test['Has_2ndfloor'] = train_test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

train_test['Has_garage'] = train_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

train_test['Has_bsmt'] = train_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

train_test['Has_fireplace'] = train_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

binary_features = ['Has_pool','Has_2ndfloor','Has_garage','Has_bsmt','Has_fireplace']
# train_test['MSSubClass'] = train_test['MSSubClass'].astype('category')

# train_test['OverallCond'] = train_test['OverallCond'].astype('category')

# train_test['YrSold'] = train_test['YrSold'].astype('category')

# train_test['MoSold'] = train_test['MoSold'].astype('category')
ordinal_features = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 

                    'GarageCond', 'ExterQual', 'ExterCond','HeatingQC', 

                    'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 

                    'Functional', 'BsmtExposure', 'GarageFinish', 

                    'LandSlope','LotShape', 'PavedDrive', 'Street', 

                    'CentralAir']
bin_map  = {'TA':2,'Gd':3, 'Fa':1,'Ex':4,'Po':1,'None':0,'Y':1,'N':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,

            "No" : 2, "Mn" : 2, "Av": 3,"Gd" : 4,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 4, "ALQ" : 5, "GLQ" : 6,

            'Typ':7, 'Min1':6, "Min2":5, 'Mod':4, "Maj1":3, "Maj2":2, "Sev":1, "Sal":0,'Grvl':0,'Pave':1

            }

train_test['ExterQual'] = train_test['ExterQual'].map(bin_map)

train_test['ExterCond'] = train_test['ExterCond'].map(bin_map)

train_test['BsmtCond'] = train_test['BsmtCond'].map(bin_map)

train_test['BsmtQual'] = train_test['BsmtQual'].map(bin_map)

train_test['HeatingQC'] = train_test['HeatingQC'].map(bin_map)

train_test['KitchenQual'] = train_test['KitchenQual'].map(bin_map)

train_test['FireplaceQu'] = train_test['FireplaceQu'].map(bin_map)

train_test['Functional'] = train_test['Functional'].map(bin_map)

train_test['Street'] = train_test['Street'].map(bin_map)

train_test['GarageQual'] = train_test['GarageQual'].map(bin_map)

train_test['GarageCond'] = train_test['GarageCond'].map(bin_map)

train_test['CentralAir'] = train_test['CentralAir'].map(bin_map)

train_test['LotShape'] = train_test['LotShape'].map(bin_map)

train_test['BsmtExposure'] = train_test['BsmtExposure'].map(bin_map)

train_test['BsmtFinType1'] = train_test['BsmtFinType1'].map(bin_map)

train_test['BsmtFinType2'] = train_test['BsmtFinType2'].map(bin_map)



PavedDrive =   {"N" : 0, "P" : 1, "Y" : 2}

train_test['PavedDrive'] = train_test['PavedDrive'].map(PavedDrive)

GarageFinish = {'Fin':3, 'RFn':2, 'Unf':1, 'NA':0,'None':0}

train_test['GarageFinish'] = train_test['GarageFinish'].map(GarageFinish)

LandSlope = {'Gtl':2, 'Mod':1, 'Sev':0}

train_test['LandSlope'] = train_test['LandSlope'].map(LandSlope)

train_test["MSZoning"] = train_test.MSZoning.map({'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4})

train_test["oCondition1"] = train_test.Condition1.map({'Artery':1,

                                           'Feedr':2, 'RRAe':2,

                                           'Norm':3, 'RRAn':3,

                                           'PosN':4, 'RRNe':4,

                                           'PosA':5 ,'RRNn':5})
# generalization features

train_test['Total_SF']=train_test['TotalBsmtSF'] + train_test['1stFlrSF'] + train_test['2ndFlrSF']



train_test['Total_sqr_footage'] = (train_test['BsmtFinSF1'] + train_test['BsmtFinSF2'] +

                                 train_test['1stFlrSF'] + train_test['2ndFlrSF'])



train_test['Total_Bathrooms'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) +

                               train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))



train_test['Total_porch_sf'] = (train_test['OpenPorchSF'] + train_test['3SsnPorch'] +

                              train_test['EnclosedPorch'] + train_test['ScreenPorch'] +

                              train_test['WoodDeckSF'])



train_test["+_TotalHouse_OverallQual"] = train_test["Total_SF"] * train_test["OverallQual"]

train_test["+_GrLivArea_OverallQual"] = train_test["GrLivArea"] * train_test["OverallQual"]

train_test["+_oMSZoning_TotalHouse"] = train_test["MSZoning"] * train_test["Total_SF"]

train_test["+_oMSZoning_OverallQual"] = train_test["MSZoning"] + train_test["OverallQual"]

train_test["+_oNeighborhood_TotalHouse"] = train_test["oNeighborhood"] * train_test["Total_SF"]

train_test["+_oNeighborhood_OverallQual"] = train_test["oNeighborhood"] + train_test["OverallQual"]

train_test["+_oNeighborhood_YearBuilt"] = train_test["oNeighborhood"] + train_test["YearBuilt"]

train_test["+_BsmtFinSF1_OverallQual"] = train_test["BsmtFinSF1"] * train_test["OverallQual"]



train_test["-_oFunctional_TotalHouse"] = train_test["Functional"] * train_test["Total_SF"]

train_test["-_oFunctional_OverallQual"] = train_test["Functional"] + train_test["OverallQual"]

train_test["-_LotArea_OverallQual"] = train_test["LotArea"] * train_test["OverallQual"]

train_test["-_TotalHouse_LotArea"] = train_test["Total_SF"] + train_test["LotArea"]

train_test["-_oCondition1_TotalHouse"] = train_test["oCondition1"] * train_test["Total_SF"]

train_test["-_oCondition1_OverallQual"] = train_test["oCondition1"] + train_test["OverallQual"]





train_test["Bsmt"] = train_test["BsmtFinSF1"] + train_test["BsmtFinSF2"] + train_test["BsmtUnfSF"]

train_test["Rooms"] = train_test["FullBath"]+train_test["TotRmsAbvGrd"]

train_test["TotalPlace"] = train_test["TotalBsmtSF"] + train_test["1stFlrSF"] + train_test["2ndFlrSF"] + train_test["GarageArea"] +train_test["OpenPorchSF"]+train_test["EnclosedPorch"]+train_test["3SsnPorch"]+train_test["ScreenPorch"]
def plot_skew(feature):

    """

    Function to plot distribution and probability(w.r.t quantiles of normal distribution)

    """

    fig, axs = plt.subplots(figsize=(20,10),ncols=2)

    sns.distplot(feature,kde=True,fit=norm,ax=axs[0])

    # Generates a probability plot of sample data against the quantiles of a specified theoretical distribution (the normal distribution by default).

    f=probplot(feature, plot=plt)

    print('Skewness: {:f}'.format(feature.skew()))

    print('Kurtosis: {:f}'.format(feature.kurtosis()))
plot_skew(train.SalePrice)
train['SalePrice_log'] = np.log1p(train.SalePrice)

plot_skew(train.SalePrice_log)
# fix skew in all numeric data

train_test_num = pd.DataFrame(train_test.select_dtypes(['float64','int32','int64']))

train_test_num.drop(columns=ordinal_features,inplace=True)

train_test_num.drop(columns=binary_features,inplace=True)
train_test_skewed = train_test_num.apply(lambda x:skew(x)).sort_values(key=abs,ascending=False)

train_test_skewed_df = pd.DataFrame({'Skew':train_test_skewed})

train_test_skewed_df
skewed_features = train_test_skewed_df[train_test_skewed_df.Skew>0.5].index

lmbd = 0.15

for sk in skewed_features:

    if sk not in  ['SalePrice','SalePrice_log']:

        train_test[sk] = boxcox1p(train_test[sk], lmbd)
train_test = pd.get_dummies(train_test)

train_test.isna().sum().sort_values(ascending=False)
lasso_model = Lasso(alpha=0.0001).fit(train_test[:1458],train.SalePrice_log)

ft_im_df = pd.DataFrame({"Feature Importance":lasso_model.coef_}, index=train_test.columns)

ft_im_df.sort_values('Feature Importance', key=abs,ascending=False).head(20)
CV = KFold(n_splits=5,shuffle=True,random_state=42)

rb_scaler = RobustScaler()
def rmse(model,x,y):

    return -(cross_val_score(model, x, y,cv=CV,scoring='neg_root_mean_squared_error',n_jobs=-1))
train_test = rb_scaler.fit_transform(train_test)

X = train_test[:1458]

X_test = train_test[1458:]

Y = train.SalePrice_log

X.shape
pca = PCA(n_components=0.9999)

train_test_pc = pca.fit_transform(train_test)

X_PC = train_test_pc[:1458]

X_test_PC = train_test_pc[1458:]

X_PC.shape
n_trials = 500
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     alpha = trial.suggest_loguniform('alpha',1e-5,1e5)

#     # define classifier

#     reg = Ridge(alpha=alpha,random_state=42)

#     # define evaluation matrix as objective to return

#     score = rmse(reg,X,Y)

#     return score.mean()

# # create study

# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# ridge_best_params = study.best_params

ridge_best_params = {'alpha': 36.31161037749121}

ridge_best_model = Ridge(**ridge_best_params)
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     alpha = trial.suggest_loguniform('alpha',1e-5,1e3)

#     # define classifier

#     reg = Lasso(alpha=alpha,random_state=42)

#     # define evaluation matrix as objective to return

#     score = rmse(reg,X_PC,Y)

#     return score.mean()

# # create study

# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# lasso_best_params = study.best_params

lasso_best_params = {'alpha': 0.0007350225758112973}

lasso_best_model = Lasso(**lasso_best_params)
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     alpha = trial.suggest_loguniform('alpha',1e-5,1e3)

#     l1_ratio = trial.suggest_discrete_uniform('l1_ratio',0.1,0.9,0.1)

#     # define classifier

#     reg = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)

#     # define evaluation matrix as objective to return

#     score = rmse(reg,X_PC,Y)

#     return score.mean()

# # create study

# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# elastic_best_params = study.best_params

elastic_best_params = {'alpha': 0.004962477837834253, 'l1_ratio': 0.1}

elastic_best_model = ElasticNet(**elastic_best_params)
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     epsilon = trial.suggest_loguniform('epsilon',1e-4,1e2)# specifies the epsilon-tube within which no penalty

#     kernel = trial.suggest_categorical('kernel',['poly','rbf'])

#     gamma = trial.suggest_loguniform('gamma',1e-4,1e4) # kernel coifficient

#     C = trial.suggest_loguniform('C',1e-4,1e4) # inversed regularization param

#     # define classifier

#     reg = SVR(epsilon =epsilon, C=C,kernel=kernel,gamma = gamma)

#     # define evaluation matrix as objective to return

#     score = rmse(reg,X_PC,Y)

#     return score.mean()

# # create study

# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
SVR_best_params = {'epsilon': 0.0007028077160999125, 'kernel': 'rbf', 

                   'gamma': 0.0001032023906441773, 'C': 20.673761351762828}     

# SVR_best_params = study.best_params

SVR_best_model = SVR(**SVR_best_params)
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     alpha = trial.suggest_loguniform('alpha',1e-4,1e3)

#     kernel = trial.suggest_categorical('kernel',['polynomial','rbf'])

#     gamma = trial.suggest_loguniform('gamma',1e-4,1e3)

#     degree = trial.suggest_int('degree',3,5,1)

#     # define classifier

#     reg = KernelRidge(alpha=alpha,kernel=kernel,gamma = gamma,degree = degree)

#     # define evaluation matrix as objective to return

#     score = rmse(reg,X_PC,Y)

#     return score.mean()

# # create study

# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# KNR_best_params = study.best_params

KNR_best_params = {'alpha': 0.017852963690058583, 'kernel': 'polynomial', 

                   'gamma': 0.00010010824404031089, 'degree': 4}   

KNR_best_model = KernelRidge(**KNR_best_params)
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     alpha_1 = trial.suggest_loguniform('alpha_1',1e-8,1e-5)

#     alpha_2 = trial.suggest_loguniform('alpha_2',1e-8,1e-5)

#     lambda_1 = trial.suggest_loguniform('lambda_1',1e-8,1e-5)

#     lambda_2 = trial.suggest_loguniform('lambda_2',1e-8,1e-5)

#     reg = BayesianRidge(alpha_1=alpha_1,alpha_2=alpha_1,lambda_1=lambda_1,lambda_2=lambda_2)

#     # define evaluation matrix as objective to return

#     score = rmse(reg,X_PC,Y)

#     return score.mean()

# # create study

# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=n_trials,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# Bayesian_best_params = study.best_params

Bayesian_best_params = {'alpha_1': 9.974811163419825e-06, 'alpha_2': 5.050535825995536e-06,

                        'lambda_1': 9.97006607898247e-06, 'lambda_2': 1.004807953151607e-08} 

Bayesian_best_model = BayesianRidge(**Bayesian_best_params)
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     n_layers = trial.suggest_int('n_layers', 1,1) # no. of hidden layers 

#     layers = []

#     for i in range(n_layers):

#         layers.append(trial.suggest_int('n_units_{}'.format(i+1), 100,300,20)) # no. of hidden unit

#     activation=trial.suggest_categorical('activation',[ 'tanh', 'relu']) # activation function 

#     alpha=trial.suggest_loguniform('alpha',1e-4,100) #L2 penalty (regularization term) parameter.

#     # define classifier

#     reg =  MLPRegressor(random_state=42,

#                         solver='adam',

#                         activation=activation,

#                         alpha=alpha,

#                         hidden_layer_sizes=(layers),

#                         max_iter=1000,

#                         learning_rate='adaptive',

#                         batch_size=64,

#                         learning_rate_init=0.05,

#                         early_stopping=True)

#     # define evaluation matrix as objective to return

#     score = rmse(reg,X_PC,Y)

#     return score.mean()

# # create study

# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials = 1000,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# MLP_best_params = study.best_params

MLP_best_model = MLPRegressor(random_state=42,

                        solver='adam',

                        activation='tanh',

                        alpha= 2.9344049246725894,

                        hidden_layer_sizes=(120),

                        max_iter=1000,

                        learning_rate='adaptive',

                        batch_size=300,

                        learning_rate_init=0.5,

                        early_stopping=True)
# # paramerter tuning using Optuna framework

# def objective(trial):

#     # define parameters' sample space and sample type

#     n_estimators = trial.suggest_int('n_estimators',50,500,50) 

#     learning_rate = trial.suggest_loguniform('learning_rate',1e-5,1e-3)

#     max_depth = trial.suggest_int('max_depth',3,10,1)

#     booster = trial.suggest_categorical('booster',['gbtree','gblinear','dart'])

#     gamma = trial.suggest_loguniform('gamma', 1e-4,1e2)

#     reg_alpha = trial.suggest_loguniform('reg_alpha',1e-3,1e2) # L1 regularization term on weights.

#     reg_lambda = trial.suggest_loguniform('reg_lambda',1e-3,1e2) # L2 regularization term on weights.

#     colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree',0.4,1.0,0.2) # sub-features to use 

#     subsample = trial.suggest_discrete_uniform('subsample',0.8,1.0,0.1) # subsamples to use

#     # define classifier

#     reg = XGBRegressor(n_estimators=n_estimators, 

#                        objective='reg:pseudohubererror',

#                        learning_rate=learning_rate,

#                        max_depth=max_depth,

#                        booster = booster,

#                        gamma = gamma,

#                        reg_alpha = reg_alpha,

#                        reg_lambda = reg_lambda,

#                        colsample_bytree = colsample_bytree,

#                        subsample=subsample,

#                        n_jobs=-1,

#                        random_state=42)

#     # define evaluation matrix as objective to return

#     score = rmse(reg,X_PC,Y)

#     return score.mean()

# # create study

# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=1000,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# XGB_best_params = study.best_params

XGB_best_params= {'n_estimators': 230, 'learning_rate': 0.05064350395549183, 

                  'max_depth': 2, 'booster': 'gblinear', 'gamma': 0.0005562582585578549,

                  'reg_alpha': 0.0010013453661588325, 'reg_lambda': 0.0010065125775610678, 

                  'colsample_bytree': 0.6000000000000001, 'subsample': 1.0}

XGB_best_model = XGBRegressor(**XGB_best_params,objective='reg:squaredlogerror',

                              random_state=42,n_jobs=-1)
# def objective(trial):

#     # define parameters' sample space and sample type

#     boosting_type  = trial.suggest_categorical('boosting_type',['gbdt','dart'])

#     num_leaves = trial.suggest_int('num_leaves',10,30,5)

#     learning_rate  = trial.suggest_loguniform('learning_rate',1e-5,1e-1)

#     n_estimators = trial.suggest_int('n_estimators',10,500,10) 

#     max_depth = trial.suggest_int('max_depth',1,7,1) #-1 means no limit

#     min_split_gain  = trial.suggest_loguniform('min_samples_split',1e-3,1e-1) #Minimum loss reduction required to make a further partition on a leaf node of the tree

#     min_child_samples  = trial.suggest_int('min_samples_leaf',2,122,20) # minimum num of samples required to be a leaf node

#     reg_alpha = trial.suggest_loguniform('reg_alpha',1e-3,1e2) # L1 regularization term on weights.

#     reg_lambda = trial.suggest_loguniform('reg_lambda',1e-3,1e2) # L2 regularization term on weights.

#     colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree',0.4,1.0,0.2) # sub-features to use 

#     subsample = trial.suggest_discrete_uniform('subsample',0.8,1.0,0.1) # subsamples to use

#     # define classifier

#     reg = LGBMRegressor(n_estimators=n_estimators,

#                                 boosting_type=boosting_type,

#                                 num_leaves = num_leaves,

#                                 max_depth=max_depth,

#                                 learning_rate=learning_rate,

#                                 min_split_gain = min_split_gain,

#                                 min_child_samples=min_child_samples,

#                                 reg_alpha=reg_alpha,

#                                 reg_lambda=reg_lambda,

#                                 n_jobs=-1,

#                                 random_state=42)

# #                                 device = 'gpu')

#     # define evaluation matrix as objective to return

#     score = rmse(reg,X_PC,Y)

#     return score.mean()

# # create study

# study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler())

# # run study to find best objective

# study.optimize(objective,n_trials=1000,n_jobs=-1)

# print('Best model parameters:{} '.format(study.best_params))

# print('Best score: {:.6f}'.format(study.best_value))
# LGBM_best_params = study.best_params

LGBM_best_params={'boosting_type': 'gbdt', 'num_leaves': 10, 

                  'learning_rate': 0.0623344941102757, 'n_estimators': 350, 

                  'max_depth': 6, 'min_samples_split': 0.003721770553125834, 

                  'min_samples_leaf': 2, 'reg_alpha': 0.011733455705754578, 

                  'reg_lambda': 0.0692034942836032, 'colsample_bytree': 0.6000000000000001, 

                  'subsample': 0.9}

LGBM_best_model = LGBMRegressor(**LGBM_best_params,random_state=42,n_jobs=-1)
def model_eval():

    rmse_mean = []

    std = []

    rmser = []

    regressors=['Lasso','Ridge','Elastic_Net',

                'Kernel_Ridge','SVR','Bayesian',

                'MLP','XGB','LGBM']

    models=[lasso_best_model,ridge_best_model,

            elastic_best_model,KNR_best_model,

            SVR_best_model,Bayesian_best_model,

            MLP_best_model,XGB_best_model, 

            LGBM_best_model]

    for model in models:

        cv_result = rmse(model,X_PC,Y)

        rmse_mean.append(cv_result.mean())

        std.append(cv_result.std())

        rmser.append(cv_result)

    performance_df=pd.DataFrame({'CV_Mean':rmse_mean,'Std':std},index=regressors)

    return performance_df



performance_df = model_eval()

performance_df.sort_values('CV_Mean')
# ('SVR',SVR_best_model),('Kernel_ridge',KNR_best_model), ('ElasticNet',elastic_best_model), 

#                                          ('Ridge',ridge_best_model), 

#                                          ('Lasso', lasso_best_model),

#                                          ('Bayesian',Bayesian_best_model)

stacking = StackingRegressor(estimators=[('SVR',SVR_best_model), 

                                         ('Kernel_ridge',KNR_best_model),

                                         ('ElasticNet',elastic_best_model), 

                                         ('Ridge',ridge_best_model), 

                                         ('Lasso', lasso_best_model),

                                         ('Bayesian',Bayesian_best_model), 

                                         ('MLP',MLP_best_model), 

                                         ('LGBM',LGBM_best_model),

                                         ('XGB',XGB_best_model)],

                             final_estimator=KernelRidge(),

                             cv=5,

                             n_jobs=-1)

cv_result = rmse(stacking,X_PC,Y)

stk_acc = cv_result.mean()

stk_std = cv_result.std()

performance_df.loc['Stacking'] = {'CV_Mean':stk_acc, 'Std':stk_std}

performance_df.sort_values(by=['CV_Mean'])
stack_y = stacking.fit(X_PC,Y).predict(X_test_PC)

elastic_y = elastic_best_model.fit(X_PC,Y).predict(X_test_PC)

ridge_y = ridge_best_model.fit(X_PC,Y).predict(X_test_PC)

lasso_y = lasso_best_model.fit(X_PC,Y).predict(X_test_PC)

SVR_y = SVR_best_model.fit(X_PC,Y).predict(X_test_PC)

KNR_y = KNR_best_model.fit(X_PC,Y).predict(X_test_PC)

Bayesian_y = Bayesian_best_model.fit(X_PC,Y).predict(X_test_PC)

XGB_y = XGB_best_model.fit(X_PC,Y).predict(X_test_PC)

# 0.12393

# blend = 1* stack_y + 0.0 * elastic_y + 0.0*ridge_y + 0.0*lasso_y+ 0*SVR_y

# 0.12233

blend = 0.0 * stack_y + 0.1 * elastic_y + 0.9*ridge_y + 0.0*lasso_y+ 0.0*SVR_y

# 0.12579

# blend = 0.7 * stack_y + 0.1 * elastic_y + 0.1*ridge_y + 0.1*lasso_y+ 0.0*SVR_y

# 0.12489

# blend = 0.0 * stack_y + 0.0 * elastic_y + 0.0*ridge_y + 0.0*lasso_y+ 0.0*SVR_y+0.0*KNR_y

# 0.12312

# blend = XGB_y

# blend = ridge_y

sub_pre = np.exp(blend)

sub_pre = np.around(sub_pre,decimals=-1)

sub_pre


sub_pd = pd.DataFrame({'Id':test.index,'SalePrice':sub_pre})

sub_pd.to_csv('submit.csv' ,index=False)