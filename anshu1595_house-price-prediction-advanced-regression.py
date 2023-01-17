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
import warnings

warnings.filterwarnings('ignore')
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

import scipy.stats
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.columns,test.columns
corrmat=train.corr()

plt.figure(figsize=(12,8))

sns.heatmap(corrmat, cmap=plt.cm.RdBu_r)

plt.show()
mask = np.triu(np.ones_like(corrmat, dtype=bool))

plt.figure(figsize=(14,10))

sns.heatmap(corrmat, mask=mask, center=0,cmap=plt.cm.RdBu_r, linewidths=0.1,annot=False,fmt=".1f")

plt.show()
tri_df=corrmat.mask(mask)

print([c for c in tri_df.columns if any(tri_df[c]>0.65)])
columns_to_work_engg = ['GarageYrBlt','YearRemodAdd','1stFlrSF','2ndFlrSF','BedroomAbvGr','GarageCars']
[c for c in train.columns if '1st' in c]
top_corr_cols = corrmat['SalePrice'].nlargest(11).index.tolist()

print(top_corr_cols)
sns.pairplot(train[top_corr_cols])
train.plot.scatter('GrLivArea','SalePrice')

train.GrLivArea.nlargest()
train_1=train[train['GrLivArea']<4500]
train_1.plot.scatter('TotalBsmtSF','SalePrice')

train_1.TotalBsmtSF.nlargest()
train_1=train[train['TotalBsmtSF']<3100]
train_1.plot.scatter('GarageArea','SalePrice')

train_1.GarageArea.nlargest()
train_1=train[train['GarageArea']<1200]
print(top_corr_cols)
for c in ['SalePrice', 'GrLivArea','GarageArea', 'TotalBsmtSF', 'YearBuilt']:

    plt.figure(figsize=(12,3))

    plt.subplot(1,2, 1)

    sns.distplot(train_1[c], fit=norm)

    plt.subplot(1,2,2)

    _ = scipy.stats.probplot(train_1[c],plot=plt)
for c in ['SalePrice','GrLivArea','GarageArea','TotalBsmtSF']:

    plt.figure(figsize=(25,4))

    plt.subplot(1,4, 1)

    sns.distplot(train_1[c], fit=norm)

    plt.subplot(1,4,2)

    _ = scipy.stats.probplot(train_1[c],plot=plt)

    plt.subplot(1,4, 3)

    sns.distplot(np.log1p(train_1[c]), fit=norm)

    plt.subplot(1,4,4)

    _ = scipy.stats.probplot(np.log1p(train_1[c]),plot=plt)
num_col=[c for c in train_1.columns if train_1[c].dtype !='object']
skew_before=[]

skew_after=[]

for c in num_col:

    skew_before.append(train_1[c].skew())

    skew_after.append(np.log1p(train_1[c]).skew())

skew_df=pd.DataFrame(zip(skew_before,skew_after), columns=['skew_before','skew_after'],index=num_col)

skew_df = skew_df[abs(skew_df['skew_before'])>abs(skew_df['skew_after'])] 

skew_df['skew_diff'] = abs(skew_df['skew_before'])-abs(skew_df['skew_after'])

skew_df[skew_df['skew_diff']>1]
from sklearn.preprocessing import PowerTransformer

from scipy.stats import skew
log=PowerTransformer()

skew_before=[]

skew_after=[]

for c in num_col:

    skew_before.append(train_1[c].skew())

    log.fit(train_1[c].values.reshape(-1,1))

    skew_after.append(skew((log.transform(train_1[c].values.reshape(-1,1))))[0])

skew_df=pd.DataFrame(zip(skew_before,skew_after), columns=['skew_before','skew_after'],index=num_col)

skew_df = skew_df[abs(skew_df['skew_before'])>abs(skew_df['skew_after'])] 

skew_df['skew_diff'] = abs(skew_df['skew_before'])-abs(skew_df['skew_after'])

skew_df[skew_df['skew_diff']>1]
columns_to_work_log=['LotArea', 'BsmtFinSF2', 'LowQualFinSF', 'GrLivArea', 'KitchenAbvGr', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']
train_1['MiscVal'].nlargest()
train_1.plot.scatter('MiscVal','SalePrice')
columns_to_work_drop=['MiscVal']
num_cols=[c for c in train_1.columns if train_1[c].dtype !='object']

print(num_cols)
columns_to_work_preprocess = ['MSSubClass']
cat_cols=[c for c in train_1.columns if train_1[c].dtype =='object']

print(num_cols)
y = train_1['SalePrice'].reset_index(drop=True)

feature_train = train_1.drop('SalePrice',axis=1)



print(feature_train.shape,y.shape,test.shape)
all_data = pd.concat([feature_train,test],ignore_index=True).reset_index(drop=True)

all_data.shape
print(columns_to_work_preprocess, columns_to_work_drop, columns_to_work_engg, columns_to_work_log)
all_data['MSSubClass'] = all_data['MSSubClass'].astype('object')

all_data.MSSubClass.dtype
all_data.isna().sum().nlargest(15)
columns_to_work_drop.extend(['PoolQC','MiscFeature','Alley','Fence','FireplaceQU','GarageYrBlt','GarageCars'])

print(columns_to_work_drop)
all_data.drop(['MiscVal', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageYrBlt','GarageCars'], axis=1,inplace=True)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
for c in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','MasVnrType']:

    all_data[c].fillna('None', inplace=True)
all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].mean(), inplace=True)

all_data['HasMasVnr'] = all_data['MasVnrType'].apply(lambda x: 0 if x == 'None' else 1)

all_data['HasMasVnr'] = all_data['HasMasVnr'].astype('object')
all_data['MSZoning']=all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x:x.fillna(x.mode()[0]))
for c in ['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageArea']:

    all_data[c].fillna(0,inplace=True)
for c in ['Utilities', 'Functional', 'Exterior1st', 'Exterior2nd', 'Electrical','KitchenQual', 'SaleType']:

    all_data[c].fillna(all_data[c].mode()[0], inplace=True)
all_data.isna().sum().nlargest()
print([c for c in all_data.columns if (all_data[c].value_counts().iloc[0]>2700) & (all_data[c].dtype=='object')])
all_data.drop(['Street', 'Utilities', 'LandSlope', 'Condition2', 'RoofMatl', 'Heating','Functional'], axis=1, inplace=True)
all_data.shape
columns_to_work_engg
mask=all_data['YearRemodAdd']==all_data['YearBuilt']

all_data['RemodAdd'] = all_data['YearRemodAdd'].mask(mask)

all_data['RemodAdd'] = all_data['RemodAdd'].fillna(0)

all_data['RemodAdd'] = all_data['RemodAdd'].apply(lambda x:1 if x != 0 else 0).astype('object')
all_data.drop('YearRemodAdd', axis=1, inplace=True)
corrmat=all_data.corr()

mask = np.triu(np.ones_like(corrmat, dtype=bool))

plt.figure(figsize=(15,10))

sns.heatmap(corrmat, mask=mask, center=0,cmap=plt.cm.RdBu_r, linewidths=0.1,annot=True,fmt=".1f")

plt.show()
all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



all_data['TotalBathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])
all_data['has1stfloor'] = all_data['1stFlrSF'].apply(lambda x: 1 if x > 0 else 0).astype('object')



all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0).astype('object')



all_data['hasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0).astype('object')



all_data['hasOpenPorch'] = all_data['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0).astype('object')



all_data['has3SsnPorch'] = all_data['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0).astype('object')



all_data['hasEnclosedPorch'] = all_data['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0).astype('object')



all_data['hasScreenPorch'] = all_data['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0).astype('object')



all_data['hasWoodDeckPorch'] = all_data['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0).astype('object')
all_data.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath','OpenPorchSF','3SsnPorch',

               'EnclosedPorch','ScreenPorch','WoodDeckSF'],

              axis=1,inplace=True)
all_data.drop(['TotRmsAbvGrd'],axis=1,inplace=True)
all_data.drop('Id',axis=1,inplace=True)
all_data.shape
all_data_dum = pd.get_dummies(all_data, drop_first=True)

all_data_dum.shape
X = all_data_dum.iloc[:len(y),:]

X_test = all_data_dum.iloc[len(y):,:].reset_index(drop=True)

X.shape, y.shape, X_test.shape, train.shape, test.shape
columns_to_work_log
num_col=[c for c in all_data.columns if all_data[c].dtype !='object']

print(num_col)
log=PowerTransformer()

skew_before=[]

skew_after=[]

for c in num_col:

    skew_before.append(X[c].skew())

    log.fit(X[c].values.reshape(-1,1))

    skew_after.append(skew((log.transform(X[c].values.reshape(-1,1))))[0])

skew_df=pd.DataFrame(zip(skew_before,skew_after), columns=['skew_before','skew_after'],index=num_col)

skew_df = skew_df[abs(skew_df['skew_before'])>abs(skew_df['skew_after'])] 

skew_df['skew_diff'] = abs(skew_df['skew_before'])-abs(skew_df['skew_after'])

skew_df = skew_df[skew_df['skew_diff']>0.1]

columns_to_work_log = list(skew_df.index)

skew_df
print(columns_to_work_log)
logx = PowerTransformer()

for c in ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 

          'LowQualFinSF', 'GrLivArea', 'KitchenAbvGr', 'Fireplaces', 'PoolArea', 'MoSold', 'TotalSF', 'TotalBathrooms', 'TotalPorchSF']:

    X[c] = logx.fit_transform(X[[c]])

    X_test[c] = logx.transform(X_test[[c]])
logy = PowerTransformer()

y = logy.fit_transform(y.values.reshape(-1,1))
skew_train=[]

skew_test=[]

for c in num_col:

    skew_train.append(X[c].skew())

    skew_test.append(X_test[c].skew())

skewness_df=pd.DataFrame(zip(skew_train,skew_test), columns=['skew_train','skew_test'],index=num_col)

skewness_df
X.shape, y.shape, X_test.shape
from xgboost import XGBRegressor

from sklearn.linear_model import ElasticNet, Lasso, LassoLarsCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.feature_selection import RFE

from sklearn.decomposition import PCA
def rmse_cv(model):

    neg_mse = cross_val_score(model, X,y, cv=5, scoring='neg_mean_squared_error')

    rmse=np.sqrt(-neg_mse)

    print('RMSE mean: {0:.5f} , RMSE std: {1:.5f}'.format(rmse.mean(), rmse.std()))
def rmse_cv_return(model):

    neg_mse = cross_val_score(model, X,y, cv=5, scoring='neg_mean_squared_error')

    rmse=np.sqrt(-neg_mse)

    return rmse.mean()
model_xgb = XGBRegressor(tree_method='gpu_hist',n_jobs=-1,booster='gbtree',predictor='gpu_predictor', random_state=1111)

rmse_cv(model_xgb)
model_xgboost1 = make_pipeline(StandardScaler(), XGBRegressor())

rmse_cv(model_xgboost1)
model_xgb
model_xgb_tuned = XGBRegressor(tree_method='gpu_hist',n_jobs=-1,booster='gbtree',predictor='gpu_predictor',

                               reg_lambda=4.31172,n_estimators=800,max_depth=2,learning_rate=0.112414 ,random_state=1111)

rmse_cv(model_xgb_tuned)
model_xgb_tuned1 = XGBRegressor(tree_method='gpu_hist',n_jobs=-1,booster='gbtree',predictor='gpu_predictor',

                               reg_lambda=17.144285714285715,n_estimators=500,max_depth=3,learning_rate=0.07827586206896552 ,random_state=1111)

rmse_cv(model_xgb_tuned1)
model_xgb_tuned1_sc = make_pipeline(StandardScaler(),XGBRegressor(tree_method='gpu_hist',n_jobs=-1,booster='gbtree',predictor='gpu_predictor',

                               reg_lambda=17.144285714285715,n_estimators=500,max_depth=3,learning_rate=0.07827586206896552 ,random_state=1111))

rmse_cv(model_xgb_tuned1_sc)
model_xgb_tuned2 = XGBRegressor(tree_method='gpu_hist',n_jobs=-1,booster='gbtree',predictor='gpu_predictor',

                               reg_lambda=16.362109350140535,n_estimators=400,max_depth=2,learning_rate=0.04200863893086535 ,random_state=1111)

rmse_cv(model_xgb_tuned2)
model_xgb_tuned2_sc = make_pipeline(StandardScaler(),XGBRegressor(tree_method='gpu_hist',n_jobs=-1,booster='gbtree',predictor='gpu_predictor',

                               reg_lambda=16.362109350140535,n_estimators=400,max_depth=2,learning_rate=0.04200863893086535 ,random_state=1111))

rmse_cv(model_xgb_tuned2_sc)
# params = {'learning_rate':np.linspace(0.01,1,30),

#           'max_depth': np.arange(2,15,1),

#           'n_estimators': np.arange(200,1200,100),

#           'reg_lambda': np.linspace(0.01,20,50)}
# mse = make_scorer(mean_squared_error)
# rsearch = RandomizedSearchCV(estimator=model_xgb, param_distributions = params, cv=5, n_iter=500, scoring='r2', n_jobs=-1,verbose=50)
# rsearch.fit(X,y)
# rsearch.best_params_
# rsearch.cv_results_.keys()
# max(rsearch.cv_results_['mean_test_score'])
# rsearch_df = pd.DataFrame(rsearch.cv_results_)

# rsearch_df.to_csv('rsearch_df.csv')
# rsearch_df[rsearch_df['mean_test_score']==max(rsearch_df['mean_test_score'])]
# for c in ['param_reg_lambda','param_n_estimators','param_max_depth','param_learning_rate','rank_test_score']:

#     plt.scatter(rsearch_df[c], rsearch_df['mean_test_score'])

#     plt.xlabel(c)

#     plt.show()
# from hyperopt import hp,fmin,tpe, Trials
# space = {'learning_rate':hp.uniform('learning_rate',0.01,0.2),

#          'max_depth':hp.quniform('max_depth',2,4,1),

#          'n_estimators':hp.quniform('n_estimators',300,1100,100),

#          'reg_lambda': hp.uniform('reg_lambda',15,17.5)}
# def objective(params):

#     params = {'learning_rate':float(params['learning_rate']),

#               'max_depth':int(params['max_depth']),

#               'n_estimators':int(params['n_estimators']),

#               'reg_lambda': float(params['reg_lambda'])}

#     model_xgb_ob = XGBRegressor(tree_method='gpu_hist',n_jobs=-1,booster='gbtree',predictor='gpu_predictor',random_state=1111)

#     neg_mse = cross_val_score(model_xgb_ob, X,y, cv=5, scoring='neg_mean_squared_error')

#     rmse=np.sqrt(-neg_mse)

#     return np.mean(rmse)
# def objective(params):

#     params = {'learning_rate':float(params['learning_rate']),

#               'max_depth':int(params['max_depth']),

#               'n_estimators':int(params['n_estimators']),

#               'reg_lambda': float(params['reg_lambda'])}

#     model_xgb_ob = XGBRegressor(tree_method='gpu_hist',n_jobs=-1,booster='gbtree',predictor='gpu_predictor',random_state=1111)

#     best_score = cross_val_score(model_xgb_ob, X,y, cv=5, scoring='r2')

#     best_score=np.mean(best_score)

#     loss=1-best_score

#     return loss
# trls=Trials()
# result = fmin(fn=objective,space=space,max_evals=2000,rstate=np.random.RandomState(1111), algo=tpe.suggest,trials=trls, verbose=50)
# trls.best_trial['result']['loss']
model_xgb_tuned2.fit(X,y)

y_pred_test = model_xgb_tuned2.predict(X_test)

y_pred_test = logy.inverse_transform(y_pred_test.reshape(-1,1))

y_pred_test = y_pred_test.reshape(-1)
output = pd.DataFrame({'Id': test['Id'],

                       'SalePrice': y_pred_test})

output.to_csv('submission.csv', index=False)
# rfe = RFE(estimator=model_rf1, n_features_to_select=50,step=5, verbose=1)

# rfe.fit(X,y)
# col = list(X.columns[rfe.support_])
# pca = PCA()

# X_pca = pca.fit_transform(X)
# X_pca.shape
# lcv = Lasso()

# lcv.fit(X,y)

# mask_lcv = lcv.coef_!=0
# rf = RandomForestRegressor()

# rf.fit(X,y)

# mask_rf = rf.feature_importances_ > (sorted(rf.feature_importances_, reverse=True)[100])
# gb = GradientBoostingRegressor()

# gb.fit(X,y)

# mask_gb = gb.feature_importances_ > (sorted(gb.feature_importances_,reverse=True)[100])
# votes = np.sum([mask_lcv,mask_rf,mask_gb],axis=0)

# votes = [True if i>0 else False for i in votes]
# col_imp = X.columns[votes].tolist()

# print(col_imp)
# from tpot import TPOTRegressor

# tpot = TPOTRegressor(generations=3,population_size=5,verbosity=2,offspring_size=10,scoring='neg_mean_squared_error',cv=2)

# tpot.fit(X,y)

# tpot.score(X,y)
# model_llcv = LassoLarsCV(normalize=False)

# rmse_cv(model_llcv)