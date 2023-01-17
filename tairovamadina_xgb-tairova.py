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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew
%matplotlib inline
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df.head()
df.describe()
df.info()
print ('Train shape', df.shape)
print ('Test Shape', df_test.shape)
df.isnull().sum(axis=0).sort_values().tail(15)
df_test.isnull().sum(axis=0).sort_values().tail(15)
df_test.head()
feast_counts=df.nunique(dropna=False)
feast_counts.sort_values()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew
sns.regplot(x="OverallQual",y="SalePrice",data=df)
sns.set(style="darkgrid")
sns.scatterplot(x="GrLivArea",y="SalePrice",data=df)

df=df.drop(df[(df['GrLivArea']>4000)&(df['SalePrice']<300000)].index)

sns.set(style="darkgrid")

sns.scatterplot(x="GrLivArea",y="SalePrice",data=df)

sns.set(style="darkgrid")
sns.scatterplot(x="YearBuilt",y="SalePrice",data=df)
sns.distplot(df['SalePrice'],fit=norm)

(mu,sigma)=norm.fit(df['SalePrice'])
stats.probplot(df['SalePrice'],plot=plt)
df['SalePrice']=np.log1p(df['SalePrice'])
stats.probplot(df['SalePrice'],plot=plt)
sns.distplot(df['SalePrice'],fit=norm)

df.columns
sns.scatterplot(y="SalePrice",x="LotArea",data=df)
train_ID=df['Id']
test_ID=df_test['Id']
df.drop("Id",axis=1,inplace=True)
df_test.drop("Id",axis=1,inplace=True)
n_train=df.shape[0]
n_test=df_test.shape[0]
y_train=df.SalePrice.values
all_data=pd.concat((df,df_test)).reset_index(drop=True)
all_data.drop(['SalePrice'],axis=1,inplace=True)
all_data.columns
all_data_na=all_data.isnull().sum()/len(all_data)*100
all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)
all_data_na
all_data_na.index
plt.subplots(figsize=(15,12))
plt.xticks(rotation=90)
sns.barplot(x=all_data_na.index,y=all_data_na)
corrmat=df.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,square=False,vmax=0.9)
all_data['PoolQC']=all_data['PoolQC'].fillna("None")
all_data['MiscFeature']=all_data['MiscFeature'].fillna("None")
all_data['Alley']=all_data['Alley'].fillna("None")
all_data['Fence']=all_data['Fence'].fillna("None")
all_data['FireplaceQu']=all_data['FireplaceQu'].fillna("None")
all_data['LotFrontage']=all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
for cols in ('GarageCond','GarageFinish', 'GarageQual', 'GarageType'):
    all_data[cols]=all_data[cols].fillna('None')
for cols in('GarageArea', 'GarageCars','GarageYrBlt'):
    all_data[cols]=all_data[cols].fillna(0)
for cols in ('BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1','BsmtExposure'):
    all_data[cols]=all_data[cols].fillna('None')
for cols in ('BsmtFinSF1', 'BsmtFinSF2','BsmtFullBath', 'BsmtHalfBath','BsmtUnfSF','TotalBsmtSF'):
    all_data[cols]=all_data[cols].fillna(0)
all_data['MasVnrType']=all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)
all_data['MSZoning'].value_counts()
all_data['MSZoning']=all_data['MSZoning'].fillna('RL')
all_data['Utilities'].value_counts()
all_data['Functional']=all_data['Functional'].fillna('Typ')
all_data['Electrical']=all_data['Electrical'].fillna('SBrkr')
all_data['Exterior1st']=all_data['Exterior1st'].fillna('VinylSd')
all_data['Exterior2nd']=all_data['Exterior2nd'].fillna('VinylSd')
all_data['KitchenQual']=all_data['KitchenQual'].fillna('TA')
all_data['SaleType']=all_data['SaleType'].fillna('WD')
all_data_na=all_data.isnull().sum()/len(all_data)*100
all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)
all_data_na
all_data.info()
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
all_data.shape
all_data['TotalSF']=all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']
all_data.info()
numeric_feats=all_data.dtypes[all_data.dtypes!='object'].index
numeric_feats
skewed_feats=all_data[numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
skewness=pd.DataFrame({'Skew':skewed_feats})
print(skewness.shape)
skewness
skewness=skewness[abs(skewness)>0.75]
print(skewness.shape)

from scipy.special import boxcox1p
skewed_features=skewness.index
lam=0.15
for feat in skewed_features:
    all_data[feat]=boxcox1p(all_data[feat],lam)
all_data.info()
all_data=pd.get_dummies(all_data)
all_data.info()
all_data.info()
train=all_data[:n_train]
test=all_data[n_train:]
train.shape

test.shape
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
n_folds=5
def rmsle_cv(model):
    kf=KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse=np.sqrt(-cross_val_score(model,train.values,y_train,scoring="neg_mean_squared_error",cv=kf))
    return (rmse)
lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=1))
Enet=make_pipeline(RobustScaler(),ElasticNet(alpha=0.005,l1_ratio=0.9,random_state=3))
GBoost=GradientBoostingRegressor(loss='huber',n_estimators=3000,learning_rate=0.05,max_depth=4,max_features='sqrt',min_samples_leaf=15,
                                min_samples_split=10,random_state=5)
KRR=KernelRidge(alpha=0.6,kernel='polynomial',degree=2,coef0=2.5)

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
score=rmsle_cv(lasso)
print(score.mean())
print(score.std())
score=rmsle_cv(Enet)
print(score.mean())
print(score.std())
score=rmsle_cv(KRR)
print(score.mean())
print(score.std())
score=rmsle_cv(GBoost)
print(score.mean())
print(score.std())
score=rmsle_cv(model_lgb)
print(score.mean())
print(score.std())
class StackingAverageModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,base_models,meta_model,n_folds=5):
        self.base_models=base_models
        self.meta_model=meta_model
        self.n_folds=n_folds
    
    def fit(self,X,y):
        self.base_models_=[list() for x in self.base_models]
        self.meta_model_=clone(self.meta_model)
        kfold=KFold(n_splits=self.n_folds,shuffle=True,random_state=156)
        
        out_of_fold_pred=np.zeros((X.shape[0],len(self.base_models)))
        for i,model in enumerate(self.base_models):
            for train_index,holdout_index in kfold.split(X,y):
                instance=clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index],y[train_index])
                out_of_fold_pred[holdout_index,i]=instance.predict(X[holdout_index])
        
        self.meta_model_.fit(out_of_fold_pred,y)
        return self
    
    def predict(self,X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAverageModels(base_models = (Enet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))
stacked_averaged_models.fit(train.values,y_train)
stacked_train_pred=stacked_averaged_models.predict(train.values)
stacked_predict=np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train,stacked_train_pred))
model_lgb.fit(train.values,y_train)
lgb_train_pred=model_lgb.predict(train.values)
lgb_predict=np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train,lgb_train_pred))
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_predict*0.70 + xgb_pred*0.15 + lgb_predict*0.15
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('XGB.csv',index=False)
