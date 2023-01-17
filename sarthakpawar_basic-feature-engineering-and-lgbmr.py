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
dataset=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
correlation=dataset.corr()

for row in correlation:

    print('for row',row)

    print('-------------------')

    print(correlation[row][correlation[row]>=0.7])

    print(correlation[row][correlation[row]<=-0.7])

    print('-------------------')
dataset['GarageYrBlt'].isna().sum()
dataset['TotRmsAbvGrd'].isna().sum()
dataset['GarageCars'].isna().sum()
del dataset['GarageYrBlt']

# del dataset['TotRmsAbvGrd']

# del dataset['GarageCars']

del test['GarageYrBlt']

# del test['TotRmsAbvGrd']

# del test['GarageCars']
import scipy.stats
class column_similarity(object):

    def __init__(self, data , filter_col):

        num_columns=[]

        cat_columns=[]

        self.data=data.fillna('')

        for col in data.columns:

            if self._check_numberic(data[col]):

                num_columns.append(col)

            else:

                cat_columns.append(col)

        self.cat_columns=np.array(cat_columns)

        self.num_columns=np.array(num_columns)

        self.cat_dist=None

        self.num_dist=None

        self.dist_reset()



    def dist_reset(self):

        if len(self.cat_columns)>0:

            self.cat_dist=np.zeros((len(self.cat_columns),len(self.cat_columns)))

        if len(self.num_columns)>0:

            self.num_dist=np.zeros((len(self.num_columns),len(self.num_columns)))

        

    def _check_numberic(self,data):

        return pd.to_numeric(data[data.notna()], errors='coerce').notnull().all()

    

    def filtered_similarity(self,delimiter,threshold_chi,threshold_cor, filt=None):

        self.dist_reset()

        Cat_dist_frame=pd.DataFrame()

        Num_dist_frame=pd.DataFrame()

        chi_column1=[]

        chi_column2=[]

        chi_usecase1=[]

        chi_usecase2=[]

        chi_score=[]

        if self.cat_columns is not None:

            for i in range(len(self.cat_columns)):

                for j in range(i,len(self.cat_columns)):

                    contingency_table=None

                    contingency_table=pd.crosstab(self.data[self.cat_columns[i]],self.data[self.cat_columns[j]])

                    chi2, p, dof, ex=scipy.stats.chi2_contingency(contingency_table)

                    self.cat_dist[i][j]=chi2

                    self.cat_dist[j][i]=chi2 

                    

                if self.cat_dist[i][i]>0:

                    self.cat_dist[i]= self.cat_dist[i]/self.cat_dist[i][i]

                selected_elems=list(np.where(self.cat_dist[i]>=threshold_chi)[0])

                if len(selected_elems)>0:

                    selected_elems.remove(i)

                    chi_column1=chi_column1+[self.cat_columns[i]]*len(selected_elems)

                    chi_column2=chi_column2+list(self.cat_columns[selected_elems])

                    chi_score=chi_score+list(self.cat_dist[i][selected_elems])

            Cat_dist_frame['Column1']=[col.split(delimiter)[1] if delimiter in col  else col for col in chi_column1]

            Cat_dist_frame['Column2']=[col.split(delimiter)[1] if delimiter in col  else col for col in chi_column2]

            Cat_dist_frame['Score']=chi_score 

        return Cat_dist_frame.copy()

    

    def similarity(self,delimiter='_',threshold_chi=0.0,threshold_cor=0.0):

        global_Cat_dist_frame=self.filtered_similarity(delimiter,threshold_chi,threshold_cor)

        return global_Cat_dist_frame
sm=column_similarity(dataset,dataset.columns)
similarity=sm.similarity()
similarity
# similarity.loc[similarity['Score']>0.3,:]
# del dataset['Exterior2nd']

# del test['Exterior2nd']

# del dataset['GarageFinish']

# del test['GarageFinish']

# del dataset['GarageCond']

# del test['GarageCond']
dataset['GarageCond'].value_counts()
dataset['GarageFinish']=dataset['GarageFinish'].fillna('Unf')

dataset['GarageCond']=dataset['GarageCond'].fillna('TA')

test['GarageFinish']=dataset['GarageFinish'].fillna('Unf')

test['GarageCond']=dataset['GarageCond'].fillna('TA')
dataset.columns
cols_null=dataset.isna().sum(axis=0)

cols_null=cols_null[cols_null>0]
cols_null.sort_values(ascending=False)
del dataset['PoolQC']

del test['PoolQC']

del dataset['MiscFeature']

del test['MiscFeature']

del dataset['Alley']

del test['Alley']

del dataset['Fence']

del test['Fence']
dataset[['FireplaceQu','SalePrice']].fillna('O').groupby('FireplaceQu').agg(['mean','min','max','count'])
del dataset['FireplaceQu']

del test['FireplaceQu']
dataset[['LotFrontage','SalePrice']].describe()
dataset[['LotFrontage','SalePrice']].corr()
dataset['LotFrontage']=dataset['LotFrontage'].fillna(dataset['LotFrontage'].median())

test['LotFrontage']=test['LotFrontage'].fillna(dataset['LotFrontage'].median())
cols_null=dataset.isna().sum(axis=0)

cols_null=cols_null[cols_null>0]

cols_null.sort_values(ascending=False)
dataset[['GarageQual','SalePrice']].fillna('O').groupby('GarageQual').agg(['mean','min','max','count'])
dataset['GarageQual']=dataset['GarageQual'].fillna('TA')

test['GarageQual']=test['GarageQual'].fillna('TA')
dataset['GarageType']=dataset['GarageType'].fillna('Attchd')

test['GarageType']=test['GarageType'].fillna('Attchd')
cols_null=dataset.isna().sum(axis=0)

cols_null=cols_null[cols_null>0]

cols_null.sort_values(ascending=False)
dataset[['BsmtFinType2','SalePrice']].fillna('O').groupby('BsmtFinType2').agg(['mean','median','min','max','count'])
dataset['BsmtFinType2']=dataset['BsmtFinType2'].fillna('Unf')

test['BsmtFinType2']=test['BsmtFinType2'].fillna('Unf')
dataset[['BsmtExposure','SalePrice']].fillna('No').groupby('BsmtExposure').agg(['mean','median','min','max','count'])
dataset['BsmtExposure']=dataset['BsmtExposure'].fillna('No')

test['BsmtExposure']=test['BsmtExposure'].fillna('No')
cols_null=dataset.isna().sum(axis=0)

cols_null=cols_null[cols_null>0]

cols_null.sort_values(ascending=False)
dataset[['BsmtFinType1','SalePrice']].fillna('O').groupby('BsmtFinType1').agg(['mean','median','min','max','count'])
dataset['BsmtFinType1']=dataset['BsmtFinType1'].fillna('Unf')

test['BsmtFinType1']=test['BsmtFinType1'].fillna('Unf')
cols_null=dataset.isna().sum(axis=0)

cols_null=cols_null[cols_null>0]

cols_null.sort_values(ascending=False)
dataset[['BsmtCond','SalePrice']].fillna('O').groupby('BsmtCond').agg(['mean','median','min','max','count'])
dataset['BsmtCond']=dataset['BsmtCond'].fillna('TA')

test['BsmtCond']=test['BsmtCond'].fillna('TA')
cols_null=dataset.isna().sum(axis=0)

cols_null=cols_null[cols_null>0]

cols_null.sort_values(ascending=False)
dataset[['BsmtQual','SalePrice']].fillna('O').groupby('BsmtQual').agg(['mean','median','min','max','count'])
dataset['BsmtQual']=dataset['BsmtQual'].fillna('Gd')

test['BsmtQual']=test['BsmtQual'].fillna('Gd')
cols_null=dataset.isna().sum(axis=0)

cols_null=cols_null[cols_null>0]

cols_null.sort_values(ascending=False)
dataset[['MasVnrArea','SalePrice']].fillna(0).corr()
test['MasVnrArea']=test['MasVnrArea'].fillna(test['MasVnrArea'].median())

dataset['MasVnrArea']=dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].median())
dataset['MasVnrType'].value_counts()
dataset['MasVnrType']=dataset['MasVnrType'].fillna('None')

test['MasVnrType']=test['MasVnrType'].fillna('None')
dataset[['Electrical','SalePrice']].groupby('Electrical').agg(['mean','median','min','max','count'])
dataset['Electrical']=dataset['Electrical'].fillna('SBrkr')

test['Electrical']=test['Electrical'].fillna('SBrkr')
import numpy as np

import scipy.stats as stats

import pylab as pl



def plot_normal_dist(h):

    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

    pl.plot(h,fit,'-o')

#     pl.hist(h,normed=True)      #use this to draw histogram of your data

    pl.show()
plot_normal_dist(dataset['LotFrontage'].to_list())
for col in dataset.describe():

    print(col)

    plot_normal_dist(dataset[col])
dataset=dataset.loc[dataset['LotFrontage']<300,:]
dataset=dataset.reset_index(drop=True)
for col in dataset.describe():

    print(col)

    plot_normal_dist(dataset[col])
dataset['LotArea'].quantile(0.994)
(dataset['LotArea']>dataset['LotArea'].quantile(0.995)).sum()
dataset=dataset.loc[dataset['LotArea']<dataset['LotArea'].quantile(0.995),:]

dataset=dataset.reset_index(drop=True)
for col in dataset.describe():

    print(col)

    plot_normal_dist(dataset[col])
dataset['MasVnrArea'].quantile(0.999)
(dataset['MasVnrArea']>dataset['MasVnrArea'].quantile(0.999)).sum()
dataset=dataset.loc[dataset['MasVnrArea']<=dataset['MasVnrArea'].quantile(0.999),:]

dataset=dataset.reset_index(drop=True)
for col in dataset.describe():

    print(col)

    plot_normal_dist(dataset[col])
dataset['BsmtFinSF2'].quantile(0.9991).sum()
dataset=dataset.loc[dataset['BsmtFinSF2']<=dataset['BsmtFinSF2'].quantile(0.9991),:]

dataset=dataset.reset_index(drop=True)
for col in dataset.describe():

    print(col)

    plot_normal_dist(dataset[col])
dataset['EnclosedPorch'].quantile(0.997)
(dataset['EnclosedPorch']>dataset['EnclosedPorch'].quantile(0.997)).sum()
dataset=dataset.loc[dataset['EnclosedPorch']<=dataset['EnclosedPorch'].quantile(0.997),:]

dataset=dataset.reset_index(drop=True)
for col in dataset.describe():

    print(col)

    plot_normal_dist(dataset[col])
cols_null=test.isna().sum(axis=0)

cols_null=cols_null[cols_null>0]

cols_null.sort_values(ascending=False)
from sklearn.preprocessing import StandardScaler
dataset.columns
def _check_numberic(data):

    return pd.to_numeric(data[data.notna()], errors='coerce').notnull().all()
from sklearn.preprocessing import LabelEncoder
test.isna().sum()
newdata=pd.concat([dataset,test],axis=0)
newdata.SalePrice.isna().sum()
newdata.shape
# le=LabelEncoder()

# for col in dataset.columns:

#     if  not _check_numberic(dataset[col]):

#         le.fit(dataset[col])

#         dataset[col]=le.transform(dataset[col])

#         count_values=test[col].value_counts()

#         test[col]=le.transform(test[col].fillna(count_values.idxmax()))
newdata=newdata.reset_index(drop=True)
newdata.columns
del newdata['3SsnPorch']

del newdata['Condition2']

del newdata['GarageCars']

del newdata['LowQualFinSF']

del newdata['MiscVal']

del newdata['RoofMatl']

del newdata['Street']

del newdata['Utilities']
newdata['HalfBath']=[1 if val>0 else 0 for val in newdata['HalfBath']]

newdata['PoolArea']=[1 if val>0 else 0 for val in newdata['PoolArea']]
newdata.loc[newdata.SalePrice.notna(),['1stFlrSF','2ndFlrSF','BsmtFinSF1','BedroomAbvGr','GarageArea','BsmtUnfSF','OpenPorchSF','EnclosedPorch','MasVnrArea','LotFrontage','GrLivArea','LotArea','SalePrice']].corr()
nacols=newdata.isna().sum()
nacols[nacols>0]
newdata['BsmtFinSF1']=newdata['BsmtFinSF1'].fillna(newdata['BsmtFinSF1'].median())

newdata['BsmtFinSF2']=newdata['BsmtFinSF2'].fillna(newdata['BsmtFinSF2'].median())

newdata['BsmtFullBath']=newdata['BsmtFullBath'].fillna(newdata['BsmtFullBath'].median())

newdata['BsmtHalfBath']=newdata['BsmtHalfBath'].fillna(newdata['BsmtHalfBath'].median())

newdata['BsmtUnfSF']=newdata['BsmtUnfSF'].fillna(newdata['BsmtUnfSF'].median())

newdata['GarageArea']=newdata['GarageArea'].fillna(newdata['GarageArea'].median())

newdata['TotalBsmtSF']=newdata['TotalBsmtSF'].fillna(newdata['TotalBsmtSF'].median())
newdata['MSZoning']
col='Exterior1st'

newdata[col]=newdata[col].fillna(newdata[col].value_counts().idxmax())

col='Exterior2nd'

newdata[col]=newdata[col].fillna(newdata[col].value_counts().idxmax())

col='Functional'

newdata[col]=newdata[col].fillna(newdata[col].value_counts().idxmax())

col='KitchenQual'

newdata[col]=newdata[col].fillna(newdata[col].value_counts().idxmax())

col='MSZoning'

newdata[col]=newdata[col].fillna(newdata[col].value_counts().idxmax())

col='SaleType'

newdata[col]=newdata[col].fillna(newdata[col].value_counts().idxmax())
na=newdata.isna().sum()
na[na>0]
le=LabelEncoder()

for col in newdata.columns:

    if  not _check_numberic(newdata[col]):

        le.fit(newdata[col])

        newdata[col]=le.transform(newdata[col])
test_data=newdata.loc[newdata.SalePrice.isna(),:]
train_data=newdata.loc[newdata.SalePrice.notna(),:]
train_data.shape
test_data.shape
import math

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
from sklearn.model_selection import train_test_split
features=list(train_data.columns)
features.remove('Id')

features.remove('SalePrice')
train=train_data[features].values

train_y=train_data['SalePrice']
sc= RobustScaler()

sc.fit(train)

train=sc.transform(train)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,train_y,test_size=0.2)
from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 120, random_state = 42)# Train the model on training data
rf.fit(x_train, y_train)
predictions=rf.predict(x_test)

rmsle(y_test.to_list(),predictions)
from sklearn.metrics import mean_squared_error
import xgboost as xgb
model=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.4603, gamma=0.0468,

             importance_type='split', learning_rate=0.05, max_delta_step=0,

             max_depth=20, min_child_weight=1.7817, missing=None,

             n_estimators=1024, n_jobs=1, nthread=-1, objective='reg:squarederror',

             random_state=7, reg_alpha=0.464, reg_lambda=0.8571,

             scale_pos_weight=1, seed=None, silent=1, subsample=0.5213,

             verbosity=1)
model.fit(x_train,y_train)
predictions=model.predict(x_test)

rmsle(y_test.to_list(),predictions)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)

    rmse= np.sqrt(-cross_val_score(model, train, train_y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
score=rmsle_cv(model_lgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb.fit(train,train_y)
# from sklearn.linear_model import ElasticNet

# from sklearn.model_selection import GridSearchCV

# kfolds=KFold(n_splits=10,shuffle=True,random_state=42)

# model=ElasticNet(max_iter=1000)

# ela_param_grid = {"alpha":[0.0006951927961775605],

#                  "l1_ratio":[0.90]}

# grid_search= GridSearchCV(model_lgb,param_grid=ela_param_grid,cv=kfolds,scoring="neg_mean_squared_error",n_jobs= 10, verbose = 1)

# grid_search.fit(train,train_y)

# ela_best=grid_search.best_estimator_

# np.sqrt(-grid_search.best_score_)
model.fit(x_train,y_train)

predictions=model.predict(x_test)

rmsle(y_test.to_list(),predictions)
model_lgb=lgb.LGBMRegressor(bagging_fraction=0.8, bagging_freq=5, bagging_seed=9,

              boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

              feature_fraction=0.8, feature_fraction_seed=9,

              importance_type='gain', learning_rate=0.02, max_bin=55,

              max_depth=-1, min_child_samples=30, min_child_weight=0.01,

              min_data_in_leaf=6, min_split_gain=0.0,

              min_sum_hessian_in_leaf=11, n_estimators=1024, n_jobs=-1,

              num_leaves=20, objective='regression', random_state=None,

              reg_alpha=0.01, reg_lambda=0.01, silent=True, subsample=1.0,

              subsample_for_bin=200000, subsample_freq=0)
model_lgb.fit(x_train,y_train)

predictions=model_lgb.predict(x_test)

rmsle(y_test.to_list(),predictions)
test_sc=sc.transform(test_data[features].values)
test_sc.shape
train.shape
model_lgb.fit(train,train_y)

predicted_values=model_lgb.predict(test_sc)
model.fit(train,train_y)

predicted_values2=model.predict(test_sc)
output=pd.DataFrame()

output['Id']=test['Id']

output['SalePrice']=predicted_values
output.head()
output.to_csv('sample_submission.csv',index=False)