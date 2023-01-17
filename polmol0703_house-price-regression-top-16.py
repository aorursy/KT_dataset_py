import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(train.shape))

print("The test data size before dropping Id feature is : {} ".format(test.shape))



#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))
sns.distplot(train.SalePrice)
train.SalePrice.describe()
train.SalePrice.skew()
train.SalePrice.isna().sum()
train.SalePrice.isnull().sum()
ntrain = train.shape[0]

ntest = test.shape[0]

#y_train = train.SalePrice

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
def check_nas():    

    sample_values = pd.DataFrame(index=all_data.columns,columns=['SampleValue'])

    for i in all_data.columns:

        sample_values.loc[i].SampleValue = all_data[i].value_counts().index[1]

    nas = pd.DataFrame(all_data.isnull().sum(),columns=['SumOfNA'])

    types = pd.DataFrame(all_data.dtypes,columns=['Type'])

    sample_values.sort_index(inplace=True)

    nas.sort_index(inplace=True)

    types.sort_index(inplace=True)

    alls=pd.concat([sample_values,nas,types],axis=1)

    return(alls[alls.SumOfNA>0].sort_values('SumOfNA',ascending=False))
check_nas()
# Most of the NAs are probably because the property does not contain the specific thing (eg. No Pool)

none_feats = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',

              'GarageFinish','GarageQual','GarageType','GarageCond',

              'BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtQual',

              'BsmtCond','MasVnrType','MSZoning']

zero_feats = ['LotFrontage','GarageYrBlt','MasVnrArea']
for i in none_feats:

    all_data[i].fillna('None',inplace=True)
for i in zero_feats:

    all_data[i].fillna(0,inplace=True)
all_data.drop(['MasVnrArea','MasVnrType','Electrical'],axis=1,inplace=True)
check_nas()
all_data['BsmtFullBath'].fillna(0,inplace=True)

all_data['BsmtHalfBath'].fillna(0,inplace=True)

all_data['Functional'].fillna('Typ',inplace=True)

all_data['Utilities'].fillna('AllPub',inplace=True)

all_data['BsmtFinSF1'].fillna(0,inplace=True)

all_data['BsmtFinSF2'].fillna(0,inplace=True)

all_data['BsmtUnfSF'].fillna(0,inplace=True)

all_data['Exterior1st'].fillna('VinylSd',inplace=True)

all_data['Exterior2nd'].fillna('VinylSd',inplace=True)

all_data['GarageArea'].fillna(0,inplace=True)

all_data['GarageCars'].fillna(0,inplace=True)

all_data['KitchenQual'].fillna('None',inplace=True)

all_data['SaleType'].fillna('WD',inplace=True)

all_data['TotalBsmtSF'].fillna(0,inplace=True)

check_nas()
fig = plt.figure(figsize=(12,9))

sns.heatmap(train.corr())
imp_feat=abs(train.corr()['SalePrice']).sort_values(ascending=False).head(11).index
fig = plt.figure(figsize=(10,5))

sns.heatmap(train[imp_feat].corr(),annot=True)
imp_feat_2 = imp_feat.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'])
# sub 5

all_data.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'],axis=1,inplace=True)
imp_feat_2
sns.pairplot(train[imp_feat_2])
fig = plt.figure(figsize=(10,5))

plt.scatter(x=train.GrLivArea,y=train.SalePrice)
x=abs(train[imp_feat_2].skew()).sort_values(ascending=False)

x
def check_skew(column,with_log):

    feat = train[column]

    if (with_log==False):

        fig,ax = plt.subplots(figsize = (5,3))

        ax = sns.distplot(feat,fit=norm)

        fig,ax = plt.subplots(figsize = (5,3))

        ax = stats.probplot(feat, plot=plt)

        (mu, sigma) = norm.fit(feat)

        print( 'The normal dist fit has the following parameters: \n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    elif (with_log==True):

        feat = np.log1p(feat)

        fig,ax = plt.subplots(figsize = (5,3))

        ax = sns.distplot(feat,fit=norm)

        fig,ax = plt.subplots(figsize = (5,3))

        ax = stats.probplot(feat, plot=plt)

        (mu, sigma) = norm.fit(feat)

        print( 'The normal dist fit has the following parameters: \n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    
check_skew(column='SalePrice',with_log=False)

check_skew(column='SalePrice',with_log=True)
train['SalePrice'] = np.log1p(train['SalePrice'])
check_skew(column='TotalBsmtSF',with_log=False)

check_skew(column='TotalBsmtSF',with_log=True)
all_data['TotalBsmtSF'][all_data['TotalBsmtSF']!=0] =  np.log1p(all_data['TotalBsmtSF'][all_data['TotalBsmtSF']!=0])
fig,ax = plt.subplots(figsize = (5,3))

ax = sns.distplot(train['TotalBsmtSF'][train['TotalBsmtSF']!=0],fit=norm)

fig,ax = plt.subplots(figsize = (5,3))

ax = stats.probplot(train['TotalBsmtSF'][train['TotalBsmtSF']!=0], plot=plt)

(mu, sigma) = norm.fit(train['TotalBsmtSF'][train['TotalBsmtSF']!=0])

print( 'The normal dist fit has the following parameters: \n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
x=abs(train[imp_feat_2].skew()).sort_values(ascending=False)

x
# for sub 5

#num_nonzero = list(set(all_data.dtypes[all_data.dtypes != "object"].index) & set([i for i in all_data.columns.values if sum(all_data[i]==0)==0]))

#skewed = abs(all_data[num_nonzero].skew()).sort_values(ascending=False)

#skewed_feats = skewed[skewed > 0.75].index

#all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
# for sub 4

#numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

#numeric_feats

#skewed_feats=abs(all_data[numeric_feats].skew()).sort_values(ascending=False)

#skewed_feats = skewed_feats[skewed_feats > 0.75].index

#all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data.shape
# sub 6

#from sklearn.decomposition import PCA

#pca = PCA(n_components=2)

#pca.fit(all_data)

#x_pca = pca.transform(all_data)

#x_pca = pd.DataFrame(x_pca, columns=['pca1','pca2'])
#all_data = pd.concat([all_data,x_pca],axis=1)
y_train = train.SalePrice

train = all_data[:ntrain]

test = all_data[ntrain:]
train[train.GrLivArea >= 4600].index
# drop outliers from train and y_train

# for sub 3

y_train.drop(train[train.GrLivArea >= 4600].index,inplace=True)

train.drop(train[train.GrLivArea >= 4600].index,inplace=True)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error

import xgboost as xgb
gb = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5))

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.5, random_state=1,max_iter=5000))

model_xgb = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
param_grid = {'learning_rate':[0.04,0.05,0.06],'max_depth':[2,4,6]}

search_gb = GridSearchCV(GradientBoostingRegressor(n_estimators=3000,loss='huber', 

                                                   min_samples_leaf=15,

                                                   min_samples_split=10, 

                                                   random_state =5),

                       param_grid = param_grid

                       ,cv=3)

gb_pipe_search = make_pipeline(RobustScaler(),search_gb)

search_gb.fit(X,y)

search_gb.best_params_
param_grid = {'learning_rate':[0.04,0.05,0.06],'max_depth':[2,4,6]

             }

search_xgb =GridSearchCV(xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1),param_grid=param_grid,cv=3)

xgb_pipe_search = make_pipeline(RobustScaler(),search_xgb)

search_xgb.fit(X,y)

search_xgb.best_params_
param_grid = {'alpha':np.linspace(0.0001,0.01,100),'l1_ratio':[0.3,0.5,0.9]}

search_enet = GridSearchCV(ElasticNet(random_state=3),param_grid=param_grid,cv=3)

enet_pipe_search = make_pipeline(RobustScaler(),search_enet)

search_enet.fit(X,y)

search_enet.best_params_
gb_opt = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=3000, learning_rate=0.04,

                                   max_depth=2, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5))

model_xgb_opt = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=2, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1))

ENet_opt = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
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
avg_opt = AveragingModels(models = [gb_opt, model_xgb_opt, ENet_opt])

avg.fit(X,y)
#sub = pd.DataFrame()

#sub['Id'] = test_ID

#sub['SalePrice'] = np.expm1(avg_opt.predict(test))

#sub.to_csv('submission8.csv',index=False)