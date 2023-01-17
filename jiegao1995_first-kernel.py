# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from datetime import datetime 

# import matplotlab.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    print(filenames)

    filenames

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

data_test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

data_co=[data_train,data_test]

data_train2=data_train
data_test.shape
data_train.info()

t1=datetime.now()

print('start data processing',datetime.now())
data_train.MSSubClass
data_train.describe()
qualitative = [f for f in data_train.columns

    if data_train[f].dtypes==object]



quantitative =[f for f in data_train.columns

    if data_train[f].dtypes!=object]

quantitative.remove('SalePrice')

quantitative.remove('Id')

quan=quantitative

qual=qualitative
print("quantitative:" +str(len(quantitative))+" "

     "qualitative:" + str(len(qualitative)))

y,ax=plt.subplots(figsize=(12,8))

corrmat=data_train.corr()

sns.heatmap(corrmat)
k  = 10 # 关系矩阵中将显示10个特征

corrmat = data_train.corr()

fig,ax=plt.subplots(figsize=(14,7))

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(data_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
features_int=cols

features_int
idx_qual=data_train2[qual].columns[data_train2[qual].isnull().any()==True]
idx_qual
f,ax=plt.subplots(figsize=(12,6))

data_isnull=data_train[qual].isnull().sum()



data_isnull[data_isnull>0].plot(kind='bar')

# sns.barplot(data_isnull)
type(data_isnull)
data_train['MasVnrType'][data_train['MasVnrType'].isnull()]
data_train['MasVnrType'].describe()
data_train['Electrical']=data_train['Electrical'].fillna('SBrkr')

data_test['Electrical']=data_test['Electrical'].fillna('SBrkr')

for data in data_co:

    for i in ['BsmtQual','BsmtCond','MasVnrType','GarageType','MiscFeature','GarageQual','GarageCond','BsmtExposure','GarageFinish','BsmtFinType2','BsmtFinType1']:

        data[i]=data[i].fillna('None')
data_train.columns[data_train.isnull().any()]
idx_qual
for data in data_co:

    a=data[qual]

    c=[b for b in data[qual].columns if (data[b].isnull().sum()<600 and data[b].isnull().sum()>0)]

    d=[b for b in data[qual].columns if data[b].isnull().sum()>600]

    for x in d:

        qual.remove(x)

for data in data_co:

    data=data.drop(["Alley",  "FireplaceQu",  "PoolQC","Fence"],axis=1)
data_test.shape
data_train[qual].info()
var=qual

data_tem=pd.DataFrame()

data_tem1=pd.DataFrame()

for i in qual:

    data_tem=pd.concat([data_tem,pd.get_dummies(data_train[i],prefix=i)],axis=1)

    data_tem1=pd.concat([data_tem1,pd.get_dummies(data_test[i],prefix=i)],axis=1)

data_tem
data_tem1.shape
data_tem.shape
for i in data_tem.columns:

    if i not in data_tem1.columns:

        data_tem.drop(i,axis=1,inplace=True)
data_tem1.shape
# X_train.shape

for i in data_co:

    i.fillna(i.mean(),inplace=True)


df_train=pd.concat([data_train[quan],data_tem],axis=1)

df_test=pd.concat([data_test[quan],data_tem1],axis=1)
X=df_train

y_train=data_train.SalePrice

X.isnull().any()
df_test.shape
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge,LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler,StandardScaler

import xgboost as xgb

from sklearn import linear_model

from sklearn import svm, gaussian_process

import warnings

import lightgbm as lgb

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold,cross_val_score,train_test_split,learning_curve
scaler=StandardScaler()

X_scale=scaler.fit_transform(X)

y_scale=y_train

X_train,X_test,y_train,y_test=train_test_split(X_scale,y_scale,test_size=0.3,random_state=42)



clfs={

    'svm':svm.SVR(),

    'RandomForestRegressor':RandomForestRegressor(n_estimators=400),

    'BayesianRidge':linear_model.BayesianRidge(),

    'Lasso':linear_model.Lasso(alpha =0.0005, random_state=1),

    'xgb': xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

}

for clf in clfs:

    clfs[clf].fit(X_train,y_train)

    y_pred=clfs[clf].predict(X_test)

    print(clf+'cost:'+str(np.sum(y_pred-y_test)/len(y_pred)))
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 

                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):

    train_sizes, train_scores, test_scores = learning_curve(estimator, 

    X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    if plot:

        plt.figure()

        plt.title(title)

        if ylim is not None:

            plt.ylim(*ylim)

        plt.xlabel(u"training data")

        plt.ylabel(u"score")

        plt.grid()

    

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 

                         alpha=0.1, color="b")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 

                         alpha=0.1, color="r")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="training score")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="test score")

    

        plt.legend(loc="best")

        plt.draw()

        plt.show()



        # plt.pause(0.001) 

        plt.gca().invert_yaxis()

       

    

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2

    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff



clf_xgb=clfs['xgb']

plot_learning_curve(clf_xgb,'learning curve',X_scale,y_scale)
# np.isnan(X_train_scale).any()

clf=clfs['Lasso']

rfr=clf

X_test_scale=scaler.fit_transform(df_test)

predictions=rfr.predict(X_test_scale)

sub = pd.DataFrame()

sub['Id'] = data_test['Id']

sub['SalePrice'] = predictions

sub.to_csv('submission1.csv',index=False)
print('end data processing',datetime.now())

t2=datetime.now()

print('running time:',t2-t1)