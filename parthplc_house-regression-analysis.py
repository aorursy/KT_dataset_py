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
#Importing python libraries

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

import sklearn

import scipy

import numpy

import json

import sys

import csv

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

import os

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn import linear_model
from scipy.stats import skew, norm, probplot, boxcox

# import train and test to play with it

train_df = pd.read_csv('../input/train.csv')

train_df_dup = train_df.copy()

test_df = pd.read_csv('../input/test.csv')

labels_df = train_df.pop('SalePrice') 



pd.set_option('display.max_rows',100)# amount of rows that can be seen at a time



labels_df.describe()
train_df.shape,test_df.shape
data = pd.concat([train_df, test_df], keys=['train_df', 'test_df'])

print(data.columns) # check column decorations

print('rows:', data.shape[0], ', columns:', data.shape[1]) # count rows of total dataset

print('rows in train dataset:', train_df.shape[0])

print('rows in test dataset:', test_df.shape[0])
data.info()
nans = pd.concat([train_df.isnull().sum(), train_df.isnull().sum() / train_df.shape[0], 

                  test_df.isnull().sum(), test_df.isnull().sum() / test_df.shape[0]], axis=1, 

                 keys=['Train', 'Percentage', 'Test', 'Percentage'])

print(nans[nans.sum(axis=1) > 0])
labels_df.describe()


def rstr(df, pred): 

    obs = df.shape[0]

    types = df.dtypes

    counts = df.apply(lambda x: x.count())

    uniques = df.apply(lambda x: [x.unique()])

    nulls = df.apply(lambda x: x.isnull().sum())

    distincts = df.apply(lambda x: x.unique().shape[0])

    missing_ration = (df.isnull().sum()/ obs) * 100

    skewness = df.skew()

    kurtosis = df.kurt() 

    print('Data shape:', df.shape)

    

    if pred:

        corr = df.corr()[pred]

        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)

        corr_col = 'corr '  + pred

        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]

    

    str.columns = cols

    dtypes = str.types.value_counts()

    print('___________________________\nData types:\n',str.types.value_counts())

    print('___________________________')

    return str
details = rstr(train_df_dup, 'SalePrice')

display(details.sort_values(by='corr SalePrice', ascending=False))
train_df_dup.corr()['SalePrice']
test_df.head()
test_df.isnull().sum()
# Now its time for insightful visualization
# We can clearly see a clear and strong relationship between the 'SalePrice' and 'OverQual' which makes perfect sense as we all

# know that ovelall quality of house matters the most while buying the house.
labels_df.isnull().sum() # Hence no nan values in Salesprice
labels_df.shape,train_df.shape # Thus both have same no of records so no issue with Salesprice 
labels_df.describe() 
import seaborn as sns

plt.figure(figsize=(16, 6))

sns.set(style="whitegrid")

ax = sns.boxplot(x=labels_df)
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))

sns.distplot(train_df_dup['SalePrice']);
def fig_plot(data, measure):

    fig = plt.figure(figsize=(20,7))



    #Get the fitted parameters used by the function

    (mu, sigma) = norm.fit(data)



    #Kernel Density plot

    fig1 = fig.add_subplot(121)

    sns.distplot(data, fit=norm)

    fig1.set_title(measure + ' Distribution ( mu = {:.2f} and sigma = {:.2f} )'.format(mu, sigma), loc='center')

    fig1.set_xlabel(measure)

    fig1.set_ylabel('Frequency')



    #QQ plot

    fig2 = fig.add_subplot(122)

    res = probplot(data, plot=fig2)

    fig2.set_title(measure + ' Probability Plot (skewness: {:.6f} and kurtosis: {:.6f} )'.format(data.skew(), data.kurt()), loc='center')



    plt.tight_layout()

    plt.show()

fig_plot(train_df_dup.SalePrice, 'Sales Price')
labels_df = pd.DataFrame(labels_df)
labels_df['SalePrice'].head()

# We will try the log transformation to see that can we change it to a normal distribution.


labels_df.SalePrice = np.log1p(labels_df.SalePrice)



fig_plot(labels_df.SalePrice, 'Log1P of Sales Price')
def rstr(df, pred): 

    obs = df.shape[0]

    types = df.dtypes

    counts = df.apply(lambda x: x.count())

    uniques = df.apply(lambda x: [x.unique()])

    nulls = df.apply(lambda x: x.isnull().sum())

    distincts = df.apply(lambda x: x.unique().shape[0])

    missing_ration = (df.isnull().sum()/ obs) * 100

    skewness = df.skew()

    kurtosis = df.kurt() 

    print('Data shape:', df.shape)

    

    if pred:

        corr = df.corr()[pred]

        str = pd.concat([types, counts, distincts, nulls, missing_ration, uniques, skewness, kurtosis, corr], axis = 1, sort=False)

        corr_col = 'corr '  + pred

        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'uniques', 'skewness', 'kurtosis', corr_col ]

    

    str.columns = cols

    dtypes = str.types.value_counts()

    print('___________________________\nData types:\n',str.types.value_counts())

    print('___________________________')

    return str
details = rstr(train_df_dup, 'SalePrice')

display(details.sort_values(by='corr SalePrice', ascending=False))
fig_plot(train_df_dup.OverallQual, 'OverallQual')
data.set_index('Id',inplace =True)
data.head()
data.isnull().sum() > 0

print(data.shape)
data['HouseArea'] = data['GrLivArea']+data['1stFlrSF'] 

+ data['2ndFlrSF']- data['LowQualFinSF']

data.drop(['GrLivArea','1stFlrSF','2ndFlrSF','LowQualFinSF'],axis = 1,inplace = True)

print(data.shape)
train_df_dup['HouseArea'] = train_df_dup['GrLivArea']+train_df_dup['1stFlrSF'] 

+ train_df_dup['2ndFlrSF']- train_df_dup['LowQualFinSF']

train_df_dup.drop(['GrLivArea','1stFlrSF','2ndFlrSF','LowQualFinSF'],axis = 1,inplace = True)

print(train_df_dup.shape)
test_df['HouseArea'] = test_df['GrLivArea']+test_df['1stFlrSF'] 

+ test_df['2ndFlrSF']- test_df['LowQualFinSF']

test_df.drop(['GrLivArea','1stFlrSF','2ndFlrSF','LowQualFinSF'],axis = 1,inplace = True)

print(test_df.shape)
train_df_dup.isnull().sum() > 0 
data.isnull().sum()
data.describe()
# Now basement

print(train_df_dup['BsmtFinSF1'].isna().sum());

print(train_df_dup['BsmtFinSF2'].isna().sum());

print(train_df_dup['BsmtUnfSF'].isna().sum());

print(test_df['BsmtFinSF1'].isna().sum());

print(test_df['BsmtFinSF2'].isna().sum());

print(test_df['BsmtUnfSF'].isna().sum());

print(data['BsmtFinSF1'].isna().sum());

print(data['BsmtFinSF2'].isna().sum());

print(data['BsmtUnfSF'].isna().sum());
test_df['BsmtFinSF1'].fillna(0,inplace =True)

test_df['BsmtFinSF2'].fillna(0,inplace =True)

test_df['BsmtUnfSF'].fillna(0,inplace = True)

print(test_df['BsmtFinSF1'].isna().sum());

print(test_df['BsmtFinSF2'].isna().sum());

print(test_df['BsmtUnfSF'].isna().sum());

data['BsmtFinSF1'].fillna(0,inplace =True)

data['BsmtFinSF2'].fillna(0,inplace =True)

data['BsmtUnfSF'].fillna(0,inplace = True)

print(data['BsmtFinSF1'].isna().sum());

print(data['BsmtFinSF2'].isna().sum());

print(data['BsmtUnfSF'].isna().sum());



data['BasementArea'] = (data['BsmtFinSF1'] ** 2 / data['TotalBsmtSF'])

+(data['BsmtFinSF2'] ** 2 / data['TotalBsmtSF'])

-(data['BsmtUnfSF'] ** 2 / data['TotalBsmtSF']);

data.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis = 1,inplace = True)

print(data.shape)
train_df_dup['BasementArea'] = (train_df_dup['BsmtFinSF1'] ** 2 / train_df['TotalBsmtSF'])

+(train_df_dup['BsmtFinSF2'] ** 2 / train_df['TotalBsmtSF'])

-(train_df_dup['BsmtUnfSF'] ** 2 / train_df['TotalBsmtSF']);

train_df_dup.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis = 1,inplace = True)

print(train_df_dup.shape)
test_df['BasementArea'] = (test_df['BsmtFinSF1'] ** 2 / test_df['TotalBsmtSF'])

+(test_df['BsmtFinSF2'] ** 2 / test_df['TotalBsmtSF'])

-(test_df['BsmtUnfSF'] ** 2 / test_df['TotalBsmtSF']);

test_df.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis = 1,inplace = True)

print(test_df.shape)
print(train_df_dup['BasementArea'].isna().sum(),

data['BasementArea'].isna().sum())



train_df_dup['BasementArea'].fillna(0,inplace = True)

test_df['BasementArea'].fillna(0,inplace =True)

data['BasementArea'].fillna(0,inplace =True)

print(train_df_dup['BasementArea'].isna().sum(),

test_df['BasementArea'].isna().sum(),

data['BasementArea'].isna().sum())
data['LotFrontage'].fillna(data['LotFrontage'].median()

,inplace =True)
data['LotFrontage'].isnull().sum()
train_df_dup['LotFrontage'].fillna(train_df_dup['LotFrontage'].median()

,inplace =True)

train_df_dup['LotFrontage'].isna().sum()
test_df['LotFrontage'].fillna(test_df['LotFrontage'].median()

,inplace =True)
train_df_dup.corr()['SalePrice']
data.drop(['MSSubClass'],axis = 1,inplace = True)

print(data.shape)
train_df_dup.drop(['MSSubClass'],axis = 1,inplace = True)

print(train_df_dup.shape)
test_df.drop(['MSSubClass'],axis = 1,inplace = True)

print(test_df.shape)
import seaborn as sns

sns.set(style="whitegrid")

ax = sns.barplot(x="MSZoning", y="SalePrice", data=train_df_dup)
data['MSZoning'].isna().sum()

data['MSZoning'].fillna('FV',inplace =True)

data['MSZoning'].isna().sum()

Ms = pd.get_dummies(train_df_dup['MSZoning'])

Ms1 = pd.get_dummies(test_df['MSZoning'])

Ms2 = pd.get_dummies(data['MSZoning'])
data['MSZoning'].value_counts()





train_df_dup['MSZoning'].value_counts()
train_df_dup = pd.concat([train_df_dup,Ms],axis = 1)

print(train_df_dup.shape)

test_df =  pd.concat([test_df,Ms1],axis = 1)

print(test_df.shape)
data = pd.concat([data,Ms2],axis = 1)

print(data.shape)
print(train_df_dup['LotFrontage'].isna().sum(),

data['LotFrontage'].isna().sum())

print(train_df_dup['LotArea'].isna().sum(),

data['LotArea'].isna().sum())

print(train_df_dup['Street'].value_counts())

print(train_df_dup['Alley'].value_counts())

print(test_df['Street'].value_counts())

print(test_df['Alley'].value_counts())

print(data['Street'].value_counts())

print(data['Alley'].value_counts())

replace_names = {"Street" : {"Grvl" : 0,"Pave" : 1},

                 "Alley" : {"Grvl": 1 ,"Pave" : 2 , "NA" : 0}}

train_df_dup.replace(replace_names,inplace =True)

test_df.replace(replace_names,inplace =True)

data.replace(replace_names,inplace =True)

print(data['Street'].value_counts())

print(data['Alley'].value_counts())

print(train_df_dup['Street'].value_counts())

print(train_df_dup['Alley'].value_counts())

print(test_df['Street'].value_counts())

print(test_df['Alley'].value_counts())
dict(train_df_dup.groupby('LotShape')['SalePrice'].mean())
print(train_df_dup.shape,

test_df.shape,data.shape)
from scipy import stats

sns.distplot(train_df_dup['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_df_dup['SalePrice'], plot=plt)
train_df_dup.isnull().sum()

train_df_dup['Alley'].isna().sum()

test_df['Alley'].isna().sum()

data['Alley'].isna().sum()
train_df_dup['Alley'].value_counts()
test_df['Alley'].value_counts()
data['Alley'].value_counts()
train_df_dup.Alley.fillna(0,inplace = True)

test_df.Alley.fillna(0,inplace = True)

data.Alley.fillna(0,inplace = True)
data.Alley.value_counts()

data.isna().sum()
train_df_dup['Alley'].value_counts()
test_df['Alley'].value_counts()
display(train_df_dup.Electrical.value_counts())

display(test_df.Electrical.value_counts())



train_df_dup.Electrical.fillna('SBrKr',inplace = True)

train_df_dup.Electrical.isnull().sum()
data.Electrical.fillna('SBrKr',inplace = True)

data.Electrical.isnull().sum()
list = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
for i in list:

    print(train_df_dup[i].value_counts())

    train_df_dup[i].fillna(train_df_dup[i].mode()[0],inplace = True)

    print(train_df_dup[i].isnull().sum())
list1 = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','TotalBsmtSF','MasVnrType'

         ,'MasVnrArea','Utilities','Exterior1st','MSZoning','Exterior2nd','SaleType']

for j in list1:

    print(test_df[j].value_counts())

    test_df[j].fillna(test_df[j].mode()[0],inplace = True)

    print(test_df[j].isnull().sum())
list2 = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','TotalBsmtSF','MasVnrType'

         ,'MasVnrArea','Utilities','Exterior1st','MSZoning','Exterior2nd','SaleType']

for j in list2:

    print(data[j].value_counts())

    data[j].fillna(data[j].mode()[0],inplace = True)

    print(data[j].isnull().sum())
data.isna().sum()
train_df_dup.isna().sum()
train_df_dup.MiscFeature.value_counts()
train_df_dup.MiscFeature.fillna('None',inplace = True)
train_df_dup.MiscFeature.value_counts()
train_df_dup.Fence.value_counts()
train_df_dup.Fence.fillna('None',inplace = True)
train_df_dup.Fence.value_counts()
train_df_dup.PoolQC.value_counts()
train_df_dup.PoolQC.fillna('None',inplace = True)
train_df_dup.PoolQC.value_counts()
l3 = ['PoolQC','Fence','MiscFeature']

for i in l3:

    test_df[i].fillna('None',inplace = True)

    print(test_df[i].isna().sum())
l4 = ['PoolQC','Fence','MiscFeature']

for i in l4:

    data[i].fillna('None',inplace = True)

    print(data[i].isna().sum())
l1 = ['FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond'] 

for j in l1:

    print(train_df_dup[j].value_counts())

    train_df_dup[j].fillna(train_df_dup[j].mode()[0],inplace = True)

    print(train_df_dup[j].isnull().sum())
l2 = ['FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars','GarageArea'

      ,'BsmtFullBath','BsmtHalfBath','Functional','KitchenQual'] 

for j in l2:

    print(test_df[j].value_counts())

    test_df[j].fillna(test_df[j].mode()[0],inplace = True)

    print(test_df[j].isnull().sum())
l5 = ['FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageCars','GarageArea'

      ,'BsmtFullBath','BsmtHalfBath','Functional','KitchenQual'] 

for j in l2:

    print(data[j].value_counts())

    data[j].fillna(data[j].mode()[0],inplace = True)

    print(data[j].isnull().sum())
train_df_dup.isnull().sum().sum()
data.isnull().sum().sum()
test_df.shape

train_df_dup.shape
data.shape




#Changing OverallCond into a categorical variable

data['OverallCond'] = data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'OverallCond', 

        'YrSold', 'MoSold')
for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(data[c].values)

    data[c] = lbl.transform(data[c].values)



# shape        

print('Shape all_data: {}'.format(data.shape))
data = pd.get_dummies(data)

print(data.shape)
Train = data[:1460]
Train.shape
Test = data[1460:]
Test.shape
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
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(Train.values)

    rmse= np.sqrt(-cross_val_score(model, Train.values, labels_df, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
labels_df.describe()
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
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
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

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(),

                                                score.std()))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
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
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
model_xgb.fit(Train,labels_df)
Submission = pd.DataFrame()

Submission['LogSalePrice'] = model_xgb.predict(Test)
predictions = model_xgb.predict(Test)

predictions = np.expm1(predictions)

Submission['SalePrice'] = predictions
Submission['Id'] = test_df['Id']
Submission.head()
Submission.drop('LogSalePrice',axis = 1,inplace =True)
Submission.set_index('Id',inplace =True)
Submission['SalePrice'].describe()
labels_df.SalePrice = np.log1p(labels_df.SalePrice)



fig_plot(labels_df.SalePrice, 'Log1P of Sales Price')


fig_plot(Submission.SalePrice, 'Sales Price')
Submission.head()
Output = Submission.to_csv('Output.csv')
Submission.to_csv('Output.csv')