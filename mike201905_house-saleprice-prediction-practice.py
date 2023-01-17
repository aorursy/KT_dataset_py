# Import libararies

# Data load,Data transform

import pandas as pd

import numpy as np

# Data science packages

from scipy import stats

from scipy.stats import norm

# Data Preprocessing

from sklearn.preprocessing import StandardScaler

# Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

# Import Models

# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

# https://scikit-learn.org/stable/modules/ensemble.html

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error



#Mlxtend (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks.

#http://rasbt.github.io/mlxtend/

#https://githubja.com/rasbt/mlxtend

#https://qiita.com/altescy/items/60a6def66f13267f6347

#pip install mlxtend

from mlxtend.regressor import StackingCVRegressor



#XGBoost(eXtreme Gradient Boosting):mixed Gradient Boosting and Random forest

#https://xgboost.readthedocs.io/en/latest/

#https://blog.amedama.jp/entry/2019/01/29/235642

#https://qiita.com/R1ck29/items/4607b4ac941439ed6bbc

#pip install xgboost

from xgboost import XGBRegressor



##LightGBM: A Highly Efficient Gradient Boosting Decision Tree

#https://www.codexa.net/lightgbm-beginner/

#pip install lightgbm

from lightgbm import LGBMRegressor



# utilities

from datetime import datetime

# Disable warning output

import warnings

warnings.filterwarnings('ignore')

# Show matplot graph

%matplotlib inline
# Read data from train dataset

data_train = pd.read_csv('../input/train.csv')

#check the columns

print(data_train.columns)

# Read data from test dataset

data_test = pd.read_csv('../input/test.csv')
#Explore for each key element 

#Analyze SalePrice 

#descriptive statistics summary

#https://note.nkmk.me/python-pandas-describe/

data_train['SalePrice'].describe()
#histogram

# Don't output <matplotlib.axes._subplots.AxesSubplot at 0x1a1be67390> if exist ; at the end of plot statement

sns.distplot(data_train['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % data_train['SalePrice'].skew())

print('Kurtosis: %f' % data_train['SalePrice'].kurt())
# Step1 Hypothesis

# 1, Feature Selection

# Try to explore these features which possible influent SalePrice based on Hypothesis 

#1)Relationship with numerical variables

#scatter plot grlivarea/saleprice

var = 'GrLivArea'

#https://deepage.net/features/pandas-concat.html

data = pd.concat([data_train['SalePrice'],data_train[var]],axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#2)Relationship with categorical features

#box plot overallqual/saleprice(the SalePrice range for each OverallQual)

var = 'OverallQual'

data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)

#explain plots : https://qiita.com/tsuruokax/items/90167693f142ebb55a7d

#http://ailaby.com/matplotlib_fig/

#https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.subplots.html

f, ax = plt.subplots(figsize=(8,6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
# Relationship of YearBuilt with SalePrice

var = 'YearBuilt'

data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

# Rotate X axis 90

plt.xticks(rotation=90);

# It seems not so strong relationship with SalePrice
# Step2 Structural engineering

# Correlation matrix : https://blog.csdn.net/zzw000000/article/details/81205027

# Correlation matrix (heatmap style)

corrmat = data_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

#https://seaborn.pydata.org/generated/seaborn.heatmap.html

#https://pythondatascience.plavox.info/seaborn/heatmap

sns.heatmap(corrmat, vmax=.8, square=True);

#Very heat area: TotalBsmtSF' and '1stFlrSF' ;'GarageX' variables ;'GrLivArea', 'TotalBsmtSF', and 'OverallQual' 
#SalePrice' correlation matrix (zoomed heatmap style)

#saleprice correlation matrix

#10 features most correlated with SalePrice

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

print(cols)

print(data_train[cols].values)

print(data_train[cols].values.T)

#https://deepage.net/features/numpy-corrcoef.html

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.T.html

cm = np.corrcoef(data_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

# The 10 features have correlation with Saleprice over 0.5

# OverallQUal,GrLivArea have very strong correlation

# choose one from each pair for GarageCars similar with GarageArea(Synonym), TotalBsmtSF similar with 1stFlrSF(Synonym)

# 'TotRmsAbvGrd' and 'GrLivArea' linear? number with rooms should different with living area

# Therefore Select OverallQUal,GrLivArea,GarageCars,TotalBsmtSF,FullBath,TotRmsAbvGrd,TotRmsAbvGrd as features
#Scatter plots between 'SalePrice' and correlated variables

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd' ,'YearBuilt']

sns.pairplot(data_train[cols], size = 2.5)

plt.show();
#Feature Processing

#Missing data for train dataset and test dataset

#For the elments which the NA doesn't mean missing value based on data_description.txt

elt_name = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu'

            ,'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

data_train.update(data_train[elt_name].fillna('None'))

data_test.update(data_test[elt_name].fillna('None'))
#Feature Processing

#Missing data for train dataset

#two Important things for missing data:

#a. How prevalent is the missing data?

#b.Is missing data random or does it have a pattern?

#https://note.nkmk.me/python-pandas-nan-judge-count/

#missing data

#comment: For most of the elments the NA doesn't mean missing value based on data_description.txt

#comment: should reconsider deal with these NA elements

total = data_train.isnull().sum().sort_values(ascending=False)

percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data.head(20)

#Seems no missing data for key features:

#OverallQUal,GrLivArea,GarageCars,TotalBsmtSF,FullBath,TotRmsAbvGrd,TotRmsAbvGrd
#missing data for test data

#comment: For most of the elments the NA doesn't mean missing value based on data_description.txt

#comment: should reconsider deal with these NA elements

total_test = data_test.isnull().sum().sort_values(ascending=False)

percent_test = (data_test.isnull().sum()/data_test.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total_test,percent_test],axis=1,keys=['Total','Percent'])

missing_data_test.head(20)
#reomve all the columns with missing data existing except Electrical only reomve this row don't has value

print((missing_data[missing_data['Total'] > 1]).index)

#https://note.nkmk.me/python-pandas-drop/

#data_train = data_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

#update na using most common value

data_train.update(data_train['Electrical'].fillna(data_train['Electrical'].dropna().mode().values.item()))

col_num=[f for f in data_train.columns if data_train.dtypes[f] != 'object']

data_train.update(data_train[col_num].fillna(0))

#fill for Object columns

col_obj=[f for f in data_train.columns if data_train.dtypes[f] == 'object']

data_train.update(data_train[col_obj].fillna('None'))

data_train.isnull().sum().max() #check if missing data exists
#remove same columns from test data set

#data_test = data_test.drop((missing_data[missing_data['Total'] > 1]).index,1)

#fill missing data for test data

#https://note.nkmk.me/python-pandas-nan-dropna-fillna/

#fill for Number columns

col_num=[f for f in data_test.columns if data_test.dtypes[f] != 'object']

data_test.update(data_test[col_num].fillna(0))

#fill for Object columns

col_obj=[f for f in data_test.columns if data_test.dtypes[f] == 'object']

data_test.update(data_test[col_obj].fillna('None'))

data_test.isnull().sum().max() #check if missing data exists
#Out liars

#Outliers is a complex subject and in here just do a quick analysis --

#through the standard deviation of 'SalePrice' and a set of scatter plots.

#Univariate analysis

#standardizing data

saleprice_scaled = StandardScaler().fit_transform(data_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#Bivariate analysis

#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#The two values with the most biggest 'GrLivArea' seem strange and not following the crowd. 

#We can speculate why this is happening. Maybe they refer to agricultural area and that could explain the low price. 

#As execise, we'll define them as outliers and delete them

#deleting points

outlier_ids = data_train.sort_values(by = 'GrLivArea', ascending = False)[:2]['Id']

print(outlier_ids.index)

data_train = data_train.drop(outlier_ids.index)

print(data_train.sort_values(by = 'GrLivArea', ascending = False)[:2][['SalePrice','GrLivArea']])
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#It seems not so strange
#Check Data Quality

# 4 import factors

#1,Normality

#2,Homoscedasticity

#3,Linearity

#4,Absence of correlated errors



#Pay attention to the following 2 points for each key features

#Histogram - Kurtosis and skewness.

#Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.

#histogram and normal probability plot

#SalePrice

sns.distplot(data_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(data_train['SalePrice'], plot=plt)

#From the following graph

#1)'SalePrice' is not normal. It shows 'peakedness'

#2) show positive skewness and does not follow the diagonal line.

#Solution

#A simple data transformation can solve the problem. 

#can learn in statistical books: in case of positive skewness, log transformations usually works well.
#log transformations for SalePrice

#applying log transformation

#SalePrice onlu exists in train dataset

data_train['SalePrice'] = np.log(data_train['SalePrice'])
#transformed histogram and normal probability plot again

sns.distplot(data_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(data_train['SalePrice'], plot=plt)

#Now It seems normal and skewness follows the diagonal line.
#Feature engineering for both train and test dataset 

#Do the same for GrLivArea

#Before log transformations

#histogram and normal probability plot

sns.distplot(data_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(data_train['GrLivArea'], plot=plt)
#data transformation

data_train['GrLivArea'] = np.log(data_train['GrLivArea'])

data_test['GrLivArea'] = np.log(data_test['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(data_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(data_train['GrLivArea'], plot=plt)
# Do the same for TotalBsmtSF

# Before log transformations

#histogram and normal probability plot

sns.distplot(data_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(data_train['TotalBsmtSF'], plot=plt)

#issue:

#1)A lots of observations with value zero (houses without basement).

#2)the value zero doesn't allow to do log transformations.

#solution:

#create a new feature that can get the effect of having or not having basement (binary variable). 

#log transformation to all the non-zero observations, ignoring those with value zero. 

#Present basement by combining this two featurs

#This way we can transform data, without losing the effect of having or not basement.
#create column for new feature (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

data_train['HasBsmt'] = pd.Series(len(data_train['TotalBsmtSF']), index=data_train.index)

data_train['HasBsmt'] = 0 

data_train.loc[data_train['TotalBsmtSF']>0,'HasBsmt'] = 1

data_test['HasBsmt'] = pd.Series(len(data_test['TotalBsmtSF']), index=data_test.index)

data_test['HasBsmt'] = 0 

data_test.loc[data_test['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

data_train.loc[data_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(data_train['TotalBsmtSF'])

data_test.loc[data_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(data_test['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(data_train[data_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(data_train[data_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#Check homoscedasticity

#The best approach to test homoscedasticity for two metric variables is graphically. 

#Good graph:

#small dispersion at one side of the graph, large dispersion at the opposite side) 

#or diamonds (a large number of points at the center of the distribution).

#for GrLivArea and SalePrice

#scatter plot

plt.scatter(data_train['GrLivArea'], data_train['SalePrice']);

#It seems good after log transformation
#for TotalBsmtSF with value over 0 and SalePrice

#scatter plot

plt.scatter(data_train[data_train['TotalBsmtSF']>0]['TotalBsmtSF'], data_train[data_train['TotalBsmtSF']>0]['SalePrice']);
#dummy variables

#convert categorical variable into dummy

#https://note.nkmk.me/python-pandas-get-dummies/

data_all = pd.concat([data_train.drop(['SalePrice'], axis=1), data_test]).reset_index(drop=True)

data_all = pd.get_dummies(data_all)

data_all.drop(['Id'],axis=1,inplace=True)

#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
#Creat train Data set ,label, test dataset

y = data_train['SalePrice'].reset_index(drop=True)

#http://ailaby.com/lox_iloc_ix/

#https://note.nkmk.me/python-pandas-at-iat-loc-iloc/

x_train = data_all.iloc[:len(y), :]

x_test = data_all.iloc[len(y):, :]

#same result with the above

train = data_all[:len(y)]

test = data_all[len(y):]

x_train.shape, x_test.shape, y.shape,train.shape,test.shape
#Data Model Part

print('START ML', datetime.now(), )

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

#rmsle

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



#build our model scoring function

def cv_rmse(model, x=x_train, y=y):

    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
#setup models

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
# Create models

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                

#svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
gbr = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01, max_depth=30, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)
random_forest = RandomForestRegressor(n_estimators=2000,

                                      max_depth=30,

                                      min_samples_split=5,

                                      min_samples_leaf=5,

                                      max_features=None,

                                      random_state=10,

                                      oob_score=True

                                     )
# lightgbm = LGBMRegressor(objective='regression', 

#                                        num_leaves=4,

#                                        learning_rate=0.01, 

#                                        n_estimators=5000,

#                                        max_bin=200, 

#                                        bagging_fraction=0.75,

#                                        bagging_freq=5, 

#                                        bagging_seed=7,

#                                        feature_fraction=0.2,

#                                        feature_fraction_seed=7,

#                                        verbose=-1,

#                                        )
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
#stack

# stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, random_forest),

#                                 meta_regressor=xgboost,

#                                 use_features_in_secondary=True)

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, random_forest),

                                meta_regressor=gbr,

                                use_features_in_secondary=True)
print('TEST score on CV')

names = ["Kernel Ridge", "Lasso", "ElasticNet", "xgboost", "gbr", "stack_gen"]

models= [ridge, lasso, elasticnet, xgboost, gbr, stack_gen]

# names = ["Kernel Ridge", "Lasso", "ElasticNet", "GradientBoosting", "xgb" ,"lightgbm" ,"stack_gen"]

# models= [ridge, lasso, elasticnet, gbr, xgboost, lightgbm, stack_gen]

for name, model in zip(names, models):

    score = cv_rmse(model)

    print("{} score: {:.6f} ({:.4f})".format(name,score.mean(),score.std()))
# Predict by mixing models with respective weight

class MixModelWeight():

    def __init__(self,mod,weight):

        self.mod = mod

        self.weight = weight

    

    def fit(self,X,y):

        for model in self.mod:

            model.fit(X,y)

        return self



    def predict(self,X):

        w = list()

        pred = np.array([model.predict(X) for model in self.mod])

        # for every data point, single model prediction times weight, then add them together

        for data in range(pred.shape[1]):

            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]

            w.append(np.sum(single))

        return w



# Mix model

mix_model = MixModelWeight(mod = [elasticnet,lasso,ridge,xgboost,stack_gen],weight=[0.1,0.1,0.1,0.1,0.6])

# Fit

print('START fit model')

mix_model.fit(x_train,y)

print('RMSLE score on train data:')

score = rmsle(y, mix_model.predict(x_train))

print(score)
print('Predict submission', datetime.now(),)

submission=pd.DataFrame({'Id':data_test['Id'], 'SalePrice':np.floor(np.exp(mix_model.predict(x_test)))})

submission_stack=pd.DataFrame({'Id':data_test['Id'], 'SalePrice':np.floor(np.exp(stack_gen.predict(x_test)))})

submission_mix=pd.DataFrame({'Id':data_test['Id'], 'SalePrice':(submission['SalePrice'])*0.8+(submission_stack['SalePrice']*0.2)})

print(submission)
#output to submission csv

submission.to_csv("submission_v4_1.csv", index=False)

submission_stack.to_csv("submission_v4_2.csv", index=False)

submission_mix.to_csv("submission_v4_3.csv", index=False)

print('Save submission', datetime.now(),)