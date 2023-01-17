# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

#

from sklearn import preprocessing

from statistics import mean

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

%matplotlib inline

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

desc = open('../input/data_description.txt').read()

le = preprocessing.LabelEncoder()
sampl = pd.read_csv('../input/sample_submission.csv')
data.columns
data.head()
#histogram

sns.distplot(data['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % data['SalePrice'].skew())

print("Kurtosis: %f" % data['SalePrice'].kurt())
#Lets check the correlation between other factors and Sale Price

cormat = data.corr()

f, ax = plt.subplots(figsize=(15,10))

sns.heatmap(cormat, vmax=0.8, square=True)
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data1 = pd.concat([data['SalePrice'], data[var]], axis=1)

data1.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot LotArea/saleprice

var = 'LotArea'

data1 = pd.concat([data['SalePrice'], data[var]], axis=1)

data1.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot OverallQual/saleprice

sns.distplot(data['OverallQual'], fit=norm)

fig = plt.figure()

prb = stats.probplot(data['OverallQual'], plot=plt)

var = 'OverallQual'

data1 = pd.concat([data['SalePrice'], data[var]], axis=1)

f, ax = plt.subplots(figsize=(8,6))

fig = sns.boxplot(x=var, y='SalePrice', data=data1)

fig.axis(ymin=0, ymax=800000)
#scatter plot totalbsmtsf/saleprice

var = 'YearBuilt'

data1 = pd.concat([data['SalePrice'], data[var]], axis=1)

f, ax = plt.subplots(figsize=(18,10))

fig = sns.boxplot(x=var, y='SalePrice', data=data1)

fig.axis(ymin=0, ymax=800000)

plt.xticks(rotation=90)
#scatter plot totalbsmtsf/saleprice

var = 'GarageCars'

data1 = pd.concat([data['SalePrice'], data[var]], axis=1)

data1.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#saleprice correlation matrix with higher Correlation

k = 10 #number of variables for heatmap

cols = cormat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#Remove Outlier LotArea

data = data.drop(data[data['LotArea'] > 30000].index)
mean_area = float(data['LotArea'].mean())
data['AreaMeanVar'] = data.LotArea - mean_area
# Lets fill non existing data either by zeroes for continous data, and Mode for Caregorical data

lst_d = {'LotFrontage'}

for i in lst_d:

    df = pd.DataFrame(data[i])

    if(data[i].dtypes=='float64'):

        data[i] = df.fillna(0)
#get Id of Outliar LotFrontage 

#data.loc[data['LotFrontage'] > 250]
# Let's remove the outlier LotFrontage

data = data.drop(data[data['LotFrontage'] > 250].index)
#var = 'LotFrontage'

sns.distplot(data['LotFrontage'] , fit=norm)

fig = plt.figure()

prb = stats.probplot(data['LotFrontage'], plot=plt)
# Lets fill non existing data either by zeroes for continous data, and Mode for Caregorical data

l_fill = {'Fence'}

for i in l_fill:

    df = pd.DataFrame(data[i])

    data[i] = df.fillna('None')
total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#missing_data.head(20)
# Lets remove the not needed columns (Outliers)

data = data.drop((missing_data[missing_data['Total'] > 1]).index,1)
#in Electrical column only 1 data is missing, lets delete only this observation

data = data.drop(data.loc[data['Electrical'].isnull()].index)
#Lets analysis saleprice vs grlivarea

var = 'GrLivArea'

data1 = pd.concat([data['SalePrice'], data[var]], axis=1)

data1.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#get Id of Outliar GrLivArea 

#data.loc[data['LotArea'] == 0]
#Lets sort the data by GrLivArea and Delete the Outliar values

#data.sort_values(by = 'GrLivArea', ascending=False)[:2]

data = data.drop(data[data['Id'] == 524].index)

data = data.drop(data[data['Id'] == 1299].index)
#Lets Analyze the Normality of SalePrice using Histogram and Normal Probability plot

sns.distplot(data['SalePrice'], fit=norm)

fig = plt.figure()

prb = stats.probplot(data['SalePrice'], plot=plt)
# Since Price has Positive Skewness and Peakedness, it can be resolved using log transformation

data['SalePrice'] = np.log(data['SalePrice'])
#Lets Analyze again the Normality of SalePrice using Histogram and Normal Probability plot

sns.distplot(data['SalePrice'], fit=norm)

fig = plt.figure()

prb = stats.probplot(data['SalePrice'], plot=plt)
#Lets Analyze  the Normality of GrLivArea using Histogram and Normal Probability plot

sns.distplot(data['GrLivArea'], fit=norm)

fig = plt.figure()

prb = stats.probplot(data['GrLivArea'], plot=plt)
# Since GrLivArea has Positive Skewness and Peakedness, it can be resolved using log transformation

data['GrLivArea'] = np.log(data['GrLivArea'])
#Lets Analyze  the Normality of GrLivArea using Histogram and Normal Probability plot

sns.distplot(data['GrLivArea'], fit=norm)

fig = plt.figure()

prb = stats.probplot(data['GrLivArea'], plot=plt)
#Lets Analyze  the Normality of TotalBsmtSF using Histogram and Normal Probability plot

sns.distplot(data['TotalBsmtSF'], fit=norm)

fig = plt.figure()

prb = stats.probplot(data['TotalBsmtSF'], plot=plt)
data.head()
# List of Features to remove from data

lst_remv = {'EnclosedPorch','Exterior2nd','Exterior1st','BedroomAbvGr','HouseStyle','HeatingQC',

            'Functional','SaleCondition','LandContour','YrSold','LandSlope','ExterQual','PavedDrive',

            'LotConfig','Foundation','RoofStyle','KitchenAbvGr','BsmtFullBath','HalfBath','Electrical',

            'Condition1','BsmtFinSF2','BldgType','ScreenPorch','SaleType','BsmtHalfBath','MiscVal',

            'MiscVal','Heating','RoofMatl','LowQualFinSF','Utilities','Street','PoolArea','Condition2',

            '3SsnPorch','GarageArea','MoSold','Id','CentralAir','ExterCond'}

# List of Features to label Encode

ls = {'MSSubClass','MSZoning','LotShape','Neighborhood','KitchenQual'

     ,'YearBuilt','YearRemodAdd','FullBath','TotRmsAbvGrd','Fireplaces',

     'GarageCars','Fence'}
# Feature Removal

for i in lst_remv:

    data =  data.drop([i],axis=1)
# Lets use Label Encoder to change String content to Integer content

for i in ls:

    le.fit(data[i])

    data[i] = le.transform(data[i])
train = data
reslt1 = data.filter(['SalePrice'],axis=1)
train =  train.drop(['SalePrice'],axis=1)
train.head()
#Lets split data into Train and Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, reslt1, test_size=0.2, random_state=10)

snum = 0

enum = len(y_test)
import xgboost as xgb

model= xgb.XGBClassifier(max_depth=5,booster='gbtree',learning_rate=0.05,subsample=0.5,

                         colsample_bytree=0.5,colsample_bylevel=0.5)
model.fit(X_train,y_train)

pr = model.predict(X_test[snum:enum])
names = X_train.columns.values

print("Features sorted by their score:")

print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), 

             reverse=True))
#prd1 = np.exp(pr)

#t_data = np.exp(y_test[snum:enum])

#print('Predicted result: ', prd1, '\nActual result:   ', t_data.values.reshape(1,enum))
t_data2 = y_test[snum:enum].values

t_data2 = t_data2.reshape(enum,)

total_error=0

for i in range(len(t_data2)):

    error = (t_data2[i] - pr[i])**2

    total_error = total_error+ error



total_error = total_error**(1/2) / len(t_data2)

print(total_error)
from sklearn.ensemble import GradientBoostingRegressor

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
GBoost.fit(X_train, y_train)  

pr_gbr = GBoost.predict(X_test[snum:enum])
t_datal1 = y_test[snum:enum].values

t_datal1 = t_datal1.reshape(enum,)



total_error=0

for i in range(len(t_datal1)):

    error = (t_datal1[i] - pr_gbr[i])**2

    total_error = total_error+ error



total_error = total_error**(1/2) / len(t_datal1)

print("total_error: ", total_error)
names = X_train.columns.values

print("Features sorted by their score:")

print(sorted(zip(map(lambda x: round(x, 4), GBoost.feature_importances_), names), 

             reverse=True))
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X_train, y_train)  

pr_lgb = model_lgb.predict(X_test[snum:enum])
pr1_lgb = np.exp(pr_lgb)

t_lgb = np.exp(y_test[snum:enum])

#print('Predicted result: ', prd3, '\nActual result:   ', t_lgb.values.reshape(1,enum))
t_datal1 = y_test[snum:enum].values

t_datal1 = t_datal1.reshape(enum,)



total_error=0

for i in range(len(t_datal1)):

    error = (t_datal1[i] - pr_lgb[i])**2

    total_error = total_error+ error



total_error = total_error**(1/2) / len(t_datal1)

print("total_error: ", total_error)
names = X_train.columns.values

print("Features sorted by their score:")

print(sorted(zip(map(lambda x: round(x, 4), model_lgb.feature_importances_), names), 

             reverse=True))
# Lets fill non existing data either by zeroes for continous data, and Mode for Caregorical data

test_dt = {'LotFrontage','Fence'}

for i in test_dt:

    df = pd.DataFrame(test[i])

    if(test[i].dtypes=='float64'):

        test[i] = df.fillna(0)

    elif(test[i].dtypes=='object'):

        test[i] = df.fillna('None')
totalt = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(22)
# Lets remove the not needed columns (Outliers)

test = test.drop((missing_data[missing_data['Total'] > 4]).index,1)
total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(11)
# Lets fill non existing data either by zeroes for continous data, and Mode for Caregorical data

lst = {'MSZoning','BsmtHalfBath','BsmtFullBath','Functional','Utilities',

       'Exterior2nd','KitchenQual','GarageCars','BsmtFinSF1',

       'SaleType','TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','Exterior1st'}

for i in lst:

    df = pd.DataFrame(test[i])

    if(test[i].dtypes=='float64'):

        test[i] = df.fillna(0)

    elif(test[i].dtypes=='object'):

        test[i] = df.fillna(test[i].mode())
# Adding feature 

mean_area_t = float(test['LotArea'].mean())



test['AreaMeanVar'] = test.LotArea - mean_area_t
# Log transform the test data

test['GrLivArea'] = np.log(test['GrLivArea'])
# Feature Removal

for i in lst_remv:

    test =  test.drop([i],axis=1)
# Lets use Label Encoder to change String content to Integer content

for i in ls:

    if(test[i].dtypes=='float64'):

        le.fit(test[i])

        test[i] = le.transform(test[i])

    elif(test[i].dtypes=='object'):

        le.fit(test[i].astype(str))

        test[i] = le.transform(test[i].astype(str))
snum = 0

enum = len(test)

#prdt1 = model.predict(test[snum:enum])

#prdt11 = np.exp(prdt1)

#print('Predicted result: ', prdt11)
prdt2 = model.predict(test[snum:enum])   #using Xtreme Gradient Boost Classifier

#prdt2 = model_lgb.predict(test[snum:enum])   #using LGBM

#prdt2 = GBoost.predict(test[snum:enum])      #using Gradient Boost Regressor

prdt21 = np.exp(prdt2)

print('Predicted result: ', prdt21)
sampl['SalePrice'] = pd.DataFrame(prdt21)
sampl.to_csv('submission.csv', index=False)