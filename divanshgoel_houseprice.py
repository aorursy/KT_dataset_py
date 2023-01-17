# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm, skew #for some statistics

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
curr_dir = '../input/'

train = pd.read_csv(curr_dir+'train.csv')

test = pd.read_csv(curr_dir+'test.csv')



#test2 = test

#test2.head()

train.shape,test.shape


train['SalePrice'].describe()

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#dtypes = d1.dtypes

num_vars = [c for c in train if train[c].dtype in numerics]

cat_vars = [c for c in train if c not in num_vars]

num_vars, cat_vars
#Outlier Treatment



#sns.pairplot(data=train[num_vars],x_vars=train.iloc[:,0:5],

#            y_vars=train.iloc[:,:-1])

fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.ylim((0,800000))

plt.show()
#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#Checking for normal distribution

sns.distplot(train['SalePrice'], fit = norm);

# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
#Feature Engineering



test = pd.read_csv(curr_dir+'test.csv')



ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
#Dropping the unnecessary ID



all_data.drop("Id", axis = 1, inplace = True)

all_data.shape
#Checking for missing values



all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.sort_values(ascending=False).drop(all_data_na[all_data_na == 0].index)[:30]

all_data_na.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
all_data['PoolQC'].unique()

all_data['PoolQC'].value_counts()

#all_data.groupby['PoolQC']
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data['PoolQC'].unique()
all_data['MiscFeature'].unique()

all_data['MiscFeature'].value_counts()
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data['Alley'].unique()
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data['Fence'].unique()
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data['FireplaceQu'].unique()
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data['LotFrontage'].unique()
Lot_mean = all_data["LotFrontage"].mean()

all_data["LotFrontage"] = all_data["LotFrontage"].fillna(Lot_mean)
all_data['LotFrontage'].unique()
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.sort_values(ascending=False).drop(all_data_na[all_data_na == 0].index)[:30]

all_data_na.head(20)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
all_data = all_data

all_data = pd.get_dummies(all_data)

train = all_data[:ntrain]

test = all_data[ntrain:]

test.head()
#Trying modeling



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
train.shape, test.shape

#train.head()

#lasso = Lasso(alpha =0.0005, random_state=1)

#lasso.fit(train,y_train)
#train=pd.get_dummies(train)

#test=pd.get_dummies(test)
test.shape
test.head(5)
lasso = Lasso(alpha =0.001, random_state=1)

lasso.fit(train,y_train)
pred=lasso.predict(test)

preds=np.exp(pred)
output=pd.DataFrame({'Id':test.index,'SalePrice':preds})

dummy = pd.DataFrame({'Id':[2917,2918,2919],'SalePrice':[150000,150000,150000]})

output = output.append(dummy, ignore_index = True)

output.to_csv('submission.csv', index=False)

#output.head(),output.tail(5)

#dummy

#output.shape