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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#input the dataset
df = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_test.shape
#EDA
df.head()
df.info()
df.shape
#checking non integer features
df.select_dtypes(include='object').columns
df.select_dtypes(include ='int').columns
df.select_dtypes(include='float').columns
df.select_dtypes(include=['category','int','float']).dtypes
df.select_dtypes(include=['object']).dtypes
df.info(memory_usage='deep')
df.SalePrice.max()
df.SalePrice.min()
df.SalePrice.mean()
df.SalePrice.std()
df.SalePrice.median()
df.SalePrice.kurtosis()

#Long tail
df.SalePrice.skew()

#Left skewed
df.SalePrice.hist(bins =75, rwidth = 0.8,figsize = (10,4))

plt.show()
#how old is the house

print(list(df.columns))
df.SaleCondition.head()
df.SaleCondition.describe()
df.MSSubClass.describe()
df.MSSubClass.hist(bins = 30, rwidth = 10, figsize = (14,4))

plt.show()
df.MSZoning.describe()
pd.set_option('display.max_rows',30)

pd.set_option('display.max_columns',50)

df.describe()
df.OverallQual.hist(bins = 75, rwidth = 30)

plt.show()
df.OverallQual.skew()

#fairly symmetrical
df.OverallQual.kurtosis()
import seaborn as sns

var = 'OverallQual'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

#f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)
pd.set_option('display.max_rows',30)

pd.set_option('display.max_columns',50)

df.select_dtypes(include=['category','int','float']).dtypes
# When were the houses built?

print('Oldest house built in {}. Newest house built in {}.'.format(

    df.YearBuilt.min(), df.YearBuilt.max()))

df.YearBuilt.hist(bins=14, rwidth=.9, figsize=(12,4))

plt.title('When were the houses built?')

plt.show()
var = 'YearBuilt'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
# When where houses sold?

df.groupby(['YrSold','MoSold']).Id.count().plot(kind='bar', figsize=(14,4))

plt.title('When where houses sold?')

plt.show()
df.groupby('Neighborhood').Id.count().sort_values().plot(kind='barh',figsize = (6,6))
# How big are houses

print('The average house has {:,.0f} sq ft of space, the median {:,.0f} sq ft'.format(

    df.GrLivArea.mean(), df.GrLivArea.median()))

print('The biggest house has {:,.0f} sq ft of space, the smallest {:,.0f} sq ft'.format(

    df.GrLivArea.max(), df.GrLivArea.min()))

df.GrLivArea.hist(bins=21, rwidth=.8, figsize=(8,4))

plt.title('How big are houses? (in sq feet)')

plt.show()
# How big are lots

sqft_to_acres = 43560.

print('The average lot is {:,.2f} acres, the median {:,.2f} acres'.format(

    df.LotArea.mean()/sqft_to_acres, df.LotArea.median()/sqft_to_acres))

print('The biggest lot is {:,.2f} acres, the smallest {:,.2f} acres'.format(

    df.LotArea.max()/sqft_to_acres, df.LotArea.min()/sqft_to_acres))

(df.LotArea/sqft_to_acres).hist(bins=50, rwidth=.7, figsize=(8,4))

plt.title('How big are lots? (in acres)')

plt.show()
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))


corrmat = df.corr()

f, ax = plt.subplots(figsize=(20, 12))

sns.heatmap(corrmat, vmax=.8, square=True)
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
df.groupby(['YearBuilt','SalePrice']).Id.count().plot(figsize = (25,4))

plt.show()
#scatter plot totalbsmtsf/saleprice

var = 'YearBuilt'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#scatterplot

#sns.set()

#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

#sns.pairplot(df[cols], size =3)

#plt.show()
df.SalePrice.hist()
df['logSalePrice'] = df['SalePrice'].apply(lambda x: np.log(x))
df['logSalePrice'].hist(bins = 100,rwidth = 1,figsize = (10,4))
df['logSalePrice'].describe()
df['logSalePrice'].skew()
df['logSalePrice'].kurtosis()
pd.set_option('display.max_rows',100)

pd.set_option('display.max_columns',50)

df.isnull().sum()
df.shape[0]






percent_missing = df.isnull().sum() * 100 / len(df)

missing_value_df = pd.DataFrame({'column_name': df.columns,

                                 'percent_missing': percent_missing})

    
missing_value_df.sort_values('percent_missing', inplace=True, ascending = False)
missing_value_df
#remove all missing features with more than 40% missing values

#PoolQC	PoolQC	99.520548

#MiscFeature	MiscFeature	96.301370

#Alley	Alley	93.767123

#Fence	Fence	80.753425

#FireplaceQu	FireplaceQu	47.260274



df.drop(columns = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis =1, inplace = True)
df_test.head()
df_test.drop(columns = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis =1, inplace = True)
df.drop(columns = ['LotFrontage'],axis =1, inplace = True)

df_test.drop(columns = ['LotFrontage'],axis =1, inplace = True)
percent_missing = df.isnull().sum() * 100 / len(df)

missing_value_df = pd.DataFrame({'column_name': df.columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True, ascending = False)
missing_value_df
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
df.drop(columns = ['GarageCars'],axis =1, inplace = True)

df_test.drop(columns = ['GarageCars'],axis =1, inplace = True)
#scatter plot totalbsmtsf/saleprice

var = 'TotRmsAbvGrd'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
var = 'TotRmsAbvGrd'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90)
var = 'GrLivArea'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.scatterplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)

plt.xticks(rotation=90);
var = 'GrLivArea'

data = pd.concat([df['SalePrice'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.lineplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000)

plt.xticks(rotation=90);
percent_missing = df.isnull().sum() * 100 / len(df)

missing_value_df = pd.DataFrame({'column_name': df.columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True, ascending = False)
missing_value_df
df.GarageType

df.GarageCond.describe()
missing_value_df
df.select_dtypes(include=['object']).dtypes
df.GarageType.dtype

df.GarageType.value_counts()


carrier_count = df['GarageType'].value_counts()

sns.set(style="darkgrid")

sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)

plt.title('Frequency Distribution of Carriers')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Garagetype', fontsize=12)

plt.show()


carrier_count = df['GarageCond'].value_counts()

sns.set(style="darkgrid")

sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)

plt.title('Frequency Distribution of Carriers')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Garagetype', fontsize=12)

plt.show()
df.GarageCond.value_counts()
df.MSZoning.value_counts()
df.Street.value_counts()
df.select_dtypes(include=['object']).dtypes
missing_value_df
df.GarageType.value_counts()
df.GarageType.isnull().sum()


carrier_count = df['GarageType'].value_counts()

sns.set(style="darkgrid")

sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)

plt.title('Frequency Distribution of Garagetype')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Garagetype', fontsize=12)

plt.show()
pd.set_option('display.max_rows',300)

pd.set_option('display.max_columns',50)

df.SalePrice.loc[df.GarageType.isnull() == True]
df_garagetype_null = pd.DataFrame(df.loc[df.GarageType.isnull() == True])
df_garagetype_null.reset_index()


carrier_count = df_garagetype_null['OverallQual'].value_counts()

sns.set(style="darkgrid")

sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)

plt.title('Frequency Distribution of Garagetype')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Garagetype', fontsize=12)

plt.show()
#Remove missing values

df_train = df.dropna()
df_train.isnull().sum().sum()
df_train.drop(columns = ['SalePrice'],axis =1,inplace = True)
df_train.shape
# skewness along the index axis 

df_train.skew(axis = 0, skipna = True).sort_values(ascending= False) 
var = 'MiscVal'

data = pd.concat([df_train['logSalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.scatterplot(x=var, y="logSalePrice", data=data)

#fig.axis(ymin=0, ymax=800000)

#plt.xticks(rotation=90);
#remove skewness
def remove_skewness(var):

    df_train[var] = df_train[var].replace(0,np.NaN)

    df_test[var] = df_test[var].replace(0,np.NaN)

    avg = df_train[var].median()

    df_train[var] = df_train[var].replace(np.NaN, avg)

    df_test[var] = df_test[var].replace(np.NaN,avg)

        
remove_skewness('MiscVal')
df_train['MiscVal'].skew()
df_train['MiscVal'].isnull().sum()
#avg
var = 'LotArea'

data = pd.concat([df_train['logSalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.scatterplot(x=var, y="logSalePrice", data=data)

#fig.axis(ymin=0, ymax=800000)

#plt.xticks(rotation=90);
remove_skewness('LotArea')
df_train['LotArea'].skew()
df_train['LotArea'].isna().sum()
df_train
df_train.LotArea.value_counts()
var = 'LowQualFinSF'

data = pd.concat([df_train['logSalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.scatterplot(x=var, y="logSalePrice", data=data)

#fig.axis(ymin=0, ymax=800000)

#plt.xticks(rotation=90);
df.LowQualFinSF.value_counts()
remove_skewness('LowQualFinSF')

df_train['LowQualFinSF'].skew()
df_train['LowQualFinSF'].isnull().sum()
var = 'LowQualFinSF'

data = pd.concat([df_train['logSalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.scatterplot(x=var, y="logSalePrice", data=data)

#fig.axis(ymin=0, ymax=800000)

#plt.xticks(rotation=90);
var = '3SsnPorch'

data = pd.concat([df_train['logSalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.scatterplot(x=var, y="logSalePrice", data=data)

#fig.axis(ymin=0, ymax=800000)

#plt.xticks(rotation=90);
df_train['3SsnPorch'].skew()
remove_skewness('3SsnPorch')
df_train['3SsnPorch'].skew()
df_train['3SsnPorch']
var = 'KitchenAbvGr'

data = pd.concat([df_train['logSalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.boxplot(x=var, y="logSalePrice", data=data)

#fig.axis(ymin=0, ymax=800000)

#plt.xticks(rotation=90);
remove_skewness('KitchenAbvGr')
df_train.KitchenAbvGr.skew()
var = 'BsmtHalfBath'

data = pd.concat([df_train['logSalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.boxplot(x=var, y="logSalePrice", data=data)

#fig.axis(ymin=0, ymax=800000)

#plt.xticks(rotation=90);
df_train.BsmtHalfBath.value_counts()
df_train.BsmtHalfBath.value_counts()

remove_skewness('BsmtHalfBath')
df_train['BsmtHalfBath'].skew()
df_test['BsmtHalfBath'].skew()
var = 'ScreenPorch'

data = pd.concat([df_train['logSalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.scatterplot(x=var, y="logSalePrice", data=data)

#fig.axis(ymin=0, ymax=800000)

#plt.xticks(rotation=90);
df_train.ScreenPorch.value_counts()
def remove_skewness_int(var):

    df_train[var] = df_train[var].replace(0,np.NaN)

    df_test[var] = df_test[var].replace(0,np.NaN)

    avg = int(df_train[var].mean())

    df_train[var] = df_train[var].replace(np.NaN, avg)

    df_test[var] = df_test[var].replace(np.NaN,avg)
df_train.EnclosedPorch.value_counts()

#df_test.EnclosedPorch.value_counts()
remove_skewness_int('ScreenPorch')
df_train.ScreenPorch.value_counts()
df_train.ScreenPorch.skew()
remove_skewness_int('ScreenPorch')
df_train.ScreenPorch.skew()
df_train.ScreenPorch.value_counts()
df_train.MasVnrArea.value_counts()
remove_skewness_int('MasVnrArea')
df_train.MasVnrArea.skew()
df_train
df.MiscVal.value_counts()
var = 'PoolArea'

data = pd.concat([df_train['logSalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(22, 8))

fig = sns.scatterplot(x=var, y="logSalePrice", data=data)

#fig.axis(ymin=0, ymax=800000)

#plt.xticks(rotation=90);
df_train.PoolArea.skew()
remove_skewness_int('PoolArea')
df_train.PoolArea.skew()
df_train.skew(axis = 0, skipna = True).sort_values(ascending=False) 
df_train.LowQualFinSF.value_counts()
#conversion of categorical variables to numeric
df_train.head()
df_train.shape
df_test.shape
len(df_train.select_dtypes(include='object').columns)
len(df_test.select_dtypes(include = 'object').columns)
df_train.select_dtypes(include='object').columns
df_test.shape
#Imputation to avoid removing the missing values in test dataset
df_test.isnull().sum().sort_values(ascending = False)
df_train.isnull().sum().sum()
#df_test = df_test.dropna(how = 'any',axis =0)
df_test.shape
df_test.isnull().sum().sum()
df_train_dummy = pd.get_dummies(df_train)

df_test_dummy = pd.get_dummies(df_test)
df_train_dummy.shape
df_test_dummy.shape
y = df_train_dummy['logSalePrice']

X = df_train_dummy.drop(['logSalePrice','Id'], axis=1) 

df_test_dummy = df_test_dummy.drop(['Id'],axis =1)

X.shape
df_test_dummy.shape
# Get missing columns in the training test

missing_cols = set( X.columns ) - set( df_test_dummy.columns )

# Add a missing column in test set with default value equal to 0

for c in missing_cols:

    df_test_dummy[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

df_test_dummy = df_test_dummy[X.columns]
df_test_dummy.shape
df_test_dummy.isnull().sum().sum()
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

imputed_df_train = pd.DataFrame(my_imputer.fit_transform(X))

imputed_df_test = pd.DataFrame(my_imputer.transform(df_test_dummy))



# Imputation removed column names; put them back

imputed_df_train.columns = X.columns

imputed_df_test.columns = df_test_dummy.columns



#print("MAE from Approach 2 (Imputation):")

#print(score_dataset(imputed_df_train, imputed_df_test, y_train, y_valid))
imputed_df_train.shape
imputed_df_test.shape
#renaming the imputed test back to df_test_dummy to avoid tinkering with code below
df_test_dummy = imputed_df_test.copy()
#standardize the data

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
from sklearn.preprocessing import RobustScaler

mm_scaler = preprocessing.RobustScaler()

X_train_minmax = mm_scaler.fit_transform(X_train)

X_test_minmax = mm_scaler.transform(X_test)

df_test_minmax = mm_scaler.transform(df_test_dummy)
X_train_scaled = pd.DataFrame(X_train_minmax, index=X_train.index, columns=X_train.columns)

X_test_scaled = pd.DataFrame(X_test_minmax, index=X_test.index, columns=X_test.columns)

df_test_scaled = pd.DataFrame(df_test_minmax, index=df_test_dummy.index, columns=df_test_dummy.columns)
X_train_scaled.head()
X_test_scaled.head()
df_test_scaled.head()
#Baseline Model
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import VotingRegressor

from sklearn.linear_model import LinearRegression

import mlxtend

from sklearn import ensemble
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
X_train_scaled.shape
X_train_scaled.head()
X_test_scaled.shape
y_train.shape
y_train = y_train.values.reshape(-1,1)

y_train.shape

y_test = y_test.values.reshape(-1,1)
y_test.shape
df_test_scaled.shape
from sklearn.model_selection import KFold





def rmse_cv(estimator, xtrain, ytrain, cv=3):

    kfold = KFold(n_splits=cv)

    results = list()

    

    for train_idx, test_idx in kfold.split(xtrain):

        estimator.fit(xtrain[train_idx], ytrain[train_idx])

        predicted = estimator.predict(xtrain[test_idx])

        actual = ytrain[test_idx]

        mse = mean_squared_error(predicted, actual)

        rmse = np.sqrt(mse)

        results.append(rmse)

        

    return np.array(results).mean()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor



tree1 = DecisionTreeRegressor(random_state=420)

rmse_cv(tree1, X_train_minmax, y_train)
from sklearn.model_selection import GridSearchCV

tree2 = DecisionTreeRegressor(random_state=420)

parameters = {

    'splitter': ['best', 'random'],

    'max_depth': [10, 100, 1000],

    'min_samples_split': [10, 100, 1000],

    'min_samples_leaf': [10, 100, 1000]

}



tree2 = GridSearchCV(DecisionTreeRegressor(random_state=420), parameters)

rmse_cv(tree2, X_train_minmax, y_train)
from sklearn.ensemble import RandomForestRegressor



forest1 = RandomForestRegressor(random_state=420)

parameters = {

    'n_estimators': [10, 100],

    'max_depth': [10, 100],

    'min_samples_split': [10, 100],

    'min_samples_leaf': [10, 100],

}

forest1 = GridSearchCV(forest1, parameters)

rmse_cv(forest1, X_train_minmax, y_train.reshape(-1))
from sklearn.linear_model import Ridge



parameters = {

    'alpha': np.arange(0.1, 1, 0.25),

    'solver': ['svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag'],

}



ridge1 = GridSearchCV(Ridge(random_state=420),parameters)

ridge1.fit(X_train_minmax, y_train)
kfold = KFold(n_splits=3)

results = list()

    

for train_idx, test_idx in kfold.split(X_train_minmax):

    ridge1.fit(X_train_minmax[train_idx], y_train[train_idx])

    predicted_standardized = ridge1.predict(X_train_minmax[test_idx])

    predicted = predicted_standardized * y_train[train_idx].std() + y_train[train_idx].mean()

    actual = y_train[test_idx]

    mse = mean_squared_error(predicted, actual)

    rmse = np.sqrt(mse)

    results.append(rmse)

        

np.array(results).mean()
from sklearn.linear_model import Lasso



parameters = {

    'alpha': np.arange(0.1, 1, 0.1),

    'fit_intercept': [True, False],

    'selection': ['cyclic', 'random']

}



lasso1 = GridSearchCV(Lasso(random_state=420),parameters)



kfold = KFold(n_splits=3)

results = list()

    

for train_idx, test_idx in kfold.split(X_train_minmax):

    lasso1.fit(X_train_minmax[train_idx], y_train[train_idx])

    predicted_standardized = lasso1.predict(X_train_minmax[test_idx])

    predicted = predicted_standardized * y_train[train_idx].std() + y_train[train_idx].mean()

    actual = y_train[test_idx]

    mse = mean_squared_error(predicted, actual)

    rmse = np.sqrt(mse)

    results.append(rmse)

        

np.array(results).mean()


parameters = {

    'alpha': np.arange(0.1, 1, 0.1),

    'l1_ratio': np.arange(0.1, 1, 0.1),

    'fit_intercept': [True, False],

    'selection': ['cyclic', 'random']

}



elastic1 = GridSearchCV(ElasticNet(random_state=420),parameters)



kfold = KFold(n_splits=3)

results = list()

    

for train_idx, test_idx in kfold.split(X_train_minmax):

    elastic1.fit(X_train_minmax[train_idx], y_train[train_idx])

    predicted_standardized = elastic1.predict(X_train_minmax[test_idx])

    predicted = predicted_standardized * y_train[train_idx].std() + y_train[train_idx].mean()

    actual = y_train[test_idx]

    mse = mean_squared_error(predicted, actual)

    rmse = np.sqrt(mse)

    results.append(rmse)

        

np.array(results).mean()
import xgboost

best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)

clf = best_xgb_model.fit(X_train_minmax,y_train)




best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)



kfold = KFold(n_splits=3)

results = list()

    

for train_idx, test_idx in kfold.split(X_train_minmax):

    best_xgb_model.fit(X_train_minmax[train_idx], y_train[train_idx])

    predicted_standardized = best_xgb_model.predict(X_train_minmax[test_idx])

    predicted = predicted_standardized * y_train[train_idx].std() + y_train[train_idx].mean()

    actual = y_train[test_idx]

    mse = mean_squared_error(predicted, actual)

    rmse = np.sqrt(mse)

    results.append(rmse)

        

np.array(results).mean()
df_test_scaled1 = df_test_scaled.dropna(how = 'any',axis =0)
df_test_scaled1.shape
df_test.isnull().sum().sum()
X_train.isnull().sum().sum()
df_test_scaled.isnull().sum().sort_values(ascending = False)
forest1.fit(X_train_minmax, y_train)

forest_predictions = forest1.predict(df_test_minmax)
forest_predictions = forest_predictions * y_train.std() + y_train.mean()

forest_predictions = np.e**forest_predictions

forest_predictions
test_ids = df_test['Id']

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': forest_predictions})

submission
submission.to_csv('submission.csv',index = False)
submission
df_test.shape