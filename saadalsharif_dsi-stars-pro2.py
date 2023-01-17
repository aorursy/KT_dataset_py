# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, Lasso

from scipy.stats import norm

import warnings

import datetime

import time





def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

plt.style.use('ggplot')

sns.set(font_scale=1.5)

%config InlineBackend.figure_format = 'retina'

%matplotlib inline
#import data file 

boston_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

boston_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#f=open('data_description.txt', 'r').print(f.read())

# To see data_description
#show all the colunms,to get view to all its data

pd.set_option('display.max_columns', None)

boston_train.head(3)
pd.set_option('display.max_columns', None)

boston_test.head(3)
boston_train.shape, boston_test.shape
boston_train.info()
boston_test.info()
pd.set_option('display.max_columns', None)

boston_train.describe()
#print colunms with object values

object_df = boston_train.select_dtypes(include=object)

object_df.head()
#print colunms with object values

object_test_df = boston_test.select_dtypes(include=object)

object_test_df.head()
#print colunms with numeric type values

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



numeric_df = boston_train.select_dtypes(include=numerics)

numeric_df.head()
numeric_test_df  = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



numeric_test_df  = boston_test.select_dtypes(include=numerics)

numeric_test_df.head()
boston_train[['YearBuilt','GarageType','GarageYrBlt']].head(3)
null_dict_t = dict(boston_test.isnull().sum().sort_values(ascending= True ))
for a, b in null_dict_t.items():

    if b > 0:

        print(a + ': ', b)

    else:

        pass
null_dict = dict(boston_train.isnull().sum().sort_values(ascending= True ))
for a, b in null_dict.items():

    if b > 0:

        print(a + ': ', b)

    else:

        pass
#isnull() return True 

#for all the places where the data is missing.



#create an inch-by-inch image

plt.figure(figsize=(12,8))



#Plot a heatmap for visualization missing data

sns.heatmap(boston_train.isnull(), cbar=True)

plt.show()
boston_train.head(2)
# data description says NA means No Garage and that mean the rest columns of Garage will be the same 

col_G = ['GarageType','GarageQual' , 'GarageCond', 'GarageFinish']

for col in col_G:

    boston_train[col] = boston_train[col].fillna('NoGarage')
# data description says NA means No Garage and that mean the rest columns of Garage will be the same 

col_G = ['GarageType','GarageQual' , 'GarageCond', 'GarageFinish']

for col in col_G:

    boston_test[col] = boston_test[col].fillna('NoGarage')
# data description says NA means No Basement

#Replacing the missing data with NoGrage (Since  NA = No Basement ) then all the missing value will be no Basement

col_G = ['BsmtQual','BsmtCond' , 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for col in col_G:

    boston_train[col] = boston_train[col].fillna('NoBasement')
#all of this columns missing it is not missing value it is mean:

# PoolQC : data description says NA means "No Pool"

# Fence : data description says NA means "No Fence"

# MiscFeature: data description says NA means "No misc value"

# Alley : data description says NA means "No alley access"

# FireplaceQu : data description says NA means "No Fireplace"



col_str = ['PoolQC', 'Fence', 'MiscFeature', 'Alley','Electrical', 'FireplaceQu','MasVnrType' ]

for col in col_str:

    boston_train[col] = boston_train[col].fillna('None')
#all of this columns missing it is not missing value it is mean:

# PoolQC : data description says NA means "No Pool"

# Fence : data description says NA means "No Fence"

# MiscFeature: data description says NA means "No misc value"

# Alley : data description says NA means "No alley access"

# FireplaceQu : data description says NA means "No Fireplace"



col_str = ['PoolQC','Utilities', 'Fence', 'MiscFeature', 'Alley','Electrical', 'FireplaceQu','MasVnrType', 'Functional', 'Electrical', 'KitchenQual', 'MSSubClass' ]

for col in col_str:

    boston_test[col] = boston_train[col].fillna('None')
# data description says NA means No Basement

#Replacing the missing data with NoGrage (Since  NA = No Basement ) then all the missing value will be no Basement

col_G = ['BsmtQual','BsmtCond' , 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for col in col_G:

    boston_test[col] = boston_test[col].fillna('NoBasement')
col_str = ['PoolQC', 'Fence', 'MiscFeature', 'Alley','Electrical', 'FireplaceQu','MasVnrType' ]

for col in col_str:

    boston_test[col] = boston_test[col].fillna('None')
# Replacing missing data with 0 (Since No garage = no cars in such garage.) and no 

col_num = ['GarageYrBlt','MasVnrArea' ]

for col in col_num:

    boston_train[col] = boston_train[col].fillna(0)

# group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood    

boston_train['LotFrontage'] = boston_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# NA values for LotFrontage with mean of column

#boston_train.LotFrontage.fillna(value=combined['LotFrontage'].mean(), inplace=True)
#changing the value of month from number to month name to make it more understand 

month_map =  {"MoSold":     {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 

         7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'} }



boston_train.replace(month_map, inplace=True)
# Replacing missing data with 0 (Since No garage = no cars in such garage.) and no 

col_num = ['GarageYrBlt','MasVnrArea' ]

for col in col_num:

    boston_test[col] = boston_test[col].fillna(0)

# group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood    

boston_test['LotFrontage'] = boston_test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#changing the value of month from number to month name to make it more understand 

month_map =  {"MoSold":     {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 

         7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'} }



boston_test.replace(month_map, inplace=True)
# Replacing missing data with 0 (Since No garage = no cars in such garage.) and no 

col_num = ['GarageYrBlt','MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath' , 'BsmtFinSF1', 'TotalBsmtSF', 'BsmtFinSF2', 'GarageCars','GarageArea' , 'BsmtUnfSF']

for col in col_num:

    boston_test[col] = boston_test[col].fillna(0)
# SaleType : Fill in again with most frequent which is "WD"

boston_test['SaleType'].fillna(boston_test['SaleType'].mode()[0], inplace=True)
plt.subplots(figsize=(10, 5))



boston_train.MoSold.value_counts().plot(kind='bar',color='gray');
Over_all_Qual =  {"OverallQual":     {1:'VeryPoor', 2:'Poor', 3:'Fair', 4:'BelowAverage', 5:'Average', 6:'Above Average', 

         7:'Good', 8:'VeryGood', 9:'Excellent', 10:'VeryExcellent'} }



boston_train.replace(Over_all_Qual, inplace=True)
Over_all_Qual =  {"OverallQual":     {1:'VeryPoor', 2:'Poor', 3:'Fair', 4:'BelowAverage', 5:'Average', 6:'Above Average', 

         7:'Good', 8:'VeryGood', 9:'Excellent', 10:'VeryExcellent'} }



boston_test.replace(Over_all_Qual, inplace=True)
plt.subplots(figsize=(10, 5))



boston_train.OverallQual.value_counts().plot(kind='bar',color='gray');
Over_all_Cond =  {"OverallCond":     {1:'VeryPoor', 2:'Poor', 3:'Fair', 4:'BelowAverage', 5:'Average', 6:'Above Average', 

         7:'Good', 8:'VeryGood', 9:'Excellent', 10:'VeryExcellent'} }



boston_train.replace(Over_all_Cond, inplace=True)
Over_all_Cond =  {"OverallCond":     {1:'VeryPoor', 2:'Poor', 3:'Fair', 4:'BelowAverage', 5:'Average', 6:'Above Average', 

         7:'Good', 8:'VeryGood', 9:'Excellent', 10:'VeryExcellent'} }



boston_test.replace(Over_all_Cond, inplace=True)
plt.subplots(figsize=(10, 5))



boston_train.OverallCond.value_counts().plot(kind='bar',color='gray');
#Some of the non-numeric features are stored as numbers. They should be converted to strings.

#train data frame

boston_train['MSSubClass'] = boston_train['MSSubClass'].apply(str)

boston_train['YrSold'] = boston_train['YrSold'].apply(str)

boston_train['MoSold'] = boston_train['MoSold'].apply(str)

# Changing OverallCond into a categorical variable

boston_train['OverallCond'] = boston_train['OverallCond'].astype(str)

boston_train['OverallQual'] = boston_train['OverallQual'].astype(str)
#Some of the non-numeric features are stored as numbers. They should be converted to strings.

#test data frame

boston_test['MSSubClass'] = boston_test['MSSubClass'].apply(str)

boston_test['YrSold'] = boston_test['YrSold'].apply(str)

boston_test['MoSold'] = boston_test['MoSold'].apply(str)

# Changing OverallCond into a categorical variable

boston_test['OverallCond'] = boston_test['OverallCond'].astype(str)

boston_test['OverallQual'] = boston_test['OverallQual'].astype(str)
#replace the all missing values in colunms MSZoning with RM

boston_test.MSZoning.replace(np.nan,'RM', inplace=True, regex=True)
#replace the NaN value in columns 'Exterior1st','Exterior2nd' with "other"

col_G = ['Exterior1st','Exterior2nd']

for col in col_G:

    boston_test[col] = boston_test[col].fillna('Other')
boston_test['MSZoning'] = boston_test['MSZoning'].astype(str)
boston_test.GarageCars.replace(np.nan,0, inplace=True, regex=True)

boston_test.GarageArea.replace(np.nan,0, inplace=True, regex=True)
#check if there any missing values test data frame

null_dict_t = dict(boston_test.isnull().sum())



for a, b in null_dict_t.items():

    if b > 0:

        print(a + ': ', b)

    else:

        pass
#check if there any missing values

null_dict= dict(boston_train.isnull().sum())

for a, b in null_dict.items():

    if b > 0:

        print(a + ': ', b)

    else:

        pass
for col in list(boston_train.corr()[['SalePrice']].sort_values('SalePrice', ascending=False).index.values):

    

    if col != 'SalePrice':

        plt.figure()

        plt.ylabel('Sale Price')

        plt.xlabel(col)

        plt.scatter(boston_train[col], boston_train['SalePrice']);

    else:

        pass
sns.lmplot(x='GrLivArea', y='SalePrice', hue='BldgType', 

           aspect=2,

           fit_reg=False,

           data=boston_train);
# remove the outliner in columns GrLiveArea

boston_train.drop(boston_train[boston_train['SalePrice'] > 500000].index, inplace=True)



# remove the outliner in columns GrLiveArea

boston_train.drop(boston_train[(boston_train['GrLivArea'] > 4000) ].index, inplace=True)

sns.lmplot(x='GrLivArea', y='SalePrice', hue='BldgType', 

           aspect=2,

           fit_reg=False,

           data=boston_train);
# dist plot shows skewness in the target variable 'SalePrice'

fig, ax = plt.subplots(figsize=(10,5))

ax = sns.distplot(boston_train['SalePrice'], kde=True, bins=20);
# Set the default matplotlib figure size:

fig, ax = plt.subplots(figsize=(20, 21))   



# x and y labels.

ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=17, rotation=60)

ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=17, rotation=0)





sns.heatmap(boston_train.corr(), ax = ax, fmt='.1f', annot=True)

plt.title('Correlation of licenses and accidents')



#show the plot and get  

plt.show()
#Find the top 12 Features Most Correlated With Sale Price

cols = boston_train.corr().nlargest(12, 'SalePrice').index
# HeatMap for the 12 fr

plt.subplots(figsize=(20,20))

sns.set(font_scale=1.25)

sns.heatmap(boston_train[cols].corr() ,annot=True)
corr = boston_train.corr().nlargest(12, 'SalePrice')



plt.subplots(figsize=(16, 9))

sns.barplot(x=corr.index, y=corr['SalePrice'])



plt.title('Top 12 Features Most Correlated With Sale Price', fontsize=24)

plt.xlabel('Feature', fontsize=18)

plt.ylabel('Correlation Sale Price', fontsize=18)



plt.xticks(rotation=60)

plt.tight_layout()
boston_train.head(1)
var = 'OverallQual'

plt.subplots(figsize=(15,15))

sns.boxplot(x=boston_train[var], y=boston_train['SalePrice'])
#plot size 

plt.subplots(figsize=(16, 9))



hood_prices=pd.DataFrame(boston_train.groupby('Neighborhood')['SalePrice'].mean())

sns.barplot(x=hood_prices.index, y=hood_prices['SalePrice'],color='gray')





plt.title('Most Sale Neighbourhood', fontsize=20)

plt.xlabel('Neighbourhood', fontsize=15)

plt.ylabel('Sale Price', fontsize=15)

plt.xticks(rotation=60)

plt.tight_layout()

# plt.savefig('figures/Neighborhood_vs_SalePrice_boxplot.png')
plt.subplots(figsize=(16, 9))



boston_train.HouseStyle.value_counts().plot(kind='bar',color='gray');

plt.title('Style of dwelling', fontsize=20)

plt.xlabel('House Style', fontsize=20)

plt.ylabel('Number of Houses', fontsize=20)

plt.xticks(rotation=60,fontsize=20)
plt.subplots(figsize=(16, 9))



hood_prices=pd.DataFrame(boston_train.groupby('HouseStyle')['SalePrice'].mean())

sns.barplot(x=hood_prices.index, y=hood_prices['SalePrice'],color='gray')



plt.title('The Most House Style correlated with Sale Price', fontsize=20)

plt.xlabel('House Style', fontsize=20)

plt.ylabel('Sale Price', fontsize=20)

plt.xticks(rotation=60,fontsize=20)

plt.tight_layout()

# plt.savefig('figures/Neighborhood_vs_SalePrice_boxplot.png')
fig, ax = plt.subplots(figsize=(10,5))

ax.scatter(boston_train['SalePrice'], boston_train['GrLivArea'])

ax.set_xlabel('Buildings Sale Price')

ax.set_ylabel('living area square feet')

plt.show()
# sklearn imports

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.metrics import mean_squared_error, mean_squared_log_error
boston_test.shape, boston_train.shape
#make copy and dumies for train data



train_copy = boston_train

train_copy = pd.get_dummies(train_copy)

train_copy.shape
#make copy and dumies for test data



test_copy = boston_test

test_copy = pd.get_dummies(test_copy)

test_copy.shape
# to show the missing colunms that are not in test_copy 



set(test_copy.columns).symmetric_difference(set(train_copy.columns))
#Fill all the missing colunms with 0 values 

#to get the same number of colunms in both data 

#also will add SalePrice columns in test , will deleted it next

list_not_test = list(set(train_copy.columns) - set(test_copy.columns))



for col in list_not_test:

    test_copy[col] = 0
#check that the both data have the same number of columns 

test_copy.shape, train_copy.shape
#delete the SalePrice and 'Exterior1st_Other','MSSubClass_150' in test_copy 

test_copy.drop(['SalePrice'],axis=1,inplace=True)
test_copy.drop(['Exterior1st_Other'],axis=1,inplace=True)
set(test_copy.columns).symmetric_difference(set(train_copy.columns))
#creating matrices for sklearn:



y = train_copy['SalePrice'] 

X = train_copy.drop('SalePrice',axis=1)#will take all colnums from train exsecpt SalePrice columns

#check after delete SalePrice 

X.shape,test_copy.shape
#split the data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
#Instantiate StandardScaler

ss = StandardScaler()



#fit

ss.fit(X_train)



#transform

X_train_sc = ss.transform(X_train)

X_test_sc = ss.transform(X_test)
from sklearn.model_selection import GridSearchCV

score_calc = 'neg_mean_squared_error'
def get_best_score(grid):

    

    best_score = np.sqrt(-grid.best_score_)

    print(best_score)    

    print(grid.best_params_)

    print(grid.best_estimator_)

    

    return best_score


linreg = LinearRegression()

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

grid_linear = GridSearchCV(linreg, parameters, cv=5, verbose=1 , scoring = score_calc)

grid_linear.fit(X_train, y_train)



sc_linear = get_best_score(grid_linear)
linreg_sc = LinearRegression()

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

grid_linear_sc = GridSearchCV(linreg_sc, parameters, cv=5, verbose=1 , scoring = score_calc)

grid_linear_sc.fit(X_train_sc, y_train)



sc_linear_sc = get_best_score(grid_linear_sc)
linregr_all = LinearRegression()



linregr_all.fit(X, y)

pred_linreg_all = linregr_all.predict(X_test)

pred_linreg_all[pred_linreg_all < 0] = pred_linreg_all.mean()
from sklearn.neighbors import KNeighborsRegressor



param_grid = {'n_neighbors' : [2,3,4,5,6,7,10,15] ,    

              'weights' : ['uniform','distance'] ,

              'algorithm' : ['ball_tree', 'kd_tree', 'brute']}



grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=10, refit=True, verbose=1, scoring = score_calc)

grid_knn.fit(X_train_sc, y_train)



sc_knn = get_best_score(grid_knn)
pred_knn = grid_knn.predict(X_test_sc)


param_grid = { 'max_depth' : [7,8,9,10] , 'max_features' : [11,12,13,14] ,

               'max_leaf_nodes' : [None, 12,15,18,20] ,'min_samples_split' : [20,25,30],

                'presort': [False,True] , 'random_state': [5] }

            

grid_dtree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=8, refit=True, verbose=1, scoring = score_calc)

grid_dtree.fit(X_train_sc, y_train)



sc_dtree = get_best_score(grid_dtree)



pred_dtree = grid_dtree.predict(X_test)
from sklearn.ensemble import RandomForestRegressor



param_grid = {'min_samples_split' : [3,4,5,7,10], 'n_estimators' : [70,100], 'random_state': [5] }

grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=7, refit=True, verbose=1, scoring = score_calc)

grid_rf.fit(X_train_sc, y_train)



sc_rf = get_best_score(grid_rf)

sc_rf
from sklearn.linear_model import Lasso



lasso = Lasso()

parameters = {'alpha':[1e-03,0.01,0.1,0.5,0.8,1], 'normalize':[True,False], 'tol':[1e-06,1e-05,5e-05,1e-04,5e-04,1e-03]}

grid_lasso = GridSearchCV(lasso, parameters, cv=5, verbose=1, scoring = score_calc)

grid_lasso.fit(X_train_sc, y_train)



sc_lasso = get_best_score(grid_lasso)



pred_lasso = grid_lasso.predict(X_test)
from sklearn.linear_model import Ridge



ridge = Ridge()

parameters = {'alpha':[0.001,0.005,0.01,0.1,0.5,1], 'normalize':[True,False], 'tol':[1e-06,5e-06,1e-05,5e-05]}

grid_ridge = GridSearchCV(ridge, parameters, cv=5, verbose=1, scoring = score_calc)

grid_ridge.fit(X_train, y_train)



sc_ridge = get_best_score(grid_ridge)
ridge_sc = Ridge()

parameters = {'alpha':[0.001,0.005,0.01,0.1,0.5,1], 'normalize':[True,False], 'tol':[1e-06,5e-06,1e-05,5e-05]}

grid_ridge_sc = GridSearchCV(ridge_sc, parameters, cv=10, verbose=1, scoring = score_calc)

grid_ridge_sc.fit(X_train_sc, y_train)



sc_ridge_sc = get_best_score(grid_ridge_sc)
# it seems we're not getting any better results than these, lets export the csv and get our score on kaggle

pred_rf = grid_ridge.predict(test_copy)

scoring_df = pd.concat((pd.Series(boston_test.Id, name='Id'), pd.Series(pred_rf, name='SalePrice')), axis=1)

scoring_df.head(10)
scoring_df.to_csv("house_price_submission.csv", index=False)
scoring_df.tail(5)