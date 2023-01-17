# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import warnings

#import xgboost as xgb

#import lightgbm as lgb

from scipy.stats import skew

from scipy import stats

from scipy.stats.stats import pearsonr

from scipy.stats import norm

from collections import Counter

from sklearn.linear_model import LinearRegression,LassoCV, Ridge, LassoLarsCV,ElasticNetCV

from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler

warnings.filterwarnings('ignore')

sns.set(style='white', context='notebook', palette='deep')

%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load train and Test set

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Check the numbers of samples and features

print("The train data size is : {} ".format(train.shape))

print("The test data size is : {} ".format(test.shape))
# Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']
# Now drop the 'Id' column since it's unnecessary for the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
# Check data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))
train.head()
test.describe()
test.head()
#Analysing the Test , which has Sale Price

train['SalePrice'].describe()
# Avrage price for average home $180921

# Comparing both data frames, we can see that the only difference in features is "Sale Price". 

# This makes sense because we are trying to predict it!
#Checking skewness and kurtosis

sns.distplot(train['SalePrice']);



plt.ylabel('Frequency')

plt.title('SalePrice distribution')

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
#What we see is the target variable SalePrice is not normally distributted (+Skewness)

# This will effect the performance of ML prediction models

# making a log transformation, to make the distribution look a bit better

train['SalePrice_Log'] = np.log(train['SalePrice'])



sns.distplot(train['SalePrice_Log']);

# skewness and kurtosis

print("Skewness: %f" % train['SalePrice_Log'].skew())

print("Kurtosis: %f" % train['SalePrice_Log'].kurt())

# dropping old column

train.drop('SalePrice', axis= 1, inplace=True)
# Checking Categorical Data

train.select_dtypes(include=['object']).columns
# Checking Numerical Data

train.select_dtypes(include=['int64','float64']).columns
#count number of categoricals columns

cat = len(train.select_dtypes(include=['object']).columns)

cat
#count number of numerical dataslen

numerical = len(train.select_dtypes(include=['int64','float64']).columns)

numerical
total_features = cat + numerical

total_features
#Get top 10 numerical important values

#top 10

top = 10

corrmat = train.corr()

f, ax = plt.subplots(figsize=(20, 10))



cols = corrmat.nlargest(top, 'SalePrice_Log')['SalePrice_Log'].index

cm = np.corrcoef(train[cols].values.T)

#sns.heatmap(corrmat, vmax=.8, square=True);

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#check if the data set has any missing values. 

train.columns[train.isnull().any()]
#find total missing

total_missing = train.isnull().sum().sort_values(ascending=False)

total_missing
#Picking top 20 missing

total_missing = total_missing.head(20)

total_missing
#Convert into dataframe

total_missing = total_missing.to_frame()
total_missing.columns = ['count']
total_missing.index.names = ['name']

total_missing['name'] = total_missing.index
total_missing.head()
#Checking there is any null value or not

plt.figure(figsize=(10, 5))

sns.heatmap(train.isnull())
## lots of null values

#Understanding the missing values:

#Many columns has 'NaN' value.

#Closer look tells that they are not missing, instead it means it dosen't exist in that property.

# like for PoolQC - Nan means no swiming pool

#plot Missing values



plt.figure(figsize=(13, 5))

sns.set(style='whitegrid')

sns.barplot(x='name', y='count', data=total_missing)

plt.xticks(rotation = 90)

plt.show()
#get dataframe column list

list(train)
col = ['SalePrice_Log', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

sns.set(style='ticks')

sns.pairplot(train[col], height=3,kind='reg')
#Find most important features - top 20

xcorr = train.corr()

xcorr.sort_values(['SalePrice_Log'], ascending=False, inplace=True)

xcorr.SalePrice_Log.head(20)
#most correlated feature to Sale Price is... Sale Price

#short description of features from the data_description.txt

# OverallQual: Rates the overall material and finish of the house (1 = Very Poor, 10 = Very Excellent)

# GrLivArea: Above grade (ground) living area square feet

# GarageCars: Size of garage in car capacity

# GarageArea: Size of garage in square feet

# TotalBsmtSF: Total square feet of basement area

# 1stFlrSF: First Floor square feet

# FullBath: Full bathrooms above grade

# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

# YearBuilt: Original construction date
#Defination will be used later

def r2(x, y):

    return stats.pearsonr(x, y)[0] ** 2
# plot / check how each relates to Sale Price

# Overall Quality vs Sale Price



sns.boxplot(x=train['OverallQual'], y=train['SalePrice_Log'])
# Living Area vs Sale Price

sns.jointplot(x=train['OverallQual'], y=train['SalePrice_Log'], kind='reg',stat_func=r2)
# Above shows people pay more if the quality is better
# Living Area vs Sale Price

sns.jointplot(x=train['GrLivArea'], y=train['SalePrice_Log'], kind='reg',stat_func=r2)
# more the living area -> more is the price

#there are few points GrLivArea > 4000, makes no propper sense, so removing them



# Removing outliers manually (Two points in the bottom right)

train = train.drop(train[(train['GrLivArea']>4000) 

                         & (train['SalePrice_Log']<300000)].index).reset_index(drop=True)
# Living Area vs Sale Price

sns.jointplot(x=train['GrLivArea'], y=train['SalePrice_Log'], kind='reg',stat_func=r2)
# Garage Area vs Sale Price

sns.boxplot(x=train['GarageCars'], y=train['SalePrice_Log'])

# Issue in data, 4 car garage means less price, not right, will drop it manually
# Removing outliers manually (More than 4-cars, less than $300k)

train = train.drop(train[(train['GarageCars']>3) 

                         & (train['SalePrice_Log']<300000)].index).reset_index(drop=True)
# Garage Area vs Sale Price

sns.boxplot(x=train['GarageCars'], y=train['SalePrice_Log'])
# Total Rooms vs Sale Price

sns.boxplot(x=train['TotRmsAbvGrd'], y=train['SalePrice_Log'])
# Year Build vs Sale Price

plt.figure(figsize=(16, 8))

#sns.boxplot(x=train['YearBuilt'], y=train['SalePrice_Log'])

sns.jointplot(x=train['YearBuilt'], y=train['SalePrice_Log'], kind='reg',stat_func=r2)
# this is not a good comparisism, as it looks like house prices decrease with age, 

# not sure because , other factors might also play like inflation or stock market crashes.
# PoolQC has missing value ratio is 99%+. So, there is fill by None

train['PoolQC'] = train['PoolQC'].fillna('None')



test['PoolQC'] = test['PoolQC'].fillna('None')
#Arround 50% missing values attributes have been fill by None

train['MiscFeature'] = train['MiscFeature'].fillna('None')

train['Alley'] = train['Alley'].fillna('None')

train['Fence'] = train['Fence'].fillna('None')

train['FireplaceQu'] = train['FireplaceQu'].fillna('None')



# # columns where NaN values have meaning e.g. no pool etc. chaanging in Test:



test['MiscFeature'] = test['MiscFeature'].fillna('None')

test['Alley'] = test['Alley'].fillna('None')

test['Fence'] = test['Fence'].fillna('None')

test['FireplaceQu'] = test['FireplaceQu'].fillna('None')

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



test['LotFrontage'] = test.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



#GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    train[col] = train[col].fillna('None')

train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())

#MasVnrArea: Masonry veneer area in square feet

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())



test['GarageYrBlt'] = test['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())

#MasVnrArea: Masonry veneer area in square feet

test['MasVnrArea'] = test['MasVnrArea'].fillna(train['MasVnrArea'].mean())



#GarageYrBlt, GarageArea and GarageCars these are replacing with zero

#for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:

#    train[col] = train[col].fillna(int(0))
#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None

for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):

    train[col] = train[col].fillna('None')

    

    

for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):

    test[col] = test[col].fillna('None')
#MasVnrArea: Masonry veneer area in square feet

#MasVnrArea : replace with mean

trnMean_value=train['MasVnrArea'].mean()

train['MasVnrArea'] = train['MasVnrArea'].fillna(trnMean_value)





tstMean_value=test['MasVnrArea'].mean()

test['MasVnrArea'] = test['MasVnrArea'].fillna(tstMean_value)



#MasVnrType : replace with None

train['MasVnrType'] = train['MasVnrType'].fillna('None')



test['MasVnrType'] = test['MasVnrType'].fillna('None')
#There is put mode value 

train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]



test['Electrical'] = test['Electrical'].fillna(test['Electrical']).mode()[0]
#There is no need of Utilities

train = train.drop(['Utilities'], axis=1)



test = test.drop(['Utilities'], axis=1)
#check if the data set has any missing values. 

train.columns[train.isnull().any()]
total = train.isnull().sum().sort_values(ascending=False)

total
#replace non-important columns with -None



train['GarageFinish'] = train['GarageFinish'].fillna('None')

train['GarageQual'] = train['GarageQual'].fillna('None')

train['GarageCond'] = train['GarageCond'].fillna('None')

train['GarageType'] = train['GarageType'].fillna('None')





test['GarageFinish'] = test['GarageFinish'].fillna('None')

test['GarageQual'] = test['GarageQual'].fillna('None')

test['GarageCond'] = test['GarageCond'].fillna('None')

test['GarageType'] = test['GarageType'].fillna('None')

#check if the data set has any missing values. 

train.columns[train.isnull().any()]
test.columns[test.isnull().any()]
total = test.isnull().sum().sort_values(ascending=False)

total
#replace non-important columns with -None





test['MSZoning'] = test['MSZoning'].fillna('None')

test['Functional'] = test['Functional'].fillna('None')

test['KitchenQual'] = test['KitchenQual'].fillna('None')

test['Exterior2nd'] = test['Exterior2nd'].fillna('None')

test['Exterior1st'] = test['Exterior1st'].fillna('None')

test['SaleType'] = test['SaleType'].fillna('None')



tstMean_value=test['BsmtHalfBath'].mean()

test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(tstMean_value)



tstMean_value=test['BsmtFullBath'].mean()

test['BsmtFullBath'] = test['BsmtFullBath'].fillna(tstMean_value)



tstMean_value=test['BsmtFinSF2'].mean()

test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(tstMean_value)



tstMean_value=test['BsmtFinSF2'].mean()

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(tstMean_value)



tstMean_value=test['GarageCars'].mean()

test['GarageCars'] = test['GarageCars'].fillna(tstMean_value)



tstMean_value=test['TotalBsmtSF'].mean()

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(tstMean_value)



tstMean_value=test['BsmtUnfSF'].mean()

test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(tstMean_value)



tstMean_value=test['GarageArea'].mean()

test['GarageArea'] = test['GarageArea'].fillna(tstMean_value)





test.columns[test.isnull().any()]
total = test.isnull().sum().sort_values(ascending=False)

total
#Checking there is any null value or not

plt.figure(figsize=(10, 5))

sns.heatmap(train.isnull())
# no missing values
#Transformation/Engineering
train.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

types_train = train.dtypes

cat_train = types_train[types_train == object]

categorical_values_train = list(cat_train.index)





types_test = test.dtypes

cat_test = types_test[types_test == object]

categorical_values_test = list(cat_test.index)
for i in categorical_values_train:

    feature_set = set(train[i])

    for j in feature_set:

        feature_list = list(feature_set)

        train.loc[train[i] == j, i] = feature_list.index(j)



for i in categorical_values_test:

    feature_set = set(test[i])

    for j in feature_set:

        feature_list = list(feature_set)

        test.loc[test[i] == j, i] = feature_list.index(j)
#After the conversion of the categorical columns

train.head()
X_train = train.drop(["SalePrice_Log"], axis=1).values

y_train = train["SalePrice_Log"].values

X_test = test.values
from sklearn.model_selection import train_test_split #to create validation data set



X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets
#So original training set has 1460 records, out of that 1160 is now used in X_training, so its 79% of Test data

len(X_training)
rf = RandomForestRegressor()

paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 

                 "max_features" : ["auto", "log2"]}

grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2")

grid_rf.fit(X_training, y_training)



print("Best RandomForestRegressor Model: " + str(grid_rf.best_estimator_))

print("Best Score: " + str(grid_rf.best_score_))
rf = grid_rf.best_estimator_

rf.fit(X_training, y_training)

rf_pred = rf.predict(X_valid)

r2_rf = r2_score(y_valid, rf_pred)

rmse_rf = np.sqrt(mean_squared_error(y_valid, rf_pred))

print("R^2 Score: " + str(r2_rf))

print("RMSE Score: " + str(rmse_rf))
use_logvals = 1 

pred_rf = rf.predict(test)



sub_rf = pd.DataFrame()

sub_rf['Id'] = test_ID



sub_rf['Preditcted RF SalePrice'] = pred_rf 



if use_logvals == 1:

    sub_rf['Preditcted RF SalePrice'] = np.exp(sub_rf['Preditcted RF SalePrice']) 



    

#sub_rf.to_csv('data/rf.csv',index=False)

sub_rf
plt.plot(y_valid[:50]*100000, marker='o')

plt.plot(rf_pred[:50]*100000, marker="x")

plt.xlabel('Number of records')

plt.ylabel('House Price')

plt.title('Actual Sale Price vs Predicted Sale Price')

plt.legend(('Actual Price', 'Predited Price'),

           loc='upper right')
def get_best_score(grid):

    

    best_score = np.sqrt(-grid.best_score_)

    print(best_score)    

    print(grid.best_params_)

    print(grid.best_estimator_)

    

    return best_score

from sklearn.tree import DecisionTreeRegressor

nr_cv = 5

score_calc = 'neg_mean_squared_error'

param_grid = { 'max_depth' : [7,8,9,10] , 'max_features' : [11,12,13,14] ,

               'max_leaf_nodes' : [None, 12,15,18,20] ,'min_samples_split' : [20,25,30],

                'presort': [False,True] , 'random_state': [5] }

            

grid_dtree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)

grid_dtree.fit(X_training, y_training)



sc_dtree = get_best_score(grid_dtree)

sc_dtree

dtree_pred = grid_dtree.predict(X_valid)

r2_rf = r2_score(y_valid, dtree_pred)

r2_rf



dtree_pred = grid_dtree.predict(X_valid)

r2_rf = r2_score(y_valid, dtree_pred)

r2_rf



dtree_pred = grid_dtree.predict(test)



sub_rf['Preditcted DT SalePrice'] = dtree_pred 



if use_logvals == 1:

    sub_rf['Preditcted DT SalePrice'] = np.exp(sub_rf['Preditcted DT SalePrice'])

sub_rf
from sklearn.neighbors import KNeighborsRegressor



param_grid = {'n_neighbors' : [3,4,5,6,7,10,15] ,    

              'weights' : ['uniform','distance'] ,

              'algorithm' : ['ball_tree', 'kd_tree', 'brute']}



grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=nr_cv, refit=True, verbose=1, scoring = score_calc)

grid_knn.fit(X_training, y_training)



sc_knn = get_best_score(grid_knn)

sc_knn

pred_knn = grid_knn.predict(X_valid)

r2_rf = r2_score(y_valid, pred_knn)

r2_rf
pred_knn = grid_knn.predict(test)



sub_rf['Preditcted KNN SalePrice'] = pred_knn 



if use_logvals == 1:

    sub_rf['Preditcted KNN SalePrice'] = np.exp(sub_rf['Preditcted KNN SalePrice'])

sub_rf
#Compare all the models

list_scores = [sc_dtree, rmse_rf, sc_knn]

list_regressors = ['DTr','RF','KNN']

list_scores
fig, ax = plt.subplots()

fig.set_size_inches(10,7)

sns.barplot(x=list_regressors, y=list_scores, ax=ax)

plt.ylabel('RMSE')

plt.xlabel('Algorithm')

plt.title('RMSE Score for each Alogorithm')

plt.grid()

plt.show()
#Since RMSE (Remote mean square error) is least for Random forest , out of the three modeles , 

#here Random Forest regression is performing best
# another simple graph showing that Random Forest is givig the best results, simnce most dots are 

#closer to the actual price
dtree_pred_sample = grid_dtree.predict(X_valid)

kn_pred_sample = grid_knn.predict(X_valid)

df = pd.DataFrame([y_valid[:20],rf_pred[:20],dtree_pred_sample[:20],kn_pred_sample[:20]]).T 

df.columns = ['Actual','RF','DT','KNN']

df['RF_Diff'] = df['RF'] - df['Actual']

df['DT_Diff'] = df['DT'] - df['Actual']

df['KNN_Diff'] = df['KNN'] - df['Actual']

plt.plot(df["Actual"]*10000,df["RF_Diff"], marker='o',linestyle = 'None')

plt.plot(df["Actual"]*10000,df["DT_Diff"], marker='o',linestyle = 'None')

plt.plot(df["Actual"]*10000,df["KNN_Diff"], marker='o',linestyle = 'None')

# plt.plot(rf_pred[:20]*100000, marker="x")

plt.xlabel('Actual Price')

# plt.ylabel('House Price')

# plt.title('Actual Sale Price vs Predicted Sale Price')

plt.ylim(-1,1)

plt.axhline(linewidth=1, color='r')

plt.legend(('RF', 'DT','KNN'),loc='upper right')