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

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



# See the first few rows of train.csv

print(train.head())



# Check out the shapes of train.csv and test.csv

print(train.shape)

print(test.shape)
## LotArea



plt.subplots(figsize=(10,7))

plt.scatter(train["LotArea"], train['SalePrice'], s = 8, alpha = 0.5)

plt.xlabel('LotArea')

plt.ylabel('SalePrice')

plt.title("SalePrice vs LotArea")

plt.show()
## GrLivArea



plt.subplots(figsize=(12,9))

plt.scatter(train["GrLivArea"], train['SalePrice'], s = 8, alpha = 0.5)

plt.xlim()

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.title("SalePrice vs GrLivArea")

plt.show()
## Remove the 2 outliers

## Descending order of GrLivArea values

#GrLivArea_threshold = train['GrLivArea'].sort_values(ascending = False).iloc[1]



#train = train.drop(train[(train['GrLivArea'] >= GrLivArea_threshold)].index)



## AFTER removing outliers - GrLivArea



#plt.subplots(figsize=(12,9))

#plt.scatter(train["GrLivArea"], y, s = 8, alpha = 0.5)

#plt.xlim()

#plt.xlabel('GrLivArea')

#plt.ylabel('SalePrice')

#plt.title("SalePrice vs GrLivArea")

#plt.show()
### YearBuilt



plt.subplots(figsize=(12,9))

sns.scatterplot(x = 'YearBuilt', y = 'SalePrice', data = train, s = 100, alpha = 0.7)

plt.xlabel('YearBuilt')

plt.ylabel('SalePrice')

plt.title("SalePrice vs YearBuilt")

plt.show()
### OverallCond

## Categorical column - let's look at the possible values

## According to the data_description.txt, OverallCond ranges from 1 to 10 - 1 being the worst score, and 10 the best.



train['OverallCond'].value_counts()
### OverallCond

## Categorical column



plt.subplots(figsize=(12,9))

sns.boxplot(x = 'OverallCond', y = 'SalePrice', data = train)

plt.xlabel('OverallCond')

plt.ylabel('SalePrice')

plt.title("SalePrice vs OverallCond")

plt.show()
### OverallQual

## Categorical column - let's look at the possible values

## According to the data_description.txt, OverallQual ranges from 1 to 10 - 1 being the worst score, and 10 the best.



train['OverallQual'].value_counts()
### OverallQual

## Categorical column



plt.subplots(figsize=(12,9))

sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = train)

plt.xlabel('OverallQual')

plt.ylabel('SalePrice')

plt.title("SalePrice vs OverallQual")

plt.show()
### HouseStyle

## Categorical column - let's look at the possible values

## According to the data_description.txt, HouseStyle is the style of dwelling



train['HouseStyle'].value_counts()
### HouseStyle

## Categorical column



plt.subplots(figsize=(12,9))

sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = train)

plt.xlabel('HouseStyle')

plt.ylabel('SalePrice')

plt.title("SalePrice vs HouseStyle")

plt.show()
### BldgType

## Categorical column



plt.subplots(figsize=(12,9))

sns.boxplot(x = 'BldgType', y = 'SalePrice', data = train)

plt.xlabel('BldgType')

plt.ylabel('SalePrice')

plt.title("SalePrice vs BldgType")

plt.show()
### Neighborhood

## Categorical column



plt.subplots(figsize=(25,9))

sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = train)

plt.xlabel('Neighborhood')

plt.ylabel('SalePrice')

plt.title("SalePrice vs Neighborhood")

plt.show()
### MSZoning

## Categorical column



plt.subplots(figsize=(15,9))

sns.boxplot(x = 'MSZoning', y = 'SalePrice', data = train)

plt.xlabel('MSZoning')

plt.ylabel('SalePrice')

plt.title("SalePrice vs MSZoning")

plt.show()
### Summary Statistics



train.info()
X = train.drop(['SalePrice'], axis = 1)

y = train['SalePrice']



comb_df = pd.concat([X, test], axis = 0, sort = False )

comb_df.reset_index(inplace = True)

comb_df.drop(['index'], axis = 1, inplace = True)

comb_df.head()

comb_df.tail()
# Shape of new dataframe (Without labels)

print(comb_df.shape)
## Before Standardization



sns.distplot(y , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(y)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(y, plot=plt)

plt.show()
# Log-transformation of SalePrice - y

# Use the numpy fuction log1p which  applies log(1+x) to all elements of the column

y = np.log1p(y)



# Check the new distribution 

sns.distplot(y , fit = norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(y)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(y, plot=plt)

plt.show()
### Since the first column 'Id' is not important and useful in predicting the SalePrice, we will drop it from the dataset

comb_df = comb_df.drop(['Id'], axis = 1)

comb_df.head()
# nan_columns = list of columns with missing values

nan_columns = []



for i in range(0, comb_df.shape[1]):

    # Total number of non-null values in a column must be 2919

    # Count the sum of the series from the value_counts() of each column in comb_df

    col_value_counts = comb_df.iloc[:, i].value_counts()

    value_counts_sum = sum(col_value_counts)

    

    if value_counts_sum != comb_df.shape[0]:

        # Append column name to nan_columns

        nan_columns.append(comb_df.columns[i])
# Count the number of such columns 

print('Number of columns with missing values is', len(nan_columns))
temp = comb_df.isnull().sum().sort_values(ascending = False)

print(temp[temp > 0 ])

print("Number of columns with missing values is", len(temp[temp > 0]))



# Focus on columns with missing values

temp = temp[temp > 0]
# For columns with less than 5 missing values

temp = temp[(temp > 0) & (temp < 5)]



# Index of temp_col - Column names

temp_index = temp.index



# Check out datatype of these columns first:

comb_df.loc[:, temp_index].info()
for col in comb_df.loc[:, temp_index].select_dtypes(include=np.number):

    comb_df[col] = comb_df[col].fillna(comb_df[col].median())



# Check that the missing values of the numeric columns have been filled up

temp = comb_df.isnull().sum().sort_values(ascending = False)

temp = temp[(temp > 0) & (temp < 5)]

print(temp)



# Create a list containing the names of the columns:

col_name_nan = list(temp.index)

print(col_name_nan)
def replace_nan(df, col):

    # The arguments are df and col, where

    # df is the dataframe, and col (list) are the columns that we want to replace missing values from.

    

    for c in col:

        # Calculate distribution of values in column 'c'

        valueCounts_c = df[c].value_counts()

        

        # Obtain the value that appears the most

        majorityVal_c = valueCounts_c.index[0]

        

        # Replace the missing value with the majority value

        df[c].fillna(majorityVal_c, inplace = True)

        

    return df
comb_df = replace_nan(comb_df, col_name_nan)



# Check that the missing values of the non-numeric columns have been filled up (Columns have less than 5 missing values)

temp = comb_df.isnull().sum().sort_values(ascending = False)

temp = temp[(temp > 0)]

print(temp)



print("Number of columns with missing values is:", len(temp[temp > 0]))
comb_df.Alley.fillna('NA', inplace = True)

comb_df.BsmtQual.fillna('NA', inplace = True)

comb_df.BsmtCond.fillna('NA', inplace = True)

comb_df.BsmtExposure.fillna('NA', inplace = True)

comb_df.BsmtFinType1.fillna('NA', inplace = True)

comb_df.BsmtFinType2.fillna('NA', inplace = True)

comb_df.FireplaceQu.fillna('NA', inplace = True)

comb_df.GarageType.fillna('NA', inplace = True)

comb_df.GarageFinish.fillna('NA', inplace = True)

comb_df.GarageQual.fillna('NA', inplace = True)

comb_df.GarageCond.fillna('NA', inplace = True)

comb_df.PoolQC.fillna('NA', inplace = True)

comb_df.Fence.fillna('NA', inplace = True)

comb_df.MiscFeature.fillna('NA', inplace = True)



comb_df.MasVnrType.fillna('None', inplace = True)
temp = comb_df.isnull().sum().sort_values(ascending = False)

temp = temp[temp > 0]



# Index of temp_col - Column names

temp_index = temp.index



# Check out datatype of these columns first:

comb_df.loc[:, temp_index].info()
# Replace missing values with zero for numeric columns

comb_df.LotFrontage.fillna(0, inplace = True)

comb_df.GarageYrBlt.fillna(0, inplace = True)

comb_df.MasVnrArea.fillna(0, inplace = True)



# Check that the missing values of the numeric columns have been filled up

temp = comb_df.isnull().sum().sort_values(ascending = False)

temp = temp[(temp > 0)]

print(temp)



# Create a list containing the names of the columns:

col_name_nan = list(temp.index)

print(col_name_nan)
# def replace_nan_random(df, col):

#     # df - dataframe

#     # col - affected columns

    

#     for c in col:

#         ### print('For column:', c)

#         unique_values = list(df[c].value_counts().index)

#         df[c] = df[c].astype('category')

#         d = dict(enumerate(df[c].cat.categories))

#         ### print(d)

#         temp_df_nan = df[c].cat.codes.to_frame()   # with nan entries

#         temp_df = temp_df_nan[(temp_df_nan > -1)]

#         stats = temp_df.describe()

        

#         # New column

#         new_col = []

#         for i in range(0, len(temp_df_nan)):

#             if (temp_df_nan.iloc[i][0] == -1): # if True --> entry is -1 (-1 instead of Nan)

#                 # replace with a normal distribution based on values of the temporary 'temp_df'

#                 random_replacement = np.int(np.random.normal(loc = stats.loc['mean'], scale = stats.loc['std']))

                

#                 while random_replacement < 0 or random_replacement > len(d) - 1: # if negative, continue while loop to generate new random value

#                     random_replacement = np.int(np.random.normal(loc = stats.loc['mean'], scale = stats.loc['std']))

#                 new_col.append(random_replacement)

#             else:

#                 new_col.append(temp_df_nan.iloc[i][0])



#         # Map back to categorical values

#         new_col = list(map(d.get, new_col))

#         ### print(new_col)

#         df[c] = new_col

        

#     return df
# comb_df = replace_nan_random(comb_df, col_name_nan)



# # Check that the missing values of all of the columns have been filled up

# temp = comb_df.isnull().sum().sort_values(ascending = False)

# temp = temp[(temp > 0)]

# print(temp)
comb_df.head()
comb_df.MSSubClass = comb_df.MSSubClass.map({20 : 'onestorey', 

                                             30 : 'onestorey', 

                                             40 : 'onestorey', 

                                             45 : 'onehalfstorey', 

                                             50 : 'onehalfstorey', 

                                             60 : 'twostorey', 

                                             70 : 'twostorey', 

                                             75 : 'twohalfstorey', 

                                             80 : 'split', 

                                             85 : 'split',

                                             90 : 'duplex',

                                             120 : 'onestorey', 

                                             150 : 'onehalfstorey',

                                            160 : 'twostorey',

                                            180 : 'split',

                                            190 : 'duplex'})



### Check the updated value_counts()

comb_df.MSSubClass.value_counts()
## Create new feature - AgeSold

comb_df['AgeSold'] = comb_df.YrSold - comb_df.YearBuilt
## Visualize the ages of houses when they were sold - allows us to decide how many bins to group these values

sns.distplot(comb_df['AgeSold'], bins = 50)
### Split the AgeSold into 4 categories:

### 0 <= AgeSold <= 7

### 7 < AgeSold <= 35

### 35 < AgeSold <= 54

### AgeSold > 54



### Sort the 'Age' into 6 different categories

dataset = [comb_df]

for row in dataset:

    row['AgeSold'] = row['AgeSold'].astype(int)

    row.loc[(row['AgeSold'] <= 7), 'AgeSold'] = 0

    row.loc[(row['AgeSold'] > 7) & (row['AgeSold'] <= 35), 'AgeSold'] = 1

    row.loc[(row['AgeSold'] > 35) & (row['AgeSold'] <= 54), 'AgeSold'] = 2

    row.loc[ row['AgeSold'] > 54, 'AgeSold'] = 3

    

### View value_counts() of AgeSold column

comb_df.AgeSold.value_counts()



### View distribution of AgeSold again

## If needed, re-split the AgeSold into appropriate bins, so that the distribution of data in each category is approximately the same

sns.distplot(comb_df['AgeSold'])



## Delete YrSold, MoSold, and YearBuilt from the dataset - Not needed anymore

comb_df.drop(['YrSold', 'MoSold', 'YearBuilt'], axis = 1, inplace = True)
### Create a map to map the neighborhoods to the locations in Ames City

neighborhood_map = {'Blmngtn':'N', 'Blueste':'W', 'BrDale':'N', 'BrkSide':'C', 'ClearCr':'N', 'CollgCr':'C', 'Crawfor':'E', 'Edwards':'W', 'Gilbert':'N',

                   'IODTRR':'E', 'MeadowV':'S', 'Mitchel':'S', 'Names':'N', 'NoRidge':'N', 'NPkVill':'N', 'NridgHt':'N', 'NWAmes':'C', 'OldTown':'E',

                   'SWISU':'S', 'Sawyer':'W', 'SawyerW':'W', 'Somerst':'N', 'StoneBr':'N', 'Timber':'S', 'Veenker':'N'}



### Map the values

comb_df.Neighborhood = comb_df.Neighborhood.map(neighborhood_map)



### Check out the mapped values in the column

comb_df.Neighborhood.value_counts()
# Get column names of numeric columns

numeric_feats = comb_df.dtypes[comb_df.dtypes != "object"].index



# Check the skewness of all numerical features

skewed_feats = comb_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

print(skewness.head(10))
## Box Cox Transformation



skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lambd = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    comb_df[feat] = boxcox1p(comb_df[feat], lambd)



# AFTER: check the skewness of all numerical features

skewed_feats = comb_df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

print(skewness.head(10))
### Columns with categorical values

comb_df.info()
# Categorical columns

categorical_cols = comb_df.select_dtypes('object').columns

print(categorical_cols)

      

# Let's create another function perform Pandas dummy encoding on these categorical columns

def encode_cat(df, cols):

    # df - target dataframe

    # cols - columns that are categorical

      

    ## Steps to dummy encode:

    # 1. create dummy_encode df 

    # 2. concat to main df

    # 3. drop original cabin column

    for c in cols:

        df_dummy_c = pd.get_dummies(df[c], prefix = c)

        df = pd.concat([df, df_dummy_c], axis = 1)

        df.drop([c], axis = 1, inplace = True)

    

    # return dataframe

    return df



# Call the function encode_cat(df, cols) to create dummy columns

comb_df = encode_cat(comb_df, categorical_cols)

print(comb_df.shape)
print(comb_df.head())

print('No. of columns in final dataset is', comb_df.shape[1])
### Split comb_df back into train and test sets

# Recall that the index of the train set ranges from 0 to 1459, while the index of the test set ranges from 1460 onwards.



### Train set ###

train = comb_df.iloc[0:1460,:]

print('Shape of train set is:', train.shape)



### Test set ###

test = comb_df.iloc[1460:, :]

print('Shape of test set is:', test.shape)
# Use train_test_split to create train and test datasets

# We will create 20% test data and 80% train data out of 'train'

# train = features/predictors

# y = labels

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.2, random_state = 123)

#print(X_train.shape)

#print(X_test.shape)

#print(y_train.shape)

#print(y_test.shape)
# We will be importing the following libraries to implement the algorithms to predict housing prices.

from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, BayesianRidge, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn import metrics

from xgboost import XGBRegressor

import lightgbm as lgb



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# We use k-fold cross validation

k_folds = 10



# We need to define a function that will calculate the RMSE between the true SalePrice and predicted SalePrice

def rmse_model(model):

    kf_cv = KFold(k_folds, shuffle = True, random_state = 123).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model, train.values, y, scoring = "neg_mean_squared_error", cv = kf_cv))

    return(rmse)
### We will implement a function to plot the graph for the RMSE for different cross-validated sets:



def eval_rmse(model, model_name, result_table):

    

    import time

    start = time.time()

    

    model_rmse = rmse_model(model)

    

    end = time.time()

    duration = (end - start) / 60

    print('{} took {:.4f} minutes'.format(model, duration))

    

    # Plot the graph of RMSE for different trainings

    plt.plot(model_rmse)

    plt.xlabel('ith Iteration of K-Folds')

    plt.ylabel('RMSE')

    plt.title('RMSE for different iteraiton of K-folds')

    plt.show()

    

    # Print out the Mean (Average) and Standard Deviation of the RMSE values

    rmse_mean = model_rmse.mean()

    rmse_std = model_rmse.std()

    print('The mean RMSE is: {:.5f}'.format(rmse_mean))

    print('The standard deviation of RMSE is: {:.5f}'.format(rmse_std))

    

    # Append the results to a table for consolidation

    new_row = [model_name, rmse_mean, rmse_std]

    result_table.loc[len(result_table)] = new_row

    result_table.sort_values(by = ['Mean_RMSE'])

    print(result_table)

    

    return None
### Initialize the 'Result_Table' first

result_table = pd.DataFrame(columns = ['Model', 'Mean_RMSE', 'Std_RMSE'])
# Linear Regression

# For a start, train a simple linear regression model

lm = LinearRegression()



# Evaluate the Linear Regression Model:

eval_rmse(lm, 'LinearRegression', result_table)
# Lasso Regression

# Define the Lasso Regression model using make_pipeline to include RobustScaler()

# Another way of implementing:

# lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

# score = rmse_model(lasso)



lasso = make_pipeline(RobustScaler(), Lasso(max_iter = 1e7, alpha = 0.001, random_state = 123))



# Evaluate the Lasso Regression

eval_rmse(lasso, 'Lasso', result_table)
# KernelRidge

# Define the model:

krr = KernelRidge(alpha = 0.6, kernel='polynomial', degree = 2, coef0 = 2.5)



# Evaluate the model

eval_rmse(krr, 'KernelRidge', result_table)
# Gradient Boosting

# Define the model:

gradboost = GradientBoostingRegressor(n_estimators = 3000,

                                      learning_rate = 0.001,

                                      max_depth = 4,

                                      max_features = 'sqrt',

                                      # min_samples_leaf = 15,

                                      # min_samples_split = 10,

                                      loss = 'huber',

                                      random_state = 123,

                                      criterion = 'friedman_mse')



# Evaluate the model:

eval_rmse(gradboost, 'Gradient Boosting', result_table)
# Define the model

elasticnet = make_pipeline(RobustScaler(), ElasticNet(max_iter=1e7, alpha = 0.0005, l1_ratio= 0.9, random_state = 123))                                



# Evaluate the model

eval_rmse(elasticnet, 'Elastic Net', result_table)
# XGBoost

# Define the model

xgb = XGBRegressor(colsample_bytree=0.2,

                       gamma=0.0,

                       learning_rate=0.01,

                       max_depth=5, 

                       min_child_weight=1.5,

                       n_estimators=4000,

                       reg_alpha=0.9,

                       reg_lambda=0.6,

                       subsample=0.8,

                       verbosity=0,

                       random_state = 7,

                       objective='reg:squarederror',

                  n_jobs = -1)





# Evaluate the model

eval_rmse(xgb, 'XGBoost', result_table)
# LightGBM

# Define the model

lightGBM = lgb.LGBMRegressor(objective='regression',

                             num_leaves=10,

                             learning_rate=0.05,

                             n_estimators=1000,

                             max_bin = 55,

                             bagging_fraction = 0.8,

                             bagging_freq = 5,

                             feature_fraction = 0.2319,

                             feature_fraction_seed=9,

                             bagging_seed=9,

                             min_data_in_leaf =10,

                             min_sum_hessian_in_leaf = 11)



# Evaluate the model

eval_rmse(lightGBM, 'LightGBM', result_table)
from sklearn.feature_selection import SelectFromModel

from xgboost import plot_importance

from numpy import sort



# Define function to calculate RMSE:

def model_rmse(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



# Use train_test_split to split train into X_train, Y_train, X_test, Y_test

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.2, random_state = 123)



# Define XGB

xgb = XGBRegressor(colsample_bytree=0.2,

                       gamma=0.0,

                       learning_rate=0.001,

                       max_depth=5, 

                       min_child_weight=1.5,

                       n_estimators=4000,

                       reg_alpha=0.9,

                       reg_lambda=0.6,

                       subsample=0.5,

                       verbosity=0,

                       random_state = 123,

                       objective='reg:squarederror',

                  n_jobs = -1)



# fit XGB on all training data - X_train

xgb.fit(X_train, y_train)



# Plot features according to their importance

# Only the top 50 most important features will be plotted due to space constraints

fig, ax = plt.subplots(figsize=(20, 15))

plot_importance(xgb, max_num_features = 50, height = 0.8, ax = ax)

plt.show()
# Fit xgb using each importance as a threshold

thresholds = sort(xgb.feature_importances_)



# Do a quick plot to see the feature importance

plt.subplots(figsize=(10, 5))

plt.scatter(x = range(0, len(thresholds)), y = thresholds, s= 1)

plt.xlabel('n-th feature')

plt.ylabel('Feature_Importance')

plt.show()
# Store the threshold, no. of features, and corresponding RMSE for purpose of visualisation

rmse_feat_importance = pd.DataFrame(columns = ['thresholds', 'no_features', 'threshold_rmse'])



import time

start = time.time()



# For thresh values in interval of 5 units:

for i in range(0, len(thresholds)):

    if i % 5 == 0: # multiples of 5

        print('Current index is:', i)

        

        thresh = thresholds[i]

        # For thresh values in interval of 5 units:

        # select features using threshold

        selection = SelectFromModel(xgb, threshold = thresh, prefit = True)

        select_X_train = selection.transform(X_train)

            

        # define model

        selection_model = XGBRegressor(colsample_bytree=0.2,

                       gamma=0.0,

                       learning_rate=0.01,

                       max_depth=5, 

                       min_child_weight=1.5,

                       n_estimators=4000,

                       reg_alpha=0.9,

                       reg_lambda=0.6,

                       subsample=0.8,

                       verbosity=0,

                       random_state = 7,

                       objective='reg:squarederror',

                  n_jobs = -1)

        

        # train model

        selection_model.fit(select_X_train, y_train)



        # eval model

        select_X_test = selection.transform(X_test)

        y_pred = selection_model.predict(select_X_test)

        selection_model_rmse = model_rmse(y_test, y_pred)

        print("Thresh = {:.7f}, n = {}, RMSE = {:.5f}".format(thresh, select_X_train.shape[1], selection_model_rmse))



        # Append the results to a rmse_feat_importance for consolidation            

        new_entry = [thresh, select_X_train.shape[1], selection_model_rmse]

        rmse_feat_importance.loc[len(rmse_feat_importance)] = new_entry

    else:

        continue

                

end = time.time()

print('Time taken to run:', (end-start)/60)

# Show final 'rmse_feat_importance' table

print(rmse_feat_importance)
# Plot a graph to see the performance of XGB for different number of features

plt.subplots(figsize=(15, 10))

plt.scatter(x = rmse_feat_importance['no_features'], y = rmse_feat_importance['threshold_rmse'], s = 5)

plt.xlabel('No. of Features in XGB')

plt.ylabel('RMSE - Performance of XGB')

plt.show()
# From 35 onwards, we can pick the number of features that corresponds to the lower RMSE

row_min_rmse = rmse_feat_importance[rmse_feat_importance['threshold_rmse'] == rmse_feat_importance['threshold_rmse'].min()]

print(row_min_rmse)



# Number of features for min rmse

no_features_min_rmse = row_min_rmse['no_features'].values[0]

print(no_features_min_rmse)



# Corresponding threshold

threshold_min_rmse = row_min_rmse['thresholds'].values[0]

print(threshold_min_rmse)
### Use KFolds CV to retrain with X features



#We use k-fold cross validation

k_folds = 10



### Retrain the XGB model using X features only on FULL TRAINING DATA



# Modify the function for calculating mean RMSE

def rmse_model_feat_impt(model):

    kf_cv = KFold(k_folds, shuffle = True, random_state = 123).get_n_splits(select_train)

    rmse = np.sqrt(-cross_val_score(model, select_train, y, scoring = "neg_mean_squared_error", cv = kf_cv))

    return(rmse)



import time

start = time.time()



selection = SelectFromModel(xgb, threshold = threshold_min_rmse, prefit = True)

select_train = selection.transform(train)

select_test = selection.transform(test)

            

# Define model

selection_model = XGBRegressor(colsample_bytree=0.2,

                       gamma=0.0,

                       learning_rate=0.01,

                       max_depth=5, 

                       min_child_weight=1.5,

                       n_estimators=4000,

                       reg_alpha=0.9,

                       reg_lambda=0.6,

                       subsample=0.8,

                       verbosity=0,

                       random_state = 7,

                       objective='reg:squarederror',

                  n_jobs = -1)



# KFolds CV:

CV_rmse = rmse_model_feat_impt(selection_model)



# Print result

print('Mean RMSE of XGB training using {} features is {:.6f}'.format(no_features_min_rmse, CV_rmse.mean()))



end = time.time()

print('Time taken to complete {:.2f} mins'.format((end-start)/60))
# Define a function to calculate RMSE

def rmse(y_true, y_pred):

    return np.sqrt(np.mean((y_true-y_pred)**2))



# Define a function to calculate negative RMSE (as a score)

def nrmse(y_true, y_pred):

    return -1.0*rmse(y_true, y_pred)



from sklearn.metrics import make_scorer 

neg_rmse = make_scorer(nrmse)



estimator = XGBRegressor(colsample_bytree=0.2,

                       gamma=0.0,

                       learning_rate=0.01,

                       max_depth=5, 

                       min_child_weight=1.5,

                       n_estimators=4000,

                       reg_alpha=0.9,

                       reg_lambda=0.6,

                       subsample=0.8,

                       verbosity=0,

                       random_state = 7,

                       objective='reg:squarederror',

                  n_jobs = -1)



from sklearn.feature_selection import RFECV

import time

start = time.time()

selector = RFECV(estimator, cv = 5, n_jobs = -1, scoring = neg_rmse)

selector = selector.fit(X_train, y_train)

end = time.time()

print("Time taken for RFECV to complete:{} mins".format((end-start)/60))

print("The number of selected features is: {}".format(selector.n_features_))



features_kept = X_train.columns.values[selector.support_] 



# Select selected features from X_train and X_test (This is for retraining the XGB using selected features and calculating the RMSE based on train_test_split datasets - this step might not be really needed)

X_train = selector.transform(X_train)  

X_test = selector.transform(X_test)



# Select selected features from train and test to be used for Ensemble Learning

test = selector.transform(test)

train = selector.transform(train)



# transform the labels to a numpy array so later we can feed it to a neural network

y_train = y_train.values 

y_test = y_test.values
## Retrain XGB using selected features from RFECV



# Modify the function for calculating mean RMSE

def rmse_model_feat_impt(model):

    kf_cv = KFold(k_folds, shuffle = True, random_state = 123).get_n_splits(X_train)

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring = "neg_mean_squared_error", cv = kf_cv))

    return(rmse)



import time

start = time.time()

           

# Define XGB model again

RFECV_xgb_model = XGBRegressor(colsample_bytree=0.2,

                       gamma=0.0,

                       learning_rate=0.01,

                       max_depth=5, 

                       min_child_weight=1.5,

                       n_estimators=4000,

                       reg_alpha=0.9,

                       reg_lambda=0.6,

                       subsample=0.8,

                       verbosity=0,

                       random_state = 7,

                       objective='reg:squarederror',

                  n_jobs = -1)



# KFolds CV:

RFECV_CV_rmse = rmse_model_feat_impt(RFECV_xgb_model)



# Print result

print('Mean RMSE of XGB training using {} features is {:.6f}'.format(selector.n_features_, RFECV_CV_rmse.mean()))



end = time.time()

print('Time taken to complete {:.2f} mins'.format((end-start)/60))
## Evaluate the RFECV_xgb_model on the test dataset from train_test_split



# Fit on X_train, y_train

RFECV_xgb_model.fit(X_train, y_train)



# Predict with X_test

RFECV_xgb_pred = RFECV_xgb_model.predict(X_test)



# Evaluate RMSE score

RFECV_xgb_rmse = rmse(y_test, RFECV_xgb_pred)

print("The RMSE for REFCV_xgb_model is: {:.5f}".format(RFECV_xgb_rmse))
# ### Perform ensemble learning by using X features selected from Method 1: SelectFromModel

# print('shape of full train set (features) after feature selection', select_train.shape)

# print('shape of full test set (features) after feature selection', select_test.shape)

# print('shape of y_train (labels)', y.shape)



### Perform Ensemble Learning by using X features selected from Method 2: REFCV

print("Shape of full train set (features) after Method 2:", train.shape)

print("Shape of full test set (features) after Method 2:", test.shape)

print('shape of y_train (training labels)', y.shape)
### Define StackingCVRegressor model



from mlxtend.regressor import StackingCVRegressor



stackingreg = StackingCVRegressor(

    regressors=(krr, elasticnet, lightGBM),

    meta_regressor= xgb,

    use_features_in_secondary=True,

    random_state = 123,

    n_jobs = -1

)
import time

start = time.time()



# Begin fitting

stack_model = stackingreg.fit(np.array(train), np.array(y))



# krr_model = krr.fit(train, y)

# elasticnet_model = elasticnet.fit(train, y)

# lightGBM_model = lightGBM.fit(train, y)

# xgb_model = xgb.fit(train, y)



end = time.time()

print('The fitting is completed. Time taken is: {:.2f} minutes'.format((end-start)/60))
### Using stack_gen_model to predict using train set

stack_train_pred = stack_model.predict(train)



# Calculate RMSE on stacked models:

stack_train_rmse = model_rmse(y, stack_train_pred)

print('RMSE of stacked models is: {:.5f}'.format(stack_train_rmse))
## Define GridSearchCV for the stacked models



start = time.time()



np.random.seed(123)



grid_stacked_models = GridSearchCV(

    estimator = stackingreg, 

    param_grid = {

        #'kernelridge__alpha': [0.1, 0.5, 1.0],

        #'pipeline-2__elasticnet__alpha': [0.001, 0.01, 0.1],

        #'pipeline-1__lasso__alpha': [0.001, 0.01, 0.1],

        'lgbmregressor__n_estimators': [500, 3000],

        'meta_regressor__n_estimators': [500, 3000]

    }, 

    cv=3,

    refit=True

)



grid_stacked_models.fit(np.array(train), y)



end = time.time()

print('Time taken to run the GridSearchCV for the Stacked Models Ensemble is: {:.2f} minutes'.format((end-start)/60))



print("Best: %f using %s" % (grid_stacked_models.best_score_, grid_stacked_models.best_params_))
### Using stack_gen_model to predict using train set

stack_tuned_pred = grid_stacked_models.predict(train)



# Calculate RMSE on stacked models:

stack_tuned_rmse = model_rmse(y, stack_tuned_pred)

print('RMSE of stacked models is: {:.5f}'.format(stack_tuned_rmse))
# Compute final predictions

final_predictions = np.expm1(grid_stacked_models.predict(test))

    

# Store final_predictions in a csv file

saleprice_submission = pd.DataFrame(

    {'Id': range(1461, 2920),

    'SalePrice': final_predictions}

)

    

print('Submission:')

print(saleprice_submission.head())

    

# Save to CSV file

saleprice_submission.to_csv('submission.csv', index = False)
# import time

# start = time.time()

# grid_params_1 = {'learning_rate':[0.001, 0.01], 'n_estimators':[500, 400], 'max_depth': [2, 4, 6]}



# GS_gradboost_1 = GridSearchCV(

#     estimator = GradientBoostingRegressor(min_samples_split = 2, min_samples_leaf = 2, subsample = 1, max_features = 'auto', random_state = 123, verbose = True, loss = 'huber', criterion = 'friedman_mse'),

#     param_grid = grid_params_1,

#     scoring='neg_mean_squared_error',

#     n_jobs=-1,

#     iid=False,

#     cv=5,

#     verbose = 1

#     )



# #GS_gradboost_1.fit(train,y)
# #GS_gradboost_1.grid_scores_,

# bestparam = GS_gradboost_1.best_params_

# bestscore = GS_gradboost_1.best_score_



# print('bestscore is:', bestscore)

# print('bestparam is:', bestparam)

# end = time.time()

# print('Time taken to run is:', end - start)
# ### Retrain model again with best parameters from GridSearchCV



# final_gradboost = GradientBoostingRegressor(

#     learning_rate = bestparam['learning_rate'],

#     max_depth = bestparam['max_depth'],

#     n_estimators = bestparam['n_estimators'],

#     min_samples_split = 2,

#     min_samples_leaf = 2,

#     subsample = 1,

#     max_features = 'auto',

#     random_state = 123,

#     verbose = True,

#     loss = 'huber'

#     )



# final_gradboost.fit(train, y)
# # Churn out the predictions based on this final model

# final_gb_pred = final_gradboost.predict(train)



# # Calculate RMSE 

# final_gb_rmse = model_rmse(y, final_gb_pred)

# print('RMSE of final GradBoost is: {:.5f}'.format(final_gb_rmse))
# # Decide to use tuned Gradient Boosting or the Stack Regressor

# based on RMSE



# if (stack_tuned_rmse < final_gb_rmse):

#     Use Stack Regressor

#     print('Stack Regressor performs better')

#     Scale back the predicted values

#     final_predictions = np.expm1(grid_stacked_models.predict(test.values))

    

#     Store final_predictions in a csv file

#     saleprice_submission = pd.DataFrame(

#     {'Id': range(1461, 2920),

#     'SalePrice': final_predictions})

    

#     print('Submission:')

#     print(saleprice_submission.head())

    

#     Save to CSV file

#     saleprice_submission.to_csv('submission.csv', index = False)

    

# else:

#     Use tuned Gradient Boosting

#     print('Tuned Gradient Boosting performs better')

#     Scale back the predicted values

#     final_predictions = np.expm1(final_gradboost.predict(test))



#     Store final_prediction in a csv file

#     saleprice_submission = pd.DataFrame(

#        {'Id': range(1461, 2920),

#        'SalePrice': final_predictions})



#     print('Submission:')

#     print(saleprice_submission.head())



#     Save to CSV file

#     saleprice_submission.to_csv('submission.csv', index = False)