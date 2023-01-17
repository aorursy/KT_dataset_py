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
#importing libraries and data set

import seaborn as sns

import matplotlib.pyplot as plt

dataset=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
#Checking total number records

dataset.shape
#Checking all columns

dataset.columns
dataset.dtypes.unique()
dataset.head()
#Checking columns data types

#String data type

len(dataset.select_dtypes(include=['O']).columns)
#Integer data type

len(dataset.select_dtypes(include=['int64']).columns)
#Float data type

len(dataset.select_dtypes(include=['float64']).columns)
#Getting Correlation Coefficient of sale price with other numerical data

saleprice_corr=dataset.corr()['SalePrice']

saleprice_corr

#Creating independent variables data frame X

X=dataset[['Neighborhood','OverallQual','YearBuilt','ExterCond','TotalBsmtSF','GrLivArea','SalePrice']]
#Verifying the relation between GrLivArea and SalePrice

plt.scatter(X['GrLivArea'],X['SalePrice'])
#Verifying the relation between TotalBsmtSF and SalePrice

plt.scatter(X['TotalBsmtSF'],X['SalePrice'])
#Verifying the relation between OverallQual and SalePrice

plt.scatter(X['OverallQual'],X['SalePrice'])
#Verifying the relation between YearBuilt and SalePrice

plt.scatter(X['YearBuilt'],X['SalePrice'])
#Creating pair plot along with categorical variable 'ExterCond' to get relation with Sale Price and other variables

sns.pairplot(X,hue='ExterCond')
#Creating pair plot along with categorical variable 'Neighborhood' to get relation with Sale Price and other variables

sns.pairplot(X,hue='Neighborhood')
#Box plot between 'OverallQuality' and 'Sales Price'

sns.boxplot(x=X['OverallQual'],y=X['SalePrice'],palette='rainbow')
#Box plot between 'ExterCond' and 'Sales Price'

sns.boxplot(x=X['ExterCond'],y=X['SalePrice'],palette='rainbow')
#Box plot between 'Neighborhood' and 'Sales Price'

plt.figure(figsize=(20,10))

sns.boxplot(y=X['Neighborhood'],x=X['SalePrice'],palette='rainbow')
#Box plot between 'Year built' and 'Sales Price' to check sales price across years

plt.figure(figsize=(20, 10))

sns.boxplot(x=X['YearBuilt'],y=X['SalePrice'],palette='rainbow')
#Adding new independent Variables in X

X=dataset[['FullBath','OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','GarageCars','SalePrice']]
#Verifying the relation between FullBath and SalePrice

plt.scatter(X['FullBath'],X['SalePrice'])
#Verifying the relation between FullBath and SalePrice

plt.scatter(X['GarageCars'],X['SalePrice'])
#Create pairplot with new list

sns.pairplot(X)
#Checking missing values 

total_missing_values_X=X.isnull().sum().sort_values(ascending=False)

total_missing_values_X
#Checking first two values in 'GirLivArea' for outliers

X.sort_values(by='GrLivArea',ascending=False)[:2]
#Drop outliers rows from the data set

X=X.drop(1298)

X=X.drop(523)
#Get index for records with GarageCars as 4

indexNames=X[X['GarageCars'] == 4].index
#Drop the records for GarageCars as 4

X=X.drop(indexNames)
#Create pairplot with new list

sns.pairplot(X)
#Split independent and dependent variables

X=dataset[['FullBath','OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','GarageCars']]

y=dataset[['SalePrice']]

# Fitting Multiple linear regression to the data set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X,y)
#Predicting the train set results

y_pred=regressor.predict(X)
#Converting y from series to array , to generate a graph for comparision with y_pred

y=y.values
#Rounding off the y_pred 

y_pred=y_pred.round()
#Converting 2 dimensional y and y_pred array into single dimension 

y=y.ravel()

y_pred=y_pred.ravel()

y_pred
#Creating data frame for y and y_pred to create line plot

df=pd.DataFrame({"y":y,"y_pred":y_pred})

sns.lineplot(data=df)
#Removing 'FullBath' from list of independent variables

#X=dataset[['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']]
#Creating new regressor object and fitting the model

regressor_new=LinearRegression()

regressor_new.fit(X,y)

y_pred_new=regressor_new.predict(X)
#Rounding off the y_pred_new

y_pred_new=y_pred_new.round()
#Converting 2 dimensional y and y_pred array into single dimension 

y_pred_new=y_pred_new.ravel()
#Creating data frame for y ,y_pred,y_pred_new to create line plot

df=pd.DataFrame({"y":y,"y_pred":y_pred,"y_pred_new":y_pred_new})

sns.lineplot(data=df)
#Get test data 

dataset_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#Create X_test and fetching id in different frame

X_test=dataset_test[['FullBath','OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','GarageCars']]

y_test_id=dataset_test[['Id']]

X_test.head()
#Checking missing value in test data set

total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)

total_missing_values_X_test
#Checking the missing Garage Cars record

X_test[X_test['GarageCars'].isnull()]
 #Checking the missing Total Bsmt SF record

X_test[X_test['TotalBsmtSF'].isnull()]
#Updating Garage Cars to 2 at missing value index

X_test.at[1116,'GarageCars'] = 2
#Verifying the missing value in Garage Cars

X_test[X_test['GarageCars'].isnull()]
#Fetching 'TotalBsmtSF' information

X_test['TotalBsmtSF'].describe()
#Updating the missing value to mean value

X_test.at[660,'TotalBsmtSF'] = 1046.12
#Verifying the missing value in TotalBsmtSF

X_test[X_test['TotalBsmtSF'].isnull()]
#Checking missing value in test data set again

total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)

total_missing_values_X_test
#Visualize test data

sns.pairplot(X_test)
X_test.sort_values(by='GrLivArea',ascending=False)[:2]
X_test.sort_values(by='TotalBsmtSF',ascending=False)[:2]
#We can drop the outliers but our submission csv needs 1459 records 

#X_test=X_test.drop(1089)
#Creating predictions based on X_test

y_test_pred=regressor_new.predict(X_test)
#Converting 2 dimensional y_test_pred into single dimension 

y_test_pred=y_test_pred.ravel()

#Rounding off the values

y_test_pred=y_test_pred.round()
#Converting Id into array

y_test_id=y_test_id.values
#Converting 2 dimensional y_test_id into single dimension 

y_test_id=y_test_id.ravel()
#Creating Submission dataframe from id and predecited Sale price

submission_df=pd.DataFrame({"Id":y_test_id,"SalePrice":y_test_pred})

#Setting index as Id Column

submission_df.set_index("Id")
#Converting into CSV file for submission

submission_df.to_csv("submission_1.csv",index=False)
#Apply K-fold in current model to check model accuracy

from sklearn.model_selection import cross_val_score

accuracies_linreg_model = cross_val_score(estimator = regressor, X = X, y = y, cv = 10)

#Checking accuracies for 10 fold in linear regression model

accuracies_linreg_model
#Checking Mean and Standard Deviation between accuracies

accuracies_linreg_model.mean()

accuracies_linreg_model.std()
#Creating new Regressor model for Random forest regression

from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

rf_regressor.fit(X, y)
# Predicting from test set with this new model

y_test_rf_pred=rf_regressor.predict(X_test)
#Converting 2 dimensional y_test_pred into single dimension 

y_test_rf_pred=y_test_rf_pred.ravel()
#Rounding off the values

y_test_rf_pred=y_test_rf_pred.round()
#Creating Submission dataframe from id and predecited Sale price

submission_rf_df=pd.DataFrame({"Id":y_test_id,"SalePrice":y_test_rf_pred})

#Setting index as Id Column

submission_rf_df.set_index("Id")
#Converting into CSV file for submission

submission_rf_df.to_csv("submission_2.csv",index=False)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X,y)
#Checking Best params

rf_random.best_params_
# Predicting from test set with this new model

y_test_rf_random_pred=rf_random.predict(X_test)
#Converting 2 dimensional y_test_pred into single dimension 

y_test_rf_random_pred=y_test_rf_random_pred.ravel()

#Rounding off the values

y_test_rf_random_pred=y_test_rf_random_pred.round()
#Creating Submission dataframe from id and predecited Sale price

submission_rf_random_df=pd.DataFrame({"Id":y_test_id,"SalePrice":y_test_rf_random_pred})

#Setting index as Id Column

submission_rf_random_df.set_index("Id")
#Converting into CSV file for submission

submission_rf_random_df.to_csv("submission_3.csv",index=False)
#importing required library and creating XGboost Regressor model

from xgboost import XGBRegressor

xgboost_regressor=XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)

xgboost_regressor.fit(X,y)
# Predicting from test set with this new model

y_test_xgb_pred=xgboost_regressor.predict(X_test)
#Converting 2 dimensional y_test_pred into single dimension 

y_test_xgb_pred=y_test_xgb_pred.ravel()

#Rounding off the values

y_test_xgb_pred=y_test_xgb_pred.round()
#Creating Submission dataframe from id and predecited Sale price

submission_xgb_df=pd.DataFrame({"Id":y_test_id,"SalePrice":y_test_xgb_pred})

#Setting index as Id Column

submission_xgb_df.set_index("Id")
#Converting into CSV file for submission

submission_xgb_df.to_csv("submission_4.csv",index=False)