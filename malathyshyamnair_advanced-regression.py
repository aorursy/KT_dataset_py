import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV



import os



# hide warnings

import warnings

warnings.filterwarnings('ignore')





# Reading the dataset

train_house = pd.read_csv("../input/house-prices-advanced-regression-techniques/housetrain.csv")

#Showing the data set

train_house.head()
# Summary of data Set

print(train_house.info())
#Checking rows and columns

train_house.shape
# checking Duplicates in dataframe

train_house.duplicated() 
# Checking missing values

train_house.isnull().sum()
# Checking missing values

#train_house.isnull().sum()

def missing(dff):

    print (round((dff.isnull().sum() * 100/ len(dff)),2).sort_values(ascending=False))



missing(train_house)
#Dropping those which have more missing values (more than 50%)

train_house.drop("Alley",axis=1,inplace=True)

train_house.drop("PoolQC",axis=1,inplace=True)

train_house.drop("Fence",axis=1,inplace=True)

train_house.drop("MiscFeature",axis=1,inplace=True)

train_house.drop("GarageYrBlt",axis=1,inplace=True)
# Removing Id 

train_house.drop("Id",axis=1,inplace=True)
train_house.head()
train_house["LotFrontage"]=train_house["LotFrontage"].fillna(0)
#Filling Categorical Value with its mode

train_house["FireplaceQu"]=train_house["FireplaceQu"].fillna(train_house["FireplaceQu"].mode()[0])

train_house["GarageType"]=train_house["GarageType"].fillna(train_house["GarageType"].mode()[0])

train_house["GarageFinish"]=train_house["GarageFinish"].fillna(train_house["GarageFinish"].mode()[0])

train_house["GarageQual"]=train_house["GarageQual"].fillna(train_house["GarageQual"].mode()[0])

train_house["GarageCond"]=train_house["GarageQual"].fillna(train_house["GarageCond"].mode()[0])



#Filling missing value

train_house["BsmtQual"]=train_house["BsmtQual"].fillna(train_house["BsmtQual"].mode()[0])

train_house["MasVnrType"]=train_house["MasVnrType"].fillna(train_house["MasVnrType"].mode()[0])

train_house["MasVnrArea"]=train_house["MasVnrArea"].fillna(train_house["MasVnrArea"].mode()[0])





# Checking missing values

#train_house.isnull().sum()
#Dropping those rows which have high null values

train_house.dropna(inplace=True)
train_house.shape
# Describing the data



train_house.describe()



# Pairplot of all the numeric variables

sns.pairplot(train_house)

plt.show()
plt.figure(figsize=(25, 15))

plt.subplot(3,3,1)

sns.boxplot(x = 'Street', y = 'SalePrice', data = train_house)

plt.subplot(3,3,2)

sns.boxplot(x = 'MSZoning', y = 'SalePrice', data = train_house)

plt.subplot(3,3,3)

sns.boxplot(x = 'LotShape', y = 'SalePrice', data = train_house)

plt.subplot(3,3,4)

sns.boxplot(x = 'LandContour', y = 'SalePrice', data = train_house)

plt.subplot(3,3,5)

sns.boxplot(x = 'Utilities', y = 'SalePrice', data = train_house)

plt.subplot(3,3,6)

sns.boxplot(x = 'LotConfig', y = 'SalePrice', data = train_house)

plt.show()

plt.subplot(3,3,7)

sns.boxplot(x = 'LandSlope', y = 'SalePrice', data = train_house)

plt.subplot(3,3,8)

sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = train_house)

plt.subplot(3,3,9)

sns.boxplot(x = 'Condition1', y = 'SalePrice', data = train_house)

plt.show()
plt.figure(figsize=(25, 15))

plt.subplot(3,3,1)

sns.boxplot(x = 'Condition2', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,2)

sns.boxplot(x = 'BldgType', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,3)

sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,4)

sns.boxplot(x = 'RoofStyle', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,5)

sns.boxplot(x = 'RoofMatl', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,6)

sns.boxplot(x = 'Exterior1st', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,7)

sns.boxplot(x = 'Exterior2nd', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,8)

sns.boxplot(x = 'MasVnrType', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,9)

sns.boxplot(x = 'ExterQual', y = 'SalePrice', data = train_house)

plt.show()
plt.figure(figsize=(25, 15))

plt.subplot(3,3,1)

sns.boxplot(x = 'PavedDrive', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,2)

sns.boxplot(x = 'Foundation', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,3)

sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,4)

sns.boxplot(x = 'BsmtCond', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,5)

sns.boxplot(x = 'BsmtExposure', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,6)

sns.boxplot(x = 'BsmtFinType1', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,7)

sns.boxplot(x = 'BsmtFinType2', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,8)

sns.boxplot(x = 'Heating', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,9)

sns.boxplot(x = 'HeatingQC', y = 'SalePrice', data = train_house)

plt.show()
plt.figure(figsize=(25, 15))

plt.subplot(3,3,1)

sns.boxplot(x = 'CentralAir', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,2)

sns.boxplot(x = 'Electrical', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,3)

sns.boxplot(x = 'KitchenQual', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,4)

sns.boxplot(x = 'Functional', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,5)

sns.boxplot(x = 'FireplaceQu', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,6)

sns.boxplot(x = 'GarageType', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,7)

sns.boxplot(x = 'GarageFinish', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,8)

sns.boxplot(x = 'GarageQual', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,9)

sns.boxplot(x = 'GarageCond', y = 'SalePrice', data = train_house)

plt.show()
plt.figure(figsize=(25, 15))

plt.subplot(3,3,1)

sns.boxplot(x = 'PavedDrive', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,2)

sns.boxplot(x = 'SaleType', y = 'SalePrice', data = train_house)

#plt.show()

plt.subplot(3,3,3)

sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = train_house)

plt.show()



# creating dummy variables for categorical variables



# subset all categorical variables

categorical = train_house.select_dtypes(include=['object'])

categorical.head()



# converting categorical values into dummies

cat_dummies = pd.get_dummies(categorical, drop_first=True)

cat_dummies.head()

# drop categorical variables 

train_house = train_house.drop(list(categorical.columns), axis=1)
# concat dummy variables with train_house df

train_house = pd.concat([train_house, cat_dummies], axis=1)

train_house.head()
train_house.shape


from sklearn.preprocessing import scale



#storing column names in cols, since column names are (annoyingly) lost after 

# scaling (the df is converted to a numpy array)

cols = train_house.columns

train_house = pd.DataFrame(scale(train_house))

train_house.columns = cols

train_house.columns
# printing data set



train_house.head()
from sklearn.model_selection import train_test_split



#  The train and test data set always have the same rows, respectively



df_train, df_test = train_test_split(train_house, train_size = 0.7, test_size = 0.3, random_state = 100)
#Now Check the correlation coefficients to see which variables are highly correlated

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize = (26, 20))

sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
#Dividing  X and Y sets for the model building

y_train = df_train.pop("SalePrice")

X_train = df_train



X_train.head()





#Checking y_train rows and columns

y_train.shape
#Checking X_train rows and columns

X_train.shape
#Dividing  X and Y sets for the model building

y_test = df_test.pop('SalePrice')

X_test = df_test
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}





ridge = Ridge()



# cross validation

folds = 10

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=200]

cv_results
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
alpha = 0.0001

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.001

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.01

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.05

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.1

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.2

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.3

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.4

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.5

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.6

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.7

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.8

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 0.9

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 1.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 2.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 3.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 4.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 5.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 6.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 7.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 8.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 9.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 10.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 11

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 12

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 13

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 14

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 15

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 16

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 16

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 17

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 18

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 19

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 20

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 21

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 22

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 23

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 24

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 25

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 26

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 27

ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)

ridge.coef_
alpha = 28

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha = 29

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
alpha =30

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
#finding r squared values of train data set

import sklearn.metrics as metrics



# linear regression

lm = LinearRegression()

lm=Ridge(alpha=10)

lm.fit(X_train, y_train)



# predict

y_train_pred = lm.predict(X_train)

metrics.r2_score(y_true=y_train, y_pred=y_train_pred)
#finding r squared values of test data set

y_test_pred = lm.predict(X_test)

metrics.r2_score(y_true=y_test, y_pred=y_test_pred)
# Ridge model parameters

model_parameters = list(ridge.coef_)

model_parameters.insert(0, lm.intercept_)

model_parameters = [round(x, 3) for x in model_parameters]

cols = train_house.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))
lasso = Lasso()



# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
# Checking alpha value with 0.0001

alpha =0.0001



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 

lasso.coef_
# Checking alpha value with 0.001

alpha =0.001



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 



lasso.coef_
# Checking alpha value with 0.01

alpha =0.01



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 



lasso.coef_
# Checking alpha value with 0.02

alpha =0.02



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
lasso.coef_
# plot

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('r2 score')

plt.xscale('log')

plt.show()
# lasso regression

import sklearn.metrics as metrics

lm = Lasso(alpha=0.01)

lm.fit(X_train, y_train)



# predict

y_train_pred = lm.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lm.predict(X_test)

print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
# lasso model parameters

model_parameters = list(lasso.coef_)

model_parameters.insert(0, lm.intercept_)

model_parameters = [round(x, 3) for x in model_parameters]

cols = train_house.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))