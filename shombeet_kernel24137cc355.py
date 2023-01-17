#Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error, make_scorer,r2_score
# loading the dataset from csv file

housing = pd.read_csv('../input/housingdataset/train1.csv')

#sample values from the data set

housing.head()
# structure of data Set

housing.info()
#rows and columns of dataframe

housing.shape
# all numeric (float and int) variables in the dataset are separated

housing_num = housing.select_dtypes(include=['float64', 'int64'])

housing_num.head()
#Plot histogram of numerical variables 

fig, axes = plt.subplots(nrows = 19, ncols = 2, figsize = (40, 200))

for ax, column in zip(axes.flatten(), housing_num.columns):

    sns.distplot(housing_num[column].dropna(), ax = ax, color = 'darkred')

    ax.set_title(column, fontsize = 43)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 35)

    ax.tick_params(axis = 'both', which = 'minor', labelsize = 35)

    ax.set_xlabel('')

fig.tight_layout(rect = [0, 0.03, 1, 0.95])
#There are 38 columns which are numeric

housing_num.info()
# dropping the columns we want to treat as categorical variables from numerical columns

housing_num = housing_num.drop(['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 

                                    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                                   'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 

                                   'MoSold', 'YrSold'], axis=1)

housing_num.head()

# 17 columns are dropped
#now checking the columns which are continuous

# checking the various quantiles

housing_num.describe(percentiles=[.25, .5, .75, .90, .95, .99])
# outlier treatment of PoolArea

plt.boxplot(housing['PoolArea'])

Q1 = housing['PoolArea'].quantile(0.1)

Q3 = housing['PoolArea'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['PoolArea'] >= Q1 - 1.5*IQR) & 

                      (housing['PoolArea'] <= Q3 + 1.5*IQR)]

housing.shape
# outlier treatment of MiscVal

plt.boxplot(housing['MiscVal'])

Q1 = housing['MiscVal'].quantile(0.1)

Q3 = housing['MiscVal'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['MiscVal'] >= Q1 - 1.5*IQR) & 

                      (housing['MiscVal'] <= Q3 + 1.5*IQR)]

housing.shape
# outlier treatment of ScreenPorch

plt.boxplot(housing['ScreenPorch'])

Q1 = housing['ScreenPorch'].quantile(0.1)

Q3 = housing['ScreenPorch'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['ScreenPorch'] >= Q1 - 1.5*IQR) & 

                      (housing['ScreenPorch'] <= Q3 + 1.5*IQR)]

housing.shape
# outlier treatment of LotArea

plt.boxplot(housing['LotArea'])

Q1 = housing['LotArea'].quantile(0.1)

Q3 = housing['LotArea'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['LotArea'] >= Q1 - 1.5*IQR) & 

                      (housing['LotArea'] <= Q3 + 1.5*IQR)]

housing.shape
# outlier treatment of MasVnrArea

plt.boxplot(housing['MasVnrArea'])

Q1 = housing['MasVnrArea'].quantile(0.1)

Q3 = housing['MasVnrArea'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['MasVnrArea'] >= Q1 - 1.5*IQR) & 

                      (housing['MasVnrArea'] <= Q3 + 1.5*IQR)]

housing.shape
# outlier treatment of SalePrice

plt.boxplot(housing['SalePrice'])

Q1 = housing['SalePrice'].quantile(0.1)

Q3 = housing['SalePrice'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['SalePrice'] >= Q1 - 1.5*IQR) & 

                      (housing['SalePrice'] <= Q3 + 1.5*IQR)]

housing.shape
# checking to see if there are any duplicate rows in dataframe

housing.duplicated().sum()
# variable formats

housing.info()
#checking the number of total null values in the dataset

housing.isnull().sum()  
# Checking the percentage of missing values

round(100*(housing.isnull().sum()/len(housing.index)), 2)
housing.shape
#check columns which have missing values > 70%

columns_to_drop = housing.loc[:,(housing.isnull().sum()*100/len(housing.index)).sort_values(ascending=False) > 70].columns

columns_to_drop
#drop these columns which are > 70%

housing = housing.drop(columns_to_drop,axis=1)
#NA for these variables should be imputed with mode

housing['MasVnrType'] = housing['MasVnrType'].fillna(housing['MasVnrType'].mode()[0])

housing['FireplaceQu'] = housing['FireplaceQu'].fillna(housing['FireplaceQu'].mode()[0])

housing['GarageType'] = housing['GarageType'].fillna(housing['GarageType'].mode()[0])

housing['GarageFinish'] = housing['GarageFinish'].fillna(housing['GarageFinish'].mode()[0])

housing['GarageQual'] = housing['GarageQual'].fillna(housing['GarageQual'].mode()[0])

housing['GarageCond'] = housing['GarageCond'].fillna(housing['GarageCond'].mode()[0])

housing['BsmtQual'] = housing['BsmtQual'].fillna(housing['BsmtQual'].mode()[0])

housing['BsmtCond'] = housing['BsmtCond'].fillna(housing['BsmtCond'].mode()[0])

housing['BsmtExposure'] = housing['BsmtExposure'].fillna(housing['BsmtExposure'].mode()[0])

housing['BsmtFinType1'] = housing['BsmtFinType1'].fillna(housing['BsmtFinType1'].mode()[0])

housing['BsmtFinType2'] = housing['BsmtFinType2'].fillna(housing['BsmtFinType2'].mode()[0])

housing['Electrical'] = housing['Electrical'].fillna(housing['Electrical'].mode()[0])

housing['GarageYrBlt'] = housing['GarageYrBlt'].fillna(housing['GarageYrBlt'].mode()[0])
#NA for this variable should be imputed with median

housing['MasVnrArea'] = housing['MasVnrArea'].fillna(housing['MasVnrArea'].median())

housing['LotFrontage'] = housing['LotFrontage'].fillna(housing['LotFrontage'].median())
#check missing values again on column level

housing.isnull().sum().sum()
housing['MSSubClass'] = housing['MSSubClass'].astype('object')

housing['OverallQual'] = housing['OverallQual'].astype('object')

housing['OverallCond'] = housing['OverallCond'].astype('object')

housing['BsmtFullBath'] = housing['BsmtFullBath'].astype('object')

housing['BsmtHalfBath'] = housing['BsmtHalfBath'].astype('object')

housing['FullBath'] = housing['FullBath'].astype('object')

housing['HalfBath'] = housing['HalfBath'].astype('object')

housing['BedroomAbvGr'] = housing['BedroomAbvGr'].astype('object')

housing['KitchenAbvGr'] = housing['KitchenAbvGr'].astype('object')

housing['TotRmsAbvGrd'] = housing['TotRmsAbvGrd'].astype('object')

housing['Fireplaces'] = housing['Fireplaces'].astype('object')

housing['GarageCars'] = housing['GarageCars'].astype('object')
#checking the final shape of dataset

housing.shape
#moving a copy to final to apply modelling

final = housing
# no of categorical variables

len(housing.select_dtypes(include='object').columns)
# no of numerical variables

len(housing.select_dtypes(exclude='object').columns)
# correlation matrix

# dropping Id column which is unique

cor = housing_num.drop('Id',axis=1).corr()

cor
# plotting correlations on a heatmap

# figure size

plt.figure(figsize=(18,10))

sns.heatmap(cor, annot=True)

plt.show()
print("Find most important features relative to target")

corr = housing_num.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
plt.figure(figsize=(30, 20))

plt.subplot(5,2,1)

sns.boxplot(x = 'Street', y = 'SalePrice', data = housing)

plt.subplot(5,2,2)

sns.boxplot(x = 'MSZoning', y = 'SalePrice', data = housing)

plt.subplot(5,2,3)

sns.boxplot(x = 'LotShape', y = 'SalePrice', data = housing)

plt.subplot(5,2,4)

sns.boxplot(x = 'LandContour', y = 'SalePrice', data = housing)

plt.subplot(5,2,5)

sns.boxplot(x = 'Utilities', y = 'SalePrice', data = housing)

plt.subplot(5,2,6)

sns.boxplot(x = 'LotConfig', y = 'SalePrice', data = housing)

plt.subplot(5,2,7)

sns.boxplot(x = 'LandSlope', y = 'SalePrice', data = housing)

plt.subplot(5,2,8)

sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = housing)

plt.subplot(5,2,9)

sns.boxplot(x = 'Condition1', y = 'SalePrice', data = housing)

plt.subplot(5,2,10)

sns.boxplot(x = 'Condition2', y = 'SalePrice', data = housing)

plt.show()
plt.figure(figsize=(30, 20))

plt.subplot(5,2,1)

sns.boxplot(x = 'BldgType', y = 'SalePrice', data = housing)

plt.subplot(5,2,2)

sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = housing)

plt.subplot(5,2,3)

sns.boxplot(x = 'RoofStyle', y = 'SalePrice', data = housing)

plt.subplot(5,2,4)

sns.boxplot(x = 'RoofMatl', y = 'SalePrice', data = housing)

plt.subplot(5,2,5)

sns.boxplot(x = 'Exterior1st', y = 'SalePrice', data = housing)

plt.subplot(5,2,6)

sns.boxplot(x = 'Exterior2nd', y = 'SalePrice', data = housing)

plt.subplot(5,2,7)

sns.boxplot(x = 'MasVnrType', y = 'SalePrice', data = housing)

plt.subplot(5,2,8)

sns.boxplot(x = 'ExterQual', y = 'SalePrice', data = housing)

plt.subplot(5,2,9)

sns.boxplot(x = 'PavedDrive', y = 'SalePrice', data = housing)

plt.subplot(5,2,10)

sns.boxplot(x = 'Foundation', y = 'SalePrice', data = housing)

plt.show()
plt.figure(figsize=(30, 20))

plt.subplot(5,2,1)

sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = housing)

plt.subplot(5,2,2)

sns.boxplot(x = 'BsmtCond', y = 'SalePrice', data = housing)

plt.subplot(5,2,3)

sns.boxplot(x = 'BsmtExposure', y = 'SalePrice', data = housing)

plt.subplot(5,2,4)

sns.boxplot(x = 'BsmtFinType1', y = 'SalePrice', data = housing)

plt.subplot(5,2,5)

sns.boxplot(x = 'BsmtFinType2', y = 'SalePrice', data = housing)

plt.subplot(5,2,6)

sns.boxplot(x = 'Heating', y = 'SalePrice', data = housing)

plt.subplot(5,2,7)

sns.boxplot(x = 'HeatingQC', y = 'SalePrice', data = housing)

plt.subplot(5,2,8)

sns.boxplot(x = 'CentralAir', y = 'SalePrice', data = housing)

plt.subplot(5,2,9)

sns.boxplot(x = 'Electrical', y = 'SalePrice', data = housing)

plt.subplot(5,2,10)

sns.boxplot(x = 'KitchenQual', y = 'SalePrice', data = housing)

plt.show()
plt.figure(figsize=(30, 20))

plt.subplot(3,3,1)

sns.boxplot(x = 'Functional', y = 'SalePrice', data = housing)

plt.subplot(3,3,2)

sns.boxplot(x = 'FireplaceQu', y = 'SalePrice', data = housing)

plt.subplot(3,3,3)

sns.boxplot(x = 'GarageType', y = 'SalePrice', data = housing)

plt.subplot(3,3,4)

sns.boxplot(x = 'GarageFinish', y = 'SalePrice', data = housing)

plt.subplot(3,3,5)

sns.boxplot(x = 'GarageQual', y = 'SalePrice', data = housing)

plt.subplot(3,3,6)

sns.boxplot(x = 'GarageCond', y = 'SalePrice', data = housing)

plt.subplot(3,3,7)

sns.boxplot(x = 'PavedDrive', y = 'SalePrice', data = housing)

plt.subplot(3,3,8)

sns.boxplot(x = 'SaleType', y = 'SalePrice', data = housing)

plt.subplot(3,3,9)

sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = housing)

plt.show()
# List of variables to map

varlist_street =  ['Street']



# Defining the map function

def binary_map(x):

    return x.map({'Pave': 1, "Grvl": 0})



# Applying the function to the Lead list

final[varlist_street] = final[varlist_street].apply(binary_map)
# List of variables to map



varlist_utilities =  ['Utilities']



# Defining the map function

def binary_map(x):

    return x.map({'AllPub': 1, "NoSeWa": 0})



# Applying the function to the Lead list

final[varlist_utilities] = final[varlist_utilities].apply(binary_map)
# List of variables to map



varlist_centralair =  ['CentralAir']



# Defining the map function

def binary_map(x):

    return x.map({'Y': 1, "N": 0})



# Applying the function to the Lead list

final[varlist_centralair] = final[varlist_centralair].apply(binary_map)
# split into X and y - drop ID field since it is unique and doesnt add any patterns

X = final.drop([ 'Id'], axis=1)
# creating dummy variables for categorical variables

# subset all categorical variables

housing_cat = X.select_dtypes(include=['object'])

housing_cat.head()
# convert into dummies

housing_dum = pd.get_dummies(housing_cat, drop_first=True)

housing_dum.head()
# drop categorical variables since for these dummy variables are already created

final = final.drop(list(housing_cat.columns), axis=1)
# concat dummy variables with original dataset

final = pd.concat([final, housing_dum], axis=1)
# rows and columns of final dataset

final.shape
from sklearn.model_selection import train_test_split

df_train_housing, df_test_housing = train_test_split(final, train_size = 0.7, test_size = 0.3, random_state = 100)
#Dividing  X and Y training sets for the model building

y_train_housing = df_train_housing.pop("SalePrice")

X_train_housing = df_train_housing

X_train_housing.shape
#Checking y_train rows and columns

y_train_housing.shape
#Dividing  X and Y test sets for the model building

y_test_housing = df_test_housing.pop("SalePrice")

X_test_housing = df_test_housing

X_test_housing.shape
#Checking y_train rows and columns

y_test_housing.shape
# no of categorical variables

numerical_features = X_train_housing.select_dtypes(exclude='object').columns

numerical_features
scaler = StandardScaler()

X_train_housing.loc[:, numerical_features] = scaler.fit_transform(X_train_housing.loc[:, numerical_features])

X_test_housing.loc[:, numerical_features] = scaler.transform(X_test_housing.loc[:, numerical_features])
# hyperparameter tuning

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
# Run linear Regression without regularization

linreg = LinearRegression()

parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}

grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")

grid_linreg.fit(X_train_housing, y_train_housing)



#Enabling the learning of linear regression model and run predictions

linreg = grid_linreg.best_estimator_

linreg.fit(X_train_housing, y_train_housing)

lin_pred = linreg.predict(X_test_housing)

r2_lin = r2_score(y_test_housing, lin_pred)

print("R^2 Score: " + str(r2_lin))
#Check cross validation score for linear regression

scores_lin = cross_val_score(linreg, X_train_housing, y_train_housing, cv=5, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_lin)))
# Checking the parameters which are important after running linear regression

FI_linreg = pd.DataFrame({"Feature Importance":linreg.coef_}, index=X_train_housing.columns)

FI_linreg.sort_values("Feature Importance",ascending=False).head(15)
# Arranging the parameters in ascending order of importance

FI_linreg[FI_linreg["Feature Importance"]!=0].sort_values("Feature Importance",ascending=False).head(15).plot(kind="barh",figsize=(10,6))

plt.xticks(rotation=90)

plt.show()
lasso = Lasso()

folds = 5

# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train_housing, y_train_housing) 
# Checking the dataframe obtained from running Lasso regression

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

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
# From the above graph - alpha is optimum at 500

alpha =500

lasso = Lasso(alpha=alpha)        

lasso.fit(X_train_housing, y_train_housing) 
# Run predictions

lasso_pred = lasso.predict(X_test_housing)

r2_lasso = r2_score(y_test_housing, lasso_pred)

print("R^2 Score: " + str(r2_lasso))
# Check cross validation score

scores_lasso = cross_val_score(lasso, X_train_housing, y_train_housing, cv=5, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_lasso)))
# Checking feature importance from Lasso regularization

FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=X_train_housing.columns)

FI_lasso.sort_values("Feature Importance",ascending=False).head(15)
#Arranging the features in increasing order of importance

FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance",ascending=False).head(15).plot(kind="barh",figsize=(10,6))

plt.xticks(rotation=90)

plt.show()
# If alpha is doubled from 500 to 1000

alpha =1000

lasso_1 = Lasso(alpha=alpha)        

lasso_1.fit(X_train_housing, y_train_housing) 



# Run predictions

lasso_pred_1 = lasso_1.predict(X_test_housing)

r2_lasso_1 = r2_score(y_test_housing, lasso_pred_1)

print("R^2 Score: " + str(r2_lasso_1))



# Check cross validation score

scores_lasso_1 = cross_val_score(lasso_1, X_train_housing, y_train_housing, cv=5, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_lasso_1)))



# Checking feature importance from Lasso regularization

FI_lasso_1 = pd.DataFrame({"Feature Importance":lasso_1.coef_}, index=X_train_housing.columns)

FI_lasso_1.sort_values("Feature Importance",ascending=False).head(15)
# If top 5 predictor variables - GrLivArea - OverallQual_9,YearBuilt,OverallQual_8,BSMTFinSF1 is eliminated - Then?

columns_to_drop = ['GrLivArea','OverallQual_9','YearBuilt','OverallQual_8','BsmtFinSF1']

X_train_housing.shape

X_train_housing = X_train_housing.drop(columns_to_drop,axis=1)

X_test_housing = X_test_housing.drop(columns_to_drop,axis=1)
# If top parameters are eliminated

alpha =500

lasso_1 = Lasso(alpha=alpha)        

lasso_1.fit(X_train_housing, y_train_housing) 



# Run predictions

lasso_pred_1 = lasso_1.predict(X_test_housing)

r2_lasso_1 = r2_score(y_test_housing, lasso_pred)

print("R^2 Score: " + str(r2_lasso_1))



# Check cross validation score

scores_lasso_1 = cross_val_score(lasso_1, X_train_housing, y_train_housing, cv=5, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_lasso)))



# Checking feature importance from Lasso regularization

FI_lasso_1 = pd.DataFrame({"Feature Importance":lasso_1.coef_}, index=X_train_housing.columns)

FI_lasso_1.sort_values("Feature Importance",ascending=False).head(15)
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
# Running ridge regression for all alpha parameters

ridge = Ridge()

folds = 5

grid_ridge = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds,

                        verbose=1, 

                        return_train_score = True)

grid_ridge.fit(X_train_housing, y_train_housing)
# Checking the dataframe obtained from running Ridge regression

cv_results = pd.DataFrame(grid_ridge.cv_results_)

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
# From the above graph - optimum value of alpha is obtained

# Ridge regression run for this hyper parameter

alpha = 100

ridge = Ridge(alpha=alpha)



ridge.fit(X_train_housing, y_train_housing)

ridge.coef_
# Run predictions

ridge_pred = ridge.predict(X_test_housing)

r2_ridge = r2_score(y_test_housing, ridge_pred)

print("R^2 Score: " + str(r2_ridge))
# Check cross validation score

scores_ridge = cross_val_score(ridge, X_train_housing, y_train_housing, cv=5, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_ridge)))
# Checking feature importance from ridge regression

FI_ridge = pd.DataFrame({"Feature Importance":ridge.coef_}, index=X_train_housing.columns)

FI_ridge.sort_values("Feature Importance",ascending=False).head(15)
# Arranging the parameters in ascending order of importance

FI_ridge[FI_ridge["Feature Importance"]!=0].sort_values("Feature Importance",ascending=False).head(15).plot(kind="barh",figsize=(10,6))

plt.xticks(rotation=90)

plt.show()
# If alpha is doubled from 100 to 200

alpha =200

ridge_1 = Ridge(alpha=alpha)        

ridge_1.fit(X_train_housing, y_train_housing) 



# Run predictions

ridge_pred_1 = ridge_1.predict(X_test_housing)

r2_ridge_1 = r2_score(y_test_housing, ridge_pred_1)

print("R^2 Score: " + str(r2_ridge_1))



# Check cross validation score

scores_ridge_1 = cross_val_score(ridge_1, X_train_housing, y_train_housing, cv=5, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_ridge_1)))



# Checking feature importance from Lasso regularization

FI_ridge_1 = pd.DataFrame({"Feature Importance":ridge_1.coef_}, index=X_train_housing.columns)

FI_ridge_1.sort_values("Feature Importance",ascending=False).head(15)
model_performances = pd.DataFrame({

    "Model" : ["Linear Regression", "Ridge", "Lasso"],

    "R Squared" : [str(r2_lin)[0:5], str(r2_ridge)[0:5], str(r2_lasso)[0:5]]

})

model_performances.round(4)



print("Sorted by Best Score:")

model_performances.sort_values(by="R Squared", ascending=False)