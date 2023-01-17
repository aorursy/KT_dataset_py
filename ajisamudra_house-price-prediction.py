# Library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

import seaborn as sns # visualization

from sklearn.preprocessing import minmax_scale # min-max normalization

from sklearn.linear_model import LinearRegression # linear regression

from sklearn.linear_model import RidgeCV # ridge regularization

from sklearn.linear_model import LassoCV # lasso regularization

from sklearn.tree import DecisionTreeRegressor # decision tree

from sklearn.ensemble import RandomForestRegressor # random forest

from sklearn.model_selection import StratifiedKFold # Stratified K-Fold for cross-validation

from sklearn.model_selection import KFold # Stratified K-Fold for cross-validation

from sklearn.model_selection import GridSearchCV # to perform Grid search cross-validation 

from sklearn.model_selection import cross_val_score # to calculate cross-validation score

from sklearn.metrics import mean_squared_error # to calculate performance metric

import time

from datetime import datetime
# Read file

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# From this file, the target (Y) is SalePrice column

# Ideally we separate target (Y) and predictors (X), before that let see a bit summary of train.csv and test.csv
# Get sense in train dataset

train.describe()



# Several columns don't have complete value
# Get sense in test dataset

test.describe()



# Several columns don't have complete value
# Get sense in target distribution

original_y = train['SalePrice'].reset_index(drop=True) # will be used as comparison on performance

train['SalePrice'].hist(bins = 50)



# Right-skewed distribution
# Using Log to create normally distributed SalePrice

train.reset_index(drop=True, inplace=True)

train["SalePrice"] = np.log1p(train["SalePrice"])

train['SalePrice'].hist(bins = 50)



# Create separate data frame of target (Y)

y = train['SalePrice'].reset_index(drop=True)
# Concat train and test dataset before handling missing value

# Drop unecessary column



train = train.drop(['Id', 'SalePrice'], axis=1) # ID dont have meaning for predicting SalePrice

test = test.drop(['Id'], axis=1)



# Create separte data frame of predictors (X)

x = pd.concat([train, test]).reset_index(drop=True)

x.describe()
# Handling missing value



x.info()

# PoolQC has 2909 missing value

# MiscFeature has 2814 missing value

# Alley has 2721 missing value

# Fence has 2348 missing value

# FireplaceQu has 1420 missing value

# LotFrontage has 227 missing value

# GarageYrBlt, GarageFinish, GarageQual, GarageCond have 159 missing value

# GarageType has 157 missing value

# BsmtCond, BsmtExposure have 82 missing value

# GarageYrBlt, BsmtQual has 81 missing value

# BsmtFinType2 has 80 missing value

# BsmtFinType1 has 79 missing value

# MasVnrArea has 23 missing value

# MasVnrType has 24 missing value

# BsmtFullBath, BsmtHalfBath, Utilities, Functional has 2 missing value

# SaleType, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF have 1 missing value 

# Exterior1st, Exterior2nd, Electrical, KitchenQual  have 1 missing value

# GarageCars, GarageArea have 1 missing value

# Convert from numeric to categorical value

x['MSSubClass'] = x['MSSubClass'].apply(str)

x['YrSold'] = x['YrSold'].astype(str)

x['MoSold'] = x['MoSold'].astype(str)



# Fill these columns with suitable value

x['Functional'] = x['Functional'].fillna('Typ') 

x['Electrical'] = x['Electrical'].fillna("SBrkr") 

x['KitchenQual'] = x['KitchenQual'].fillna("TA") 



# Filling these with MODE , i.e. , the most frequent value in these columns .

x['Exterior1st'] = x['Exterior1st'].fillna(x['Exterior1st'].mode()[0]) 

x['Exterior2nd'] = x['Exterior2nd'].fillna(x['Exterior2nd'].mode()[0])

x['SaleType'] = x['SaleType'].fillna(x['SaleType'].mode()[0])

x['MasVnrArea'] = x['MasVnrArea'].fillna(x['MasVnrArea'].mode()[0])





# Filling LotFrontage

x['LotFrontage'] = x.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# Missing value in Garage-related columns most probably because the house doesnt have Garage.

# Numeric columns will be replaced with 0. Categorical columns will be replaced with 'None'

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    x[col] = x[col].fillna(0)



for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    x[col] = x[col].fillna('None')



# Missing value in Basement-related column most probably because the house doesnt have Basement. 

# Numeric columns will be replaced with 0. Categorical columns will be replaced with 'None'

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath' ):

    x[col] = x[col].fillna(0)

    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    x[col] = x[col].fillna('None')

    

# Other missing value in categorical columns will be replaced with 'None'

objects = []

for i in x.columns:

    if x[i].dtype == object:

        objects.append(i)

x.update(x[objects].fillna('None'))



# Other missing value in numeric columns will be replaced with 0.

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in x.columns:

    if x[i].dtype in numeric_dtypes:

        numerics.append(i)

x.update(x[numerics].fillna(0))

# Interesting feature need EDA 'LotFrontage', Mas Vnr Area



x.info()

# Each predictors have the 2919 value
# Create new features

# Total square feet of house

x['total_sf'] = (x['TotalBsmtSF'] + x['BsmtFinSF1'] + x['BsmtFinSF2'] +

                                 x['1stFlrSF'] + x['2ndFlrSF'])

# Total bathrooms

x['total_bathrooms'] = (x['FullBath'] + (0.5 * x['HalfBath']) +

                               x['BsmtFullBath'] + (0.5 * x['BsmtHalfBath']))

# Total porchs

x['total_porch_sf'] = (x['OpenPorchSF'] + x['3SsnPorch'] +

                              x['EnclosedPorch'] + x['ScreenPorch'] +

                              x['WoodDeckSF'])

# Boolean features

# if value feature > 0 then it'll has flag 1

x['hasPool'] = x['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

x['has2ndFloor'] = x['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

x['hasGarage'] = x['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

x['hasBasement'] = x['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

x['hasFireplace'] = x['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



x.describe()
# Many numeric features have different range of values

# It might confuse the algorithm, so we will do normalization to numeric feature

# MinMax normalization



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric_columns = []



for i in x.columns:

    if x[i].dtype in numeric_dtypes:

        numeric_columns.append(i)



x_normalized = x

x_normalized[numeric_columns] = minmax_scale(x[numeric_columns], feature_range=(0, 1))



# Separate dataset in order to measure effort performance

x_train_1 = x.iloc[:len(y), :]

x_test_1 = x.iloc[len(y):, :]



x_normalized[numeric_columns].describe() # predictors X now on the same scale
# See distribution in numeric features



# Define how many plots along and across

ncols = 5

nrows = int(np.ceil(len(x_normalized[numeric_columns].columns) / (1.0*ncols)))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))



# Counter, so we can remove unwated axes

counter = 0

for i in range(nrows):

    for j in range(ncols):



        ax = axes[i][j]



        # Plot when we have data

        if counter < len(x_normalized[numeric_columns].columns):



            ax = sns.distplot(x_normalized[x_normalized[numeric_columns].columns[counter]], bins=30, ax=axes[i,j], label = x_normalized[numeric_columns].columns[counter] )

            leg = ax.legend(loc='best')

            leg.draw_frame(False)



        # Remove axis when we no longer have data

        else:

            ax.set_axis_off()



        counter += 1

plt.show() 
# Several algorithm could not handle string value, so we should transform it into numeric value

# One of method is to create one-hot encoding

# It will create dummy column which represent category value in binary



categorical_columns = []



for i in x.columns:

    if x[i].dtype == object:

        categorical_columns.append(i)



x_train_1 = x.iloc[:len(y), :]

        

x_final = x_normalized

x_final = pd.get_dummies(x_normalized[categorical_columns]).reset_index(drop=True)

x_final.shape # get 301 features



# Separate dataset in order to measure effort performance

x_train_2 = x_normalized.iloc[:len(y), :]

x_test_2 = x_normalized.iloc[len(y):, :]
# Before input our predictors to machine learning algorithm, we will split back train and test dataset



x_train = x_final.iloc[:len(y), :]

x_test = x_final.iloc[len(y):, :]

x_train.shape, y.shape, x_test.shape
# In order to get estimation of our performance in Test dataset, we will use cross-validation

# We will split train dataset into 5 and do cross-validation



# Stratified k-fold being used in order to get stratified split

kfolds = KFold(n_splits=5, random_state=41)



# For the sake of simplicity measuring performance metric (RMSE)

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=x_train):

    rmse = np.sqrt(-cross_val_score(model, X, y, cv=kfolds, scoring="neg_mean_squared_error"))

    return (rmse)    
# Function to get best Random Forest model using Grid-Search

def rf_grid_model(X, y):

# Perform Grid-Search

    gsc = GridSearchCV(

        estimator=RandomForestRegressor(),

        param_grid={

            'n_estimators': (50, 100, 1000),

            'min_samples_split': (2, 3),

            'max_features': (1, 5, 10)

        },

        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    

    grid_result = gsc.fit(X, y)

    best_params = grid_result.best_params_

    

    rfr = RandomForestRegressor(n_estimators=best_params["n_estimators"],

                                min_samples_split=best_params['min_samples_split'], max_features=best_params['max_features'],

                                random_state=False, verbose=False)

    return rfr



# Function to get best Decision Tree model using Grid-Search

def dt_grid_model(X, y):

# Perform Grid-Search

    gsc = GridSearchCV(

        estimator=DecisionTreeRegressor(),

        param_grid={

            'min_samples_split': (2, 3),

            'max_features': (1, 5, 10)

        },

        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    

    grid_result = gsc.fit(X, y)

    best_params = grid_result.best_params_

    

    dtr = DecisionTreeRegressor( min_samples_split=best_params['min_samples_split'], max_features=best_params['max_features'],

                                random_state=False)

    return dtr
# Define models



ridge_reg = RidgeCV(alphas = [1e-3, 1e-2, 1e-1, 1, 10, 15, 20], cv = kfolds)

lasso_reg = LassoCV(alphas = [5e-05,0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006], cv = kfolds)

tree_reg = DecisionTreeRegressor()

forest_reg = RandomForestRegressor()

tree_grid_reg = dt_grid_model(x_train, y)

forest_grid_reg = rf_grid_model(x_train, y)

tree_grid_reg_numeric = dt_grid_model(x_train_2[numerics], y) # Decision Tree only numeric

forest_grid_reg_numeric = rf_grid_model(x_train_2[numerics], y) # Random Forest only numeric

# Get cross-validation score in several models



score = cv_rmse(ridge_reg , x_train)

print("RIDGE: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lasso_reg , x_train)

print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(tree_reg , x_train)

print("DECISION TREE: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(forest_reg , x_train)

print("RANDOM FOREST: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(tree_grid_reg , x_train)

print("SearchGrid - DECISION TREE: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(forest_grid_reg , x_train)

print("SearchGrid - RANDOM FOREST: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(tree_grid_reg_numeric , x_train_2[numerics])

print("SearchGrid 2 - DECISION TREE: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(forest_grid_reg_numeric , x_train_2[numerics])

print("SearchGrid 2 - RANDOM FOREST: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

# Best RMSE performance LASSO: 0.1730

lasso_reg.fit(x_train,y)



# Result submission

print('Predict submission')

submission = pd.read_csv("../input/sample_submission.csv")

# Do Expm 1 to convert Log1pm for submission

submission.iloc[:,1] = (np.expm1(lasso_reg.predict(x_test)))

submission.to_csv("submission.csv", index=False)
# Measurement in effort

# We will see impact of our effort using RMSE on training data

# List of major activity in this notebook are: (1) change target to log-scale, (2) normalization, and (3) one-hot encoding

# Using Lasso as algorithm to compare these efforts



# original_y - Y before transformed into log

# x -> original X

# x_normalized -> Normalized X

# x_final -> Transform one-hot encoding X



x_train_1.shape, x_train_2.shape, x_train.shape, x_test_1.shape, x_test_2.shape, x_test.shape
# Original Target could be used as original_y

lasso_reg.fit(x_train,original_y)

y_predicted = lasso_reg.predict(x_train)

original_y_rmse = rmsle(original_y, y_predicted) # RMSE on training data = 29242.86277804366

# original_y_rmse



# Before Normalized X

# Only include numeric features

lasso_reg.fit(x_train_1[numerics],y)

y_predicted = lasso_reg.predict(x_train_1[numerics])

original_x_rmse = rmsle(y, y_predicted) # RMSE on training data = 0.14708380182717076



# Normalized X

lasso_reg.fit(x_train_2[numerics],y)

y_predicted = lasso_reg.predict(x_train_2[numerics])

normalized_x_rmse = rmsle(y, y_predicted) # RMSE on training data = 0.14708380182717076



# Print

print("RMSE Original Y on Training Data")

print(original_y_rmse)



print("\nRMSE Original X on Training Data")

print(original_x_rmse)



print("\nRMSE Normalized X on Training Data")

print(normalized_x_rmse)
# It turns out Only normalized predictors without one-hot encoding perform better in this dataset

# Do re-submission using only Normalized-Numeric-Features

print('Predict submission')

submission2 = pd.read_csv("../input/sample_submission.csv")

# Do Expm 1 to convert Log1pm for submission

lasso_reg.fit(x_train_2[numerics],y)

submission2.iloc[:,1] = (np.expm1(lasso_reg.predict(x_test_2[numerics])))

submission2.to_csv("submission2.csv", index=False)
# Best estimated result of RMSE is 0.1409

# Random Forest with only Numeric Features perform better than others model



# Fit Random Forest model

forest_grid_reg_numeric.fit(x_train_2[numerics],y)

y_predicted = forest_grid_reg_numeric.predict(x_train_2[numerics])



print("RMSE on Training Data")

print(rmsle(y, y_predicted))

# RMSE on training data = 0.05478232174503214
# Result re-submission 3

print('Predict submission')

submission3 = pd.read_csv("../input/sample_submission.csv")

# Do Expm 1 to convert Log1pm for submission

submission3.iloc[:,1] = (np.expm1(forest_grid_reg_numeric.predict(x_test_2[numerics])))

submission3.to_csv("submission3.csv", index=False)