import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing # package containing modules for modelling, data processing, etc.

from sklearn import impute

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.metrics import mean_squared_log_error

from xgboost import XGBRegressor

import seaborn as sns # visualization package #1

import matplotlib.pyplot as plt # visualization package #2

# Configure visualisations

%matplotlib inline
# Import files containing data

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



# Convert .csv to dataframes

train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")

test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", index_col="Id")
# Remove rows with missing target, separate target from predictors

train_data = train_df.copy()

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y_train_pip = train_data.SalePrice

X_train_pip = train_data.drop(['SalePrice'], axis=1)



# Select categorical columns

categorical_cols = [col for col in X_train_pip.columns if X_train_pip[col].nunique() < 10 and X_train_pip[col].dtype == "object"]



# Select numerical columns

numerical_cols = [col for col in X_train_pip.columns if X_train_pip[col].dtype in ['int64', 'float64']]



# Joint numerical and categorical columns

my_cols = categorical_cols + numerical_cols

X_train_pip = X_train_pip[my_cols].copy()



X_train, X_valid, y_train, y_valid = train_test_split(X_train_pip, y_train_pip, random_state=1)



# Preprocessing for numerical data

numerical_transformer = impute.SimpleImputer(strategy='most_frequent')

                  

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[ ('imputer', impute.SimpleImputer(strategy='constant', fill_value='Empty')),

                                           ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore')) ])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer( transformers=[ ('num', numerical_transformer, numerical_cols),

                                                 ('cat', categorical_transformer, categorical_cols) ])



# Define model

model = XGBRegressor(n_estimators=500, learning_rate=0.1, random_state=1, objective ='reg:squarederror', 

                     early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])



# Bundle preprocessing and modeling code in a pipeline

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])



# Fit the model 

pipe.fit(X_train, y_train)

# Use cross-validation to compute the average mae score

score = cross_val_score(pipe, X_train, y_train, scoring='neg_mean_squared_log_error', cv=4)
# print scores and the mean score of the analysis

print(score)

print("Mean score: %.5f" %(-1 * score.mean()))
# join both train and test sets so that the newly created features are incorporated in both sets

dataset = pd.concat([train_df, test_df], sort=False)

dataset.index.name = 'Id'

# count number of missing entries

countmissing = dataset.isnull().sum().sort_values(ascending=False)

percentmissing = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)

# create a dataframe containing the missing values per feature and the %% of missing values

dataset_na = pd.concat([countmissing,percentmissing], axis=1)

dataset_na.head(36) # 36 entries so that all features containing missing values are shown
# the most frequent entry is used to fill the missing values of these features since a very small % is missing

dataset['Utilities'] = dataset['Utilities'].fillna("AllPub")

dataset['Electrical'] = dataset['Electrical'].fillna("SBrkr")

dataset['Exterior1st'] = dataset['Exterior1st'].fillna("VinylSd")

dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna("VinylSd")



# Missing integer values replaced with the median in order to return an integer

dataset['BsmtFullBath']= dataset.BsmtFullBath.fillna(dataset.BsmtFullBath.median())

dataset['BsmtHalfBath']= dataset.BsmtHalfBath.fillna(dataset.BsmtHalfBath.median())

dataset['GarageCars']= dataset.GarageCars.fillna(dataset.GarageCars.median())



# Missing float values were replaced with the mean for accuracy 

dataset['BsmtUnfSF']= dataset.BsmtUnfSF.fillna(dataset.BsmtUnfSF.mean())

dataset['BsmtFinSF2']= dataset.BsmtFinSF2.fillna(dataset.BsmtFinSF2.mean())

dataset['BsmtFinSF1']= dataset.BsmtFinSF1.fillna(dataset.BsmtFinSF1.mean())

dataset['GarageArea']= dataset.GarageArea.fillna(dataset.GarageArea.mean())

dataset['MasVnrArea']= dataset.MasVnrArea.fillna(dataset.MasVnrArea.mean())
# Infer missing values for the following features using information from other features

# Garage was built at the earliest when the house was built, hence it should be a good approximation

dataset.GarageYrBlt.fillna(dataset.YearBuilt, inplace=True)

# A better approximation than just using the mean for the Basement Area is to assume that it is equal to the 1st Floor Area (which usually is what happens, excluding porches or balconies)

# plot that might be used to check the inference of Basement Area values from 1st Floor Area

sns.lmplot(x="TotalBsmtSF", y="1stFlrSF", data=dataset)

plt.title("Basement Area vs 1st Floor Area")

plt.xlim(0,)

plt.ylim(0,)

plt.show()

#Update values of Basement Area

dataset.TotalBsmtSF.fillna(dataset['1stFlrSF'], inplace=True)
# Regarding LotFrontage as there is no value for the depth of the lot, one may try to infer it from the LotArea

# Let's use feature correlation to check the proposed theory

# create a new dataframe containing only the features related to the lot

lotfront_check = dataset[['LotArea','LotConfig','LotFrontage','LotShape']]

lotfront_check = pd.get_dummies(lotfront_check)

lotfront_check.corr()['LotFrontage'].sort_values(ascending=False)
# Let's assume that the lot is squared, hence the LotFrontage is going to be the square root of the LotArea. 

lotfront_check["LotAreaUnSq"] = np.sqrt(lotfront_check['LotArea'])

# Let's plot the newly created feature to check if the assumption is satisfatory or if we are thinking wrong...

sns.regplot(x="LotAreaUnSq", y="LotFrontage", data=lotfront_check)

plt.title("LotArea vs LotFrontage")

plt.xlim(0,)

plt.ylim(0,)

plt.show()
# Update the LotFrontage of dataset dataframe

dataset['LotFrontage']= dataset.LotFrontage.fillna(np.sqrt(dataset.LotArea))

dataset['LotFrontage']= dataset['LotFrontage'].astype(int)

# Plot the distribution of the imputed 'LotFrontage' to check its distribution. 

# The distribution should be similar to the original one - so that its properties are kept

# Distribution of values after replacement of missing frontage

plt.figure(figsize=(7,6))

sns.kdeplot(dataset['LotFrontage'])

sns.kdeplot(lotfront_check['LotFrontage'])

sns.kdeplot(lotfront_check['LotAreaUnSq'])

plt.title("Distribution of Lot Frontage")

plt.show() 
# Correlation of features with House SalePrice

train_corr = pd.DataFrame(dataset.corr()['SalePrice']) # convert series to dataframe to allow sorting

# correct column label from SalePrice to correlation

train_corr.columns = ["Correlation"]

# sort correlation

train_corr = train_corr.sort_values(by=['Correlation'], ascending=False)

train_corr.head(15)
# build a feature that includes all of the house area

dataset['LivingTotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF'] + dataset['GarageArea'] + dataset['WoodDeckSF'] + dataset['OpenPorchSF']

# Total Living Area divided by LotArea

dataset['PercentSQtoLot'] = dataset['LivingTotalSF'] / dataset['LotArea']

# Total count of all bathrooms including full and half through the entire building

dataset['TotalBaths'] = dataset['BsmtFullBath'] + dataset['BsmtHalfBath'] + dataset['HalfBath'] + dataset['FullBath']

# Percentage of total rooms are bedrooms

dataset['PercentBedrmtoRooms'] = dataset['BedroomAbvGr'] / dataset['TotRmsAbvGrd']

# Number of years since last remodel, if there never was one it would be since it was built

dataset['YearSinceRemodel'] = 2016 - ((dataset['YearRemodAdd'] - dataset['YearBuilt']) + dataset['YearBuilt'])
# check the correlation values after newly created features

# Correlation of features with House SalePrice

#train_corr2 = pd.DataFrame(dataset['LivingTotalSF','PercentSQtoLot','TotalBaths','PercentBedrmtoRooms','YearSinceRemodel'].corr()['SalePrice']) # convert series to dataframe to allow sorting

train_corr2 = pd.DataFrame(dataset.corr()['SalePrice']) # convert series to dataframe to allow sorting

# correct column label from SalePrice to correlation

train_corr2.columns = ['Correlation']

# sort correlation

train_corr2 = train_corr2.sort_values(by=['Correlation'], ascending=False)

train_corr2.head(15)
train_X.head()
train_X = dataset[dataset['SalePrice'].notnull()].copy()

train_X.drop(['SalePrice'], axis=1, inplace=True)

train_y = dataset[dataset['SalePrice'].notnull()]['SalePrice'].copy()



test_X =  dataset[dataset['SalePrice'].isnull()].copy()

del test_X['SalePrice']
# Select categorical columns

categorical_cols = [col for col in train_X.columns if train_X[col].nunique() < 10 and train_X[col].dtype == "object"]

# Select numerical columns

numerical_cols = [col for col in train_X.columns if train_X[col].dtype in ['int64', 'float64']]

# Joint numerical and categorical columns

my_cols = categorical_cols + numerical_cols

train_X = train_X[my_cols].copy()

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, random_state=1)

# Preprocessing for numerical data

numerical_transformer = impute.SimpleImputer(strategy='most_frequent')           

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[ ('imputer', impute.SimpleImputer(strategy='constant', fill_value='Empty')),('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore')) ])

# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer( transformers=[ ('num', numerical_transformer, numerical_cols),('cat', categorical_transformer, categorical_cols) ])

# Define model

model = XGBRegressor(n_estimators=500, learning_rate=0.1, random_state=1, objective ='reg:squarederror', early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])

# Bundle preprocessing and modeling code in a pipeline

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the model 

pipe.fit(X_train, y_train)

# Use cross-validation to compute the average mae score

score = cross_val_score(pipe, X_train, y_train, scoring='neg_mean_squared_log_error', cv=4)

print("Mean score: %.5f" %(-1 * score.mean()))
# Iteration #1

# Preprocessing for numerical data

numerical_transformer = impute.SimpleImputer(strategy='most_frequent')

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[ ('imputer', impute.SimpleImputer(strategy='most_frequent')),

                                           ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore')) ])
# Iteration #2

# Preprocessing for numerical data

numerical_transformer = impute.SimpleImputer(strategy='mean')

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[ ('imputer', impute.SimpleImputer(strategy = "constant", fill_value="Empty")),

                                           ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore')) ])
# Iteration #3

# Preprocessing for numerical data

numerical_transformer = impute.SimpleImputer(strategy='median')

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[ ('imputer', impute.SimpleImputer(strategy = "constant", fill_value="Empty")),

                                           ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore')) ])
# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer( transformers=[ ('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols) ])

# Define model

model = XGBRegressor(n_estimators=500, learning_rate=0.1, random_state=1, objective ='reg:squarederror', early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])

# Bundle preprocessing and modeling code in a pipeline

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Fit the model 

pipe.fit(X_train, y_train)

# Use cross-validation to compute the average mae score

score = cross_val_score(pipe, X_train, y_train, scoring='neg_mean_squared_log_error', cv=4)

print("Mean score: %.5f" %(-1 * score.mean()))
#Use GridSearchCV to find the best learning_rate

modelB = XGBRegressor(n_estimators=500, random_state=1, objective ='reg:squarederror', early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])

# Bundle preprocessing and modeling code in a pipeline

pipeB = Pipeline(steps=[('preprocessor', preprocessor), ('modelB', modelB)])

# Fit the model 

pipeB.fit(X_train_pip, y_train_pip)



# command to check keys of pipeline, to use on GridSearchCv

# sorted(pipe2.get_params().keys())



params = {"modelB__learning_rate" : [0.01, 0.025, 0.05, 0.1, 0.5]}



grid_search = GridSearchCV(pipeB, param_grid=params, scoring="neg_mean_squared_log_error", cv=4)

grid_result = grid_search.fit(X_train_pip, y_train_pip)
# summarize GridSearchCv results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']



for mean, stdev, param in zip(means, stds, params):

	print("%f (%f) with: %r" % (mean, stdev, param))
#Use GridSearchCV to find the best learning_rate

modelC = XGBRegressor(random_state=1, learning_rate=0.05, objective ='reg:squarederror', early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])

# Bundle preprocessing and modeling code in a pipeline

pipeC = Pipeline(steps=[('preprocessor', preprocessor), ('modelC', modelC)])

# Fit the model 

pipeC.fit(X_train_pip, y_train_pip)



paramsB = {"modelC__n_estimators" : [500, 750, 1000, 1250]}



grid_search2 = GridSearchCV(pipeC, param_grid=paramsB, scoring="neg_mean_squared_log_error", cv=4)

grid_result2 = grid_search2.fit(X_train_pip, y_train_pip)
# summarize GridSearchCv results

print("Best: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))

meansB = grid_result2.cv_results_['mean_test_score']

stdsB = grid_result2.cv_results_['std_test_score']

paramsB = grid_result2.cv_results_['params']



for mean, stdev, param in zip(meansB, stdsB, paramsB):

	print("%f (%f) with: %r" % (mean, stdev, param))
# Preprocessing for numerical data

numerical_transformer = impute.SimpleImputer(strategy='median')

# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[ ('imputer', impute.SimpleImputer(strategy = "constant", fill_value="Empty")),

                                           ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore')) ])

# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer( transformers=[ ('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols) ])

#Use GridSearchCV to find the best learning_rate

modelD = XGBRegressor(n_estimators=750, learning_rate= 0.05,random_state=1, objective ='reg:squarederror', early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])

# Bundle preprocessing and modeling code in a pipeline

pipeD = Pipeline(steps=[('preprocessor', preprocessor), ('modelD', modelD)])

# Fit the model 

pipeD.fit(X_train, y_train)



# Preprocessing of test data

X_test_pip = test_df[my_cols].copy()

# Predict on the test data

preds_test_pip = pipeD.predict(X_test_pip)

# Save test predictions to file

output = pd.DataFrame({'Id': X_test_pip.index, 'SalePrice': preds_test_pip})

output.to_csv('submission.csv', index=False)