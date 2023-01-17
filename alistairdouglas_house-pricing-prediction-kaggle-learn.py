import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

X_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')
import seaborn as sns

import matplotlib.pyplot as plt



# Get numeric and categorical columns

num_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64','float64']]

cat_cols = [cname for cname in X_full.columns if X_full[cname].dtype == 'object']



# Nominal data exploration

X_full.hist(bins=30, figsize=(25,25))

plt.show()
f = plt.figure(figsize=(25,25))



for i in range(len(num_cols)):

    f.add_subplot(10, 5, i+1)

    sns.regplot(X_full[num_cols].iloc[:,i], X_full['SalePrice'])

    

plt.tight_layout()

plt.show()
f = plt.figure(figsize=(25,25))



for i in range(len(cat_cols)):

    f.add_subplot(10, 5, i+1, ylim=[0,X_full.shape[0]])

    sns.countplot(x=X_full[cat_cols].iloc[:,i], data=X_full)

    

plt.tight_layout()

plt.show()
# Remove outliers visible in scatter plots

X_full.drop(X_full[(X_full['OverallQual'] > 9) & (X_full['SalePrice'] < 220000)].index, inplace=True)

X_full.drop(X_full[(X_full['GrLivArea'] > 4000) & (X_full['SalePrice'] < 250000)].index, inplace=True)

X_full.drop(X_full[(X_full['OverallCond'] < 3) & (X_full['SalePrice'] > 300000)].index, inplace=True)

X_full.drop(X_full[(X_full['GarageArea'] > 1230)].index, inplace=True)

X_full.drop(X_full[(X_full['TotalBsmtSF'] > 3100)].index, inplace=True)



# Log the SalePrice

X_full['SalePrice'] = np.log1p(X_full['SalePrice'])



# Drop columns with the majority of data missing

cols_to_drop = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Street','Utilities']



# Drop any columns that are not useful (assessed categorical columns)

X_full.drop(axis=1, columns=cols_to_drop, inplace=True)

X_test_full.drop(axis=1, columns=cols_to_drop, inplace=True)



# Set the target value y to the SalePrice and drop it from the training data

y = X_full['SalePrice']

X_full.drop(axis=1, columns=['SalePrice'], inplace=True)



# Get the numeric and categorical columns again since some of them have been dropped

num_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64','float64']]

cat_cols = [cname for cname in X_full.columns if X_full[cname].dtype == 'object']



# Make copies so the original isn't being manipulated

X = X_full.copy()

X_test = X_test_full.copy()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

import category_encoders as ce



# Need to be outside function for cache

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

target_encoder = ce.TargetEncoder(cols=cat_cols)

test_data = False



def general_cleaner(X, y=None):

    

    # Numerical columns

    # Decide if numerical columns should be replaced with 0 or the median

    num_cols_missing = [cname for cname in num_cols if X[cname].isnull().sum() > 0]



    num_cols_zeros = []

    num_cols_median = []



    for column in num_cols_missing:

        if X[column].min() == 0:

            num_cols_zeros.append(column)

        else:

            num_cols_median.append(column)



    # Categorical cols

    # If NaN occurs a lot, it's quite likely it stands for a category (like NA = no basement) so fill it with 'none'

    # otherwise it's probably genuine missing data so should be filled with the most frequent value.

    # 30 was determined as a good threshold for determining by visually checking the data.

    cat_cols_missing = [cname for cname in cat_cols if X[cname].isnull().sum() > 0]



    cat_cols_freq = []

    cat_cols_none = []



    for column in cat_cols_missing:

        if X[column].isnull().sum() < 30:

            cat_cols_freq.append(column)

        else:

            cat_cols_none.append(column)



    # Apply zero numeric imputer

    for cname in num_cols_zeros:

        X[cname] = SimpleImputer(strategy='constant', fill_value=0.0).fit_transform(X[[cname]])



    # Apply median numeric imputer

    for cname in num_cols_median:

        X[cname] = SimpleImputer(strategy='median').fit_transform(X[[cname]])



    # Apply highest frequency categorical imputer

    for cname in cat_cols_freq:

        X[cname] = SimpleImputer(strategy='most_frequent').fit_transform(X[[cname]])



    # Apply 'none' categorical imputer

    for cname in cat_cols_none:

        X[cname] = SimpleImputer(strategy='constant', fill_value='none').fit_transform(X[[cname]])



    # Scale all numerical values

    for cname in num_cols:

        X[cname] = StandardScaler().fit_transform(X[[cname]])



    # Apply count encoding

    count_encoded = ce.CountEncoder().fit_transform(X[cat_cols])

    X = X.join(count_encoded.add_suffix("_count"))

    

    # Apply target encoding (fit on training data, and apply to test data)

    if not test_data:

        target_encoder.fit(X[cat_cols], y)

        X = X.join(target_encoder.transform(X[cat_cols]).add_suffix('_target'))

    else:

        X = X.join(target_encoder.transform(X[cat_cols]).add_suffix('_target'))



    # Apply one hot encoding (fit on training data, and apply to test data)

    if not test_data:

        OH_X = pd.DataFrame(OH_encoder.fit_transform(X[cat_cols].astype('str')))

    else:

        OH_X = pd.DataFrame(OH_encoder.transform(X[cat_cols].astype('str')))



    # Put the correct indexes back in since OH encoding reset the index

    OH_X.index = X.index



    # Drop the columns that have now been OH encoded

    X = X.drop(cat_cols, axis=1)



    # Add the new OH columns back in

    X = pd.concat([X, OH_X], axis=1)



    return X
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



model = XGBRegressor()



# Set of parameters for XGBRegressor to try randomly

params = {

    'n_estimators' : np.arange(100,2000,100),

    'learning_rate' : np.arange(0.01,0.2,0.01),

    'max_depth' : np.arange(1,4,1),

}



# Define a cross validator, RandomizedSearchCV to iterate over the best hyper parameters

param_search = RandomizedSearchCV(model, param_distributions=params,

                                 n_iter=1,

                                 cv=5,

                                 scoring='neg_mean_squared_error',

                                 verbose=False,

                                 random_state=1)



# Clean and pre-process the training data

X_preprocessed = general_cleaner(X, y)



# Find the best model parameters for the data data using cross validation

param_search.fit(X_preprocessed, y)



# See the best results in a list

print("\n\nRandom search:")

cv_results = param_search.cv_results_

for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):

    print(-1*mean_score, params)



print(-1*param_search.best_score_,param_search.best_params_)



# 0.013460942446364737 {'learning_rate': 0.052, 'max_depth': 2, 'n_estimators': 943}



# Now that we have a first pass at the best parameters, use GridSearchCV to do a more detailed check of the best hyper parameters

params = {

    'n_estimators' : np.arange(942,944,1),

    'learning_rate' : np.arange(0.051,0.053,0.001),

    'max_depth' : [2],

}



param_search = GridSearchCV(model, param_grid=params,

                                 cv=5,

                                 scoring='neg_mean_squared_error',

                                 verbose=False)



# Find the best model parameters for the data data using cross validation

param_search.fit(X_preprocessed, y)



# See the best results in a list

print("\n\nGrid search:")

cv_results = param_search.cv_results_

for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):

    print(-1*mean_score, params)

    

print(-1*param_search.best_score_,param_search.best_params_)



# Re-fit the model using the whole data set

param_search.refit



# Flag to switch between train and test data (used in the general_cleaner function)

test_data = True



# Clean and pre-process the test data

X_test_preprocessed = general_cleaner(X_test)



# Predict house prices using the test data

preds = param_search.predict(X_test_preprocessed)



# Inverse log predictions

preds = np.expm1(preds)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds})

output.to_csv('submission.csv', index=False)