import pandas as pd

import numpy as np



# read training data

data = pd.read_csv('../input/train.csv')

# drop values without target saleprice

# (in this case, there are none, but good practice nonetheless)

data.dropna(axis=0, subset=['SalePrice'], inplace=True)



# split targets and predictors

y = data.SalePrice

X = data.drop(['Id','SalePrice'], axis=1)
# split numeric and text

X_numeric = X.select_dtypes(exclude=['object'])

X_text = X.select_dtypes(include=['object'])



##### Prepare to impute the numerical data

from sklearn.preprocessing import Imputer

# make a copy of the numerical data

X_imputed = X_numeric.copy()

# make a record of any NaNs by creating extra predictor columns

cols_with_missing = [col for col in X_imputed.columns if X_imputed[col].isnull().any() ]

for col in cols_with_missing:

    X_imputed[col + '_was_missing'] = X_imputed[col].isnull()
def impute_DataFrame(DF):

    """

    Calls SKLearn's Imputer, but casts the imputed object back

    as a DataFrame rather than a NumPy array

    """

    my_imputer = Imputer()

    columns = DF.columns

    index = DF.index

    DF_imputed = pd.DataFrame(my_imputer.fit_transform(DF))

    DF_imputed.columns = DF.columns

    DF_imputed.index = DF.index

    return DF_imputed



X_imputed = impute_DataFrame(X_imputed)
##### one hot the text data

# first let's check the cardinalities

cardinalities = [X_text[col].nunique() for col in X_text.columns]

print("One-hot cardinalities per variable:", cardinalities)



# drop high cardinality columns

max_cardinality = 10

high_cardinality_columns = [col for col in X_text.columns

                            if X_text[col].nunique() > max_cardinality]

X_text = X_text.drop(high_cardinality_columns, axis=1)

print("Dropped text columns with more than", max_cardinality, "unique labels:")

print(high_cardinality_columns)



# do the one-hotting (is that a verb?)

X_onehot = pd.get_dummies(X_text, dummy_na=True)
# recombine the imputed and one-hotted data

X_train = pd.concat([X_imputed, X_onehot], axis=1)

print("Final predictors have shape:", X_train.shape)
# Define XGBoost model

from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=500, learning_rate=0.05)
# Define error metric function, as required by XGBoost

from sklearn.metrics import mean_squared_log_error

from math import sqrt,log

def metric(a,b):

    b = b.get_label()

    RMSLE = sqrt(mean_squared_log_error(a, b))

    return ("RMSLE", RMSLE)
# Fit the number of estimators

from sklearn.model_selection import train_test_split

def get_n_estimators(X,y):

    """

    Split the traning data further, into a traning and evaluation set

    Then run XGBoost until we stop seeing an improvement

    and return the optimum number of iterations

    """

    X_1, X_2, y_1, y_2 = train_test_split(X, y, random_state=0)

    my_model.fit(X_1, y_1, eval_set=[(X_2,y_2)],

                            eval_metric=metric,

                            verbose=False, early_stopping_rounds=10)

    return my_model.best_iteration



# Set n_estimators on our model

my_model.n_estimators = int(get_n_estimators(X_train,y))

print("We set n_estimators as",my_model.n_estimators)
# Cross valildate using the metric RMSLE

# this command does a lot in one line! fits and scores 5 folds of the data

from sklearn.model_selection import cross_val_score

print("Cross-training on all of the training data...")

SLE = cross_val_score(my_model, X_train, y, scoring='neg_mean_squared_log_error', cv=5 )

RMSLE = sqrt( - SLE.mean() )

print("Final model score as RMSLE:", RMSLE)
# re-fit on 100% of training data

my_model.fit(X_train, y)
# read final test data

test_data = pd.read_csv('../input/test.csv')



##### Process the test data, exactly as we did with our training data

# split

test_numeric = test_data.select_dtypes(exclude=['object'])

test_text = test_data.select_dtypes(include=['object'])

# impute

test_imputed = test_numeric.copy()

cols_with_missing = [col for col in test_imputed.columns if test_imputed[col].isnull().any() ]

for col in cols_with_missing:

    test_imputed[col + '_was_missing'] = test_imputed[col].isnull()

test_imputed = impute_DataFrame(test_imputed)

# one-hot

high_cardinality_columns = [col for col in test_text.columns

                            if test_text[col].nunique() > max_cardinality]

test_text.drop(high_cardinality_columns, axis=1)

test_onehot = pd.get_dummies(test_text)

# recombine

test_data = pd.concat([test_imputed, test_onehot], axis=1)
# Left-join the test data with the training data

# so that our trained model can be applied

final_train, final_test = X_train.align(test_data, join='left', axis=1)



predictions =  my_model.predict(final_test)



my_submission = pd.DataFrame({'Id': test_data.Id.apply(int), 'SalePrice': predictions})

my_submission.to_csv('submission.csv', index=False)