import pandas as pd

import numpy as np



# read training data

data = pd.read_csv('../input/train.csv')

# drop values without target saleprice

data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# get targets and numerical predictors

y = data.SalePrice

X = data.drop(['Id','SalePrice'], axis=1)
##### Define our class

from sklearn.base import BaseEstimator, TransformerMixin

class One_Hot_Transformer(BaseEstimator, TransformerMixin):

    """

    Simple class to one-hot data.

    Accepts a DataFrame which can be a mix of numerical and text data.

    Skeleton class for example purposes.

    """

    

    def __init__(self, verbose=False):

        self.verbose = verbose



        

    ##### The fit function

    def fit (self, X, y=None):

        """

        Prepares a list of columns for one-hot data,

        without actually performing any operations.

        """

        # get text variables only

        X_text = X.select_dtypes(include=['object']).copy()



        # get a list of one-hot column names

        all_columns = pd.get_dummies(X_text, dummy_na=True).columns 



        # get a list of high cardinality columns

        max_cardinality = 10

        high_cardinality_columns = [col for col in X_text.columns

                                                          if X_text[col].nunique() > max_cardinality]

        

        if self.verbose:

            print("Found high-cardinality columns:", high_cardinality_columns)



        # store a final list of column names in a DataFrame

        # while dropping the high cardinality columns

        self.one_hot_columns = pd.DataFrame(columns = 

            [col for col in all_columns if col not in high_cardinality_columns] )



        return self

    

    

    ##### The transform function

    def transform(self, X, y=None):

        """

        Does the one-hotting operation.

        """

        # split numeric and text

        X_numeric = X.select_dtypes(exclude=['object']).copy()

        X_text = X.select_dtypes(include=['object']).copy()

        

        # get one-hot columns

        X_onehot = pd.get_dummies(X_text, dummy_na=True)



        # match columns of X_onehot to our fitted (training) data

        # by throwing away columns, or adding extra NaN columns

        X_onehot_aligned, dummy = X_onehot.align(self.one_hot_columns, join='right', axis=1)

        

        if self.verbose:

            throw_away = [col for col in X_onehot if col not in self.one_hot_columns]

            print("Dropped unknown columns:", throw_away)



        ### recombine the imputed and one-hotted data

        X_transformed = pd.concat([X_numeric, X_onehot_aligned], axis=1)



        if self.verbose:

            print("Final transformed predictors have shape:", X_transformed.shape)

        

        return X_transformed
# Create the transformer and modeller objects

onehot = One_Hot_Transformer(verbose=False)

from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05)



# Combine the data transformation and modeller into a pipeline

from sklearn.pipeline import make_pipeline

pipeline = make_pipeline( onehot, xgb )
# Run on training data and cross-validate

from sklearn.model_selection import cross_val_score

print("Running cross-validation on test data...")

SLE = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_log_error')



# print our score

from math import sqrt

RMSLE = sqrt( - SLE.mean())

print("Final model score as RMSLE:", RMSLE)
# re-fit pipeline on 100% of training data

pipeline.fit(X, y)



# read in test data

test_data = pd.read_csv('../input/test.csv')

test_predictors = test_data.drop(['Id'], axis=1)



# apply pipeline

predictions =  pipeline.predict(test_predictors)



# store for submission

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})

my_submission.to_csv('submission.csv', index=False)