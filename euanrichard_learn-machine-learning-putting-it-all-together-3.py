import pandas as pd

import numpy as np

from math import sqrt



# read training data

data = pd.read_csv('../input/train.csv')

# drop values without target saleprice

data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# get targets and numerical predictors

y = data.SalePrice

X = data.drop(['Id','SalePrice'], axis=1)



##### Define a class to one-hot data

# needs to inherit from the base sklearn classes.

from sklearn.base import BaseEstimator, TransformerMixin

class One_Hot_Transformer(BaseEstimator, TransformerMixin):

    """

    Simple class to one-hot data.

    Accepts a DataFrame which can be a mix of numerical and text data.

    Skeleton class for example purposes.

    """



    def __init__(self, verbose=False):

        self.verbose = verbose



    

    def fit (self, X, y=None):

        """

        Prepares a list of columns for one-hot data,

        without actually performing any operations.

        """



        # get text variables only

        X_text = X.select_dtypes(include=['object']).copy()



        # get a list of one-hot column names

        all_columns = pd.get_dummies(X_text).columns 



        # get a list of high cardinality columns

        max_cardinality = 10

        high_cardinality_columns = [col for col in X_text.columns

                                         if X_text[col].nunique() > max_cardinality]



        if self.verbose:

            print("Found high-cardinality columns:", high_cardinality_columns)



        # drop the high cardinality columns,

        # and store a final list of column names in a DataFrame

        self.one_hot_columns = pd.DataFrame(columns = 

            [col for col in all_columns if col not in high_cardinality_columns] )



        return self





    def transform(self, X, y=None):

        """

        Does the one-hotting.

        """



        # split numeric and text

        X_numeric = X.select_dtypes(exclude=['object']).copy()

        X_text = X.select_dtypes(include=['object']).copy()



        # get one-hot columns

        X_onehot = pd.get_dummies(X_text)



        # match columns of X_onehot to our training data

        # by throwing away, or adding extra NaN columns

        X_onehot_aligned, dummy = X_onehot.align(self.one_hot_columns, join='right', axis=1)



        if self.verbose:

            throw_away = [col for col in X_onehot if col not in self.one_hot_columns]

            print("Dropped unknown columns:", throw_away)



        ### recombine the imputed and one-hotted data

        X_train = pd.concat([X_numeric, X_onehot_aligned], axis=1)



        if self.verbose:

            print("Final transformed predictors have shape:", X_train.shape)

        

        return X_train

    



# Define error metric function, as required by XGBoost

# In order to use early_stopping_rounds accurately

from sklearn.metrics import mean_squared_log_error

def metric(a,b):

    b = b.get_label()

    RMSLE = sqrt(mean_squared_log_error(a, b))

    return ("RMSLE", RMSLE)



    

# Define the transformer and modeller

onehot = One_Hot_Transformer(verbose=False)

from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05)



# Combine the data transformation and modeller into a pipeline

from sklearn.pipeline import make_pipeline

pipeline = make_pipeline( onehot, xgb )
# check the Fit parameters accessible by pipeline

import pprint

pp = pprint.PrettyPrinter()

pp.pprint(pipeline.get_params())
param_grid = {

    'xgbregressor__max_depth' : list(range(1,4,1)),

    'xgbregressor__min_child_weight' : list(range(0,3,1))

}
from sklearn.model_selection import GridSearchCV



# define the grid search, using 5 k-folds

optimizer = GridSearchCV(pipeline, param_grid, cv=5,

                         scoring = 'neg_mean_squared_log_error')



print("Grid searching over parameter grid:", param_grid)

optimizer.fit(X,y)



# print results

print("Best params:",optimizer.best_params_)

print("Best RMSLE:",sqrt(-optimizer.best_score_))
# split data to x_train and x_eval

from sklearn.model_selection import train_test_split

X_train, X_eval, y_train, y_eval = train_test_split(X, y, random_state=0,

                                                                                        train_size=0.9, test_size=0.1)
# Split the pipeline into transformers and a predictor

# (N.b. "Pipeline" is essentially the same as make_pipeline, but takes a dictionary input instead)

from sklearn.pipeline import Pipeline



# Get all but the last step

pipeline_transformers = Pipeline(pipeline.steps[:-1])

print("Transformers:", pipeline_transformers)



# Get only the last step

pipeline_predictor = Pipeline(pipeline.steps[-1:])

print("Predictor:", pipeline_predictor)



# Fit & Transform the training data

X_train_transformed = pipeline_transformers.fit_transform(X_train)

# Only "transform" the eval data, without fit, to ensure trasformations (e.g. one-hot columns) are the same

X_eval_transformed = pipeline_transformers.transform(X_eval)
# Prepare grid search parameter grid (simplified for example purposes - tune here!)

param_grid = {

    'xgbregressor__min_child_weight' : list(range(0,2,1))

}



# prepare to pass XGB params as a dictionary

xgb_params = {

    "xgbregressor__eval_metric" : metric ,

    "xgbregressor__eval_set" : [(X_eval_transformed,y_eval)] ,

    "xgbregressor__early_stopping_rounds" : 10,

    "xgbregressor__verbose" : False

}



# Prepare grid search object

from sklearn.model_selection import GridSearchCV

optimizer = GridSearchCV(pipeline_predictor, param_grid, cv=5, scoring = 'neg_mean_squared_log_error')



# do the grid search with early_stopping

print("Grid searching over parameter grid:", param_grid)

optimizer.fit(X_train_transformed, y_train, **xgb_params)



# print results

print("Best params:",optimizer.best_params_)

print("Best RMSLE:",sqrt(-optimizer.best_score_))
pipeline.set_params(

    xgbregressor__min_child_weight = 0.4,

    xgbregressor__colsample_bytree = 0.5,

    xgbregressor__n_estimators = 700

)
print("Checking score...")

from sklearn.model_selection import cross_val_score

SLE = cross_val_score(pipeline, X, y, cv=5,

                      scoring='neg_mean_squared_log_error' )

print(sqrt(-SLE.mean()))
# re-fit on 100% of training data

pipeline.fit(X,y)



# read in test data

test_data = pd.read_csv('../input/test.csv')

test_predictors = test_data.drop(['Id'], axis=1)



# apply pipeline

predictions =  pipeline.predict(test_predictors)



# store for submission

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})

my_submission.to_csv('submission.csv', index=False)