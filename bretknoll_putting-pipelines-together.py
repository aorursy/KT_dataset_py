## Put imports here
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf
from sklearn.model_selection import train_test_split  # split data test and train
from sklearn.metrics import mean_absolute_error # used to calculate model error rate
from sklearn.pipeline import Pipeline # using this to create pipelines
from sklearn.preprocessing import Imputer # imputer is used to fill in nan values
from sklearn.model_selection import GridSearchCV # using to find best parameters for transformers

# base class for pipeline are used to build custom transformers
from sklearn.base import BaseEstimator, TransformerMixin # used build custom classes for transforming
from sklearn.preprocessing import FunctionTransformer # not used used to transform dataframes in pipeline

#models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#import train data
train_data = pd.read_csv("../input/melb_data.csv")
print(train_data.columns)
print(train_data.info())
### common functions
def maeOutput(y_test, predictions):    
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions )}")
# need to identify different sets of columns 
## target, numeric, non_numeric
non_numeric_features= train_data.select_dtypes(include=['object']).columns
numeric_features= train_data.select_dtypes(exclude=['object']).columns
target = 'Price'

# split data test and train
X_train, X_test, y_train, y_test = train_test_split(train_data[numeric_features], train_data[target], random_state=0)
print(X_train.head())
# create a simple pipeline of a RandomForestRegressor
rfRegressorPipeline = Pipeline([ ('step1_imputer', Imputer()),
                         ('step2_regressor', RandomForestRegressor())
                    ])
# fit model
rfRegressorPipeline.fit(X_train, y_train)
predictions = rfRegressorPipeline.predict(X_test)
maeOutput(y_test, predictions)
# get params from pipeline steps
print(rfRegressorPipeline.get_params().keys())
# build a dictionary or a list of dictionaries
# the strategy parameter can take a value of mean, median and most_frequent
paramGrid = { 
            'step1_imputer__strategy': ['mean', 'median', 'most_frequent'],
            'step2_regressor__bootstrap': [True, False],
            'step2_regressor__warm_start': [True, False]
            }
# other params
#            'step2_regressor__random_state': [0],
#            'step2_regressor__max_leaf_nodes': [None, 68, 600, 700]

# GridSearchCV
## estimator is an estimator object, I'm guessing it can be a model or a pipeline
## param_grid is the dictionary from above
gridSearch = GridSearchCV(estimator=rfRegressorPipeline, param_grid=paramGrid, cv=3)
 
# Fit and tune model
gridSearch.fit(X=X_train, y=y_train)

# now what parameter values were the best
gridSearch.best_params_
# use refit to use the best params
gridSearch.refit
# run the X_test through the model 
preds = gridSearch.predict(X_test)
maeOutput(y_test, preds)