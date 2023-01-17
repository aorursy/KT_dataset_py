# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the test data

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
# Extract the target variable from the training data

train_y = train_data.SalePrice

train_data.drop('SalePrice', axis=1, inplace=True)
# The final competition score is based on the MSE between the log of the test predictions and the log of the true SalePrice.

# With that in mind, always train to fit logy.

train_logy = np.log(train_y)

# Predictions will need to be of y, however, so for the final test submission, take the exponent of its output.
# Separate the Id column from the predictive features

train_X = train_data.drop('Id', axis=1)

test_X = test_data.drop('Id', axis=1)
train_X_numeric = train_X.select_dtypes(include=[np.number]).drop('MSSubClass', axis=1)

train_X_categorical = train_X.drop(train_X_numeric.columns, axis=1)



num_cols = train_X_numeric.columns

cat_cols = train_X_categorical.columns
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.impute import SimpleImputer



num_pipeline = Pipeline([

    ('num_imputer', SimpleImputer(strategy='median')),

    ('num_scaler', RobustScaler())

])



cat_pipeline = Pipeline([

    ('cat_nan_filler', SimpleImputer(strategy='constant', fill_value='not_in_data')),

    ('cat_onehot', OneHotEncoder(handle_unknown='ignore'))

])



# For XGBoostRegressor

minimal_preprocessor_pipeline = ColumnTransformer([

    ('num_pipeline', 'passthrough', num_cols),

    ('cat_pipeline', cat_pipeline, cat_cols)

])



# For all other models

preprocessor_pipeline = ColumnTransformer([

    ('num_pipeline', num_pipeline, num_cols),

    ('cat_pipeline', cat_pipeline, cat_cols)

])
# Best models from model examination

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import ElasticNet



best_models = {'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),

               'ExtraTrees': ExtraTreesRegressor(n_estimators=100, criterion='mse', min_samples_leaf=2)}
# Fit

models = []

model_names = []



for model_name in best_models:

    model = Pipeline([

        ('preprocessor', preprocessor_pipeline),

        ('actual_model', best_models[model_name])

    ])

    model.fit(train_X, train_logy)

    models.append(model)

    model_names.append(model_name)
from xgboost import XGBRegressor



xgb_regressor = XGBRegressor(learning_rate=0.1, max_depth=2, n_estimators=650)
model = Pipeline([

    ('preprocessor', minimal_preprocessor_pipeline),

    ('actual_model', xgb_regressor)

])

model.fit(train_X, train_logy)

models.append(model)

model_names.append('XGBoost')
train_predictions_allmodels = []

test_predictions_allmodels = []



for model in models:

    train_predictions = model.predict(train_X)

    train_predictions_allmodels.append(train_predictions)

    test_predictions = model.predict(test_X)

    test_predictions_allmodels.append(test_predictions)
model_names.append('average')

# This line must be run only AFTER all the single model predictions have been stored

averaged_train_predictions = np.stack(train_predictions_allmodels).mean(axis=0)

train_predictions_allmodels.append(averaged_train_predictions)

averaged_test_predictions = np.stack(test_predictions_allmodels).mean(axis=0)

test_predictions_allmodels.append(averaged_test_predictions)
from sklearn.linear_model import LinearRegression

from mlxtend.regressor import StackingCVRegressor



regressors_for_stacking = []

for model_name in best_models:

    model = best_models[model_name]

    regressors_for_stacking.append(model)

regressors_for_stacking.append(xgb_regressor)

    

train_X_for_stacking = preprocessor_pipeline.fit_transform(train_X)

# Not the ideal preprocessor pipeline for XGBoost,

#  but using different preprocessors with StackingCVRegressor is a problem I have yet to solve

    

model_names.append('stacked')

stack_regressor = StackingCVRegressor(regressors=regressors_for_stacking, meta_regressor=LinearRegression())

stack_regressor.fit(train_X_for_stacking, train_logy)
stack_train_predictions = stack_regressor.predict(train_X_for_stacking)

train_predictions_allmodels.append(stack_train_predictions)

stack_test_predictions = stack_regressor.predict(preprocessor_pipeline.transform(test_X))

test_predictions_allmodels.append(stack_test_predictions)
import matplotlib.pyplot as plt



def plot_train_predictions(train_targets, train_predictions, model_name):

    plt.scatter(train_targets, train_targets, label='SalePrice')

    plt.scatter(train_targets, train_predictions, label=model_name)

    plt.xlabel('Actual log(house price)')

    plt.ylabel('Predicted log(house price)')

    plt.legend()

    plt.show()
for idx in range(len(model_names)):

    plot_train_predictions(train_logy, train_predictions_allmodels[idx], model_names[idx])
## Save predictions in format used for competition scoring

for idx in range(len(model_names)):

    log_model_predictions = test_predictions_allmodels[idx]

    model_predictions = np.exp(log_model_predictions)

    output = pd.DataFrame({'Id': test_data.Id,

                           'SalePrice': model_predictions})

    output.to_csv('submission_{model_name}.csv'.format(model_name=model_names[idx]), index=False)