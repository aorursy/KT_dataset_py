# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from datetime import datetime



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# Load data

data_dir = '/kaggle/input/home-data-for-ml-course/'

train = pd.read_csv(data_dir + 'train.csv', index_col='Id')

test = pd.read_csv(data_dir + 'test.csv', index_col='Id')



print("Train set size:", train.shape)

print("Test set size:", test.shape)

print('START data processing', datetime.now())



# Remove rows without target value

train = train.dropna(subset=['SalePrice'])



# Separate train target from features

y_train = train['SalePrice']

X_train = train.drop(columns=['SalePrice'])

X_test = test
print('Train dtypes: {}'.format(X_train.dtypes.unique().tolist()))

print('Test dtypes: {}'.format(X_test.dtypes.unique().tolist()))
# Leave only number and categorial features with low ordinality

num_columns = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

cat_columns = [col for col in X_train.columns if X_train[col].dtype in ['object'] and X_train[col].nunique() < 10]



all_columns = num_columns + cat_columns



X_train = X_train[all_columns].copy()

X_test = X_test[all_columns].copy()
# Split the data

# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

print('Train size: {}'.format(X_train.shape))

print('Test size: {}'.format(X_test.shape))
print('START ML', datetime.now(), )
from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.model_selection import GridSearchCV, cross_val_score

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, make_scorer



numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(transformers=[

    ('num', numeric_transformer, num_columns),

    ('cat', categorical_transformer, cat_columns)

])



# Prediction method

model = XGBRegressor()



# Define full prediction pipeline

final_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('xgbr', model)

])



parameters = {

#     'xgbr__n_estimators': [500], # [500, 1000, 1500],

#     'xgbr__learning_rate': [0.1], # [0.05, 0.1, 0.3],

#     'xgbr__max_depth': [3], # [3, 4, 5, 6, 7, 8, 9],

#     'xgbr__subsample': [0.7], # [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

#     'xgbr__colsample_bytree': [1.0], # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

#     'xgbr__reg_lambda': [0.03], # [0, 0.01, 0.02, 0.03, 0.04],

    'xgbr__learning_rate': [0.01],

    'xgbr__n_estimators': [3460],

    'xgbr__max_depth': [3],

    'xgbr__min_child_weight': [0],

    'xgbr__gamma': [0],

    'xgbr__subsample': [0.7],

    'xgbr__colsample_bytree': [0.7],

    'xgbr__objective': ['reg:linear'],

    'xgbr__nthread': [-1],

    'xgbr__scale_pos_weight': [1],

    'xgbr__seed': [0],

    'xgbr__reg_alpha': [0.00006],

    'xgbr__reg_lambda': [1]

}



neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error, greater_is_better=False)



clf = GridSearchCV(

    estimator=final_pipeline,

    param_grid=parameters,

    scoring=neg_mean_absolute_error_scorer,

    cv=3,

    verbose=2

)



clf.fit(X_train, y_train);
print("Best score={}\n".format(-1 * clf.best_score_))

print('Best params: {}\n'.format(clf.best_params_))

# print(clf.best_estimator_)

# print('all_params: {}\n'.format(clf.cv_results_['params']))

# print('mean_fit_time: {}\n'.format(clf.cv_results_['mean_fit_time']))

# print('mean_test_score: {}\n'.format(clf.cv_results_['mean_test_score']))

# print('rank_test_score: {}'.format(clf.cv_results_['rank_test_score']))

# print(clf.cv_results_)



# print('Cross validation score: {}'.format(cost_function()))
# Train model

print('Predict submission', datetime.now())

predictions = clf.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})

output.to_csv('submission.csv', index=False)



print('Saved submission', datetime.now())