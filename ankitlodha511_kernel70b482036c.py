# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

X_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col ='Id')

X_test_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col ='Id')

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
X_full.head()


y = X_full.SalePrice

X_full.drop(['SalePrice'], axis = 1, inplace = True)



categorical_columns = [col for col in X_full.columns

                      if X_full[col].nunique()<10

                      and X_full[col].dtypes == 'object']



numerical_columns = [col for col in X_full.columns

                    if X_full[col].dtypes in (['int64', 'float64'])]

my_cols = categorical_columns + numerical_columns

X = X_full[my_cols].copy()
from sklearn.model_selection import train_test_split

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis = 1, inplace = True)



X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,

                                                               train_size = 0.8, test_size = 0.2, 

                                                                random_state = 0)

categorical_columns = [col for col in X_train_full.columns

                      if X_train_full[col].nunique()<10

                      and X_train_full[col].dtypes == 'object']



numerical_columns = [col for col in X_train_full.columns

                    if X_train_full[col].dtypes in (['int64', 'float64'])]

my_cols = categorical_columns + numerical_columns

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
from sklearn.model_selection import train_test_split

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis = 1, inplace = True)



X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,

                                                               train_size = 0.8, test_size = 0.2, 

                                                                random_state = 0)



numerical_columns = [col for col in X_train_full.columns

                    if X_train_full[col].dtypes in (['int64', 'float64'])]

my_cols = numerical_columns

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()

print(X_train.columns)

print(X_valid.columns)

print(X_test.columns)
min_train = X_train.min()

range_train = (X_train - min_train).max()

X_train_scaled = (X_train - min_train)/range_train



min_valid = X_valid.min()

range_valid = (X_valid - min_valid).max()

X_valid_scaled = (X_valid - min_valid)/range_valid



min_test = X_test.min()

range_test = (X_test - min_test).max()

X_test_scaled = (X_test - min_test)/range_test

X.describe()
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score



numerical_transformer = SimpleImputer(strategy = 'mean')

categorical_transformer = Pipeline(

    steps = [

        ('imputer', SimpleImputer(strategy = 'most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

    ])



preprocessor = ColumnTransformer(

    transformers =[

        ('num', numerical_transformer, numerical_columns),

        ('cat', categorical_transformer, categorical_columns)

    ])





from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score



numerical_transformer = SimpleImputer(strategy = 'mean')





preprocessor = ColumnTransformer(

    transformers =[

        ('num', numerical_transformer, numerical_columns)

    ])





from xgboost import XGBRegressor

model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)

from xgboost import XGBRegressor

my_pipeline = Pipeline(

    steps=[

        ('preprocessor', preprocessor),

        ('XGB', XGBRegressor())

    ])
from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline, X, y, cv = 5,

                             scoring='neg_mean_absolute_error')
print("Average MAE score (across experiments):")

print(scores.mean())
param_grid = {

    "XGB__n_estimators": [10, 50, 100, 500],

    "XGB__learning_rate": [0.1, 0.5, 1],

}



from sklearn.model_selection import GridSearchCV

searchCV = GridSearchCV(my_pipeline, cv=5, param_grid=param_grid,refit = True,verbose = 4)
searchCV.fit(X_train,y_train)
searchCV.best_params_
from sklearn.metrics import mean_absolute_error

pred_valid = searchCV.predict(X_valid)

mae = mean_absolute_error(pred_valid, y_valid)

print(mae)
preds_test = searchCV.predict(X_test)

print(preds_test)
output = pd.DataFrame({'Id': X_test.index,

                      'SalePrice' : preds_test})

output.to_csv('submission.csv', index=False)
#from sklearn.model_selection import GridSearchCV

#grid_param_pipe = {'XGB__n_estimators': [1000],

 #                  'XGB__max_depth': [3],

 #                     'XGB__reg_alpha': [0.1],

 #                     'XGB__reg_lambda': [0.1]}

 #  grid_search_pipe = GridSearchCV(my_pipeline,

 #                                  param_grid=grid_param_pipe,

 #                                  scoring="neg_mean_squared_error",

 #                                  cv=5,

 #                                  n_jobs=5,

 #                                  verbose=3)

 #  grid_search_pipe.fit_transform(X_train, y_train, XGB__early_stopping_rounds=10, XGB__eval_metric="rmse", XGB__eval_set=[[X_valid, y_valid]])
my_pipeline.fit(X_train, y_train,

               early_stopping_rounds =5,

               eval_set = [X_valid, y_valid],

               verbose = False)
fit_params = {"eval_set": [(X_valid, y_valid)], 

              "early_stopping_rounds": 10, 

              "verbose": False}

searchCV.fit(X_train, y_train,

               XGBRegressor__fit_params = fit_params)
my_pipeline.fit(X_train, y_train)
grid.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, accuracy_score

preds = my_pipeline.predict(X_valid)

mae = mean_absolute_error(y_valid,preds)

#acc = accuracy_score(y_valid,preds)

print(mae)

#print(acc)
preds_test = my_pipeline.predict(X_test)

print(preds_test)
output = pd.DataFrame({'Id': X_test.index,

                      'SalePrice' : preds_test})

output.to_csv('submission.csv', index=False)