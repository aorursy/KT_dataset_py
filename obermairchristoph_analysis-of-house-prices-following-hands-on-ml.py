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
housing_unadjusted = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

housing = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

housing.head()
housing.columns.values 
corr_matrix = housing.corr()

corr_matrix['SalePrice'].sort_values(ascending=False).tail()
#Adding more important features

housing['TotalSF'] = housing['TotalBsmtSF'] + housing['1stFlrSF'] + housing['2ndFlrSF']
#Deleting outliers

housing = housing.drop(housing[(housing['GrLivArea']>4000) & (housing['SalePrice']<300000)].index)

housing = housing.drop(housing[(housing['TotalBsmtSF']>4000) & (housing['SalePrice']<300000)].index)
housing_data = housing.drop(['Id',"SalePrice"], axis=1)

housing_labels = housing["SalePrice"]
housing_val = housing_data.loc[:,housing_data.dtypes!=np.object] # Value Data

housing_val.head()
housing_cat = housing_data.loc[:,housing_data.dtypes==np.object] # Categorical Data

housing_cat = housing_cat.fillna('missing')

housing_cat.head()
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn_pandas import CategoricalImputer





num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])



cat_pipeline = Pipeline([

        ('cat_imputer',  CategoricalImputer(strategy='constant',fill_value='missing')),

        ('one_hot',  OneHotEncoder(handle_unknown='ignore')),

    ])

full_pipeline = ColumnTransformer([

        ("num", num_pipeline, list(housing_val)),

        ("cat", cat_pipeline, list(housing_cat)),

    ])



housing_prepared = full_pipeline.fit_transform(housing_data)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score



gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=500, learning_rate=0.2, random_state=42)

gbrt.fit(housing_prepared, housing_labels)



gbrt_scores = cross_val_score(gbrt, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=5)

np.sqrt(-gbrt_scores).mean()/housing_unadjusted['SalePrice'].mean()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



forest_reg = RandomForestRegressor(bootstrap=False, max_features = 6,n_estimators=30, random_state=42)

#forest_reg = RandomForestRegressor(bootstrap=False, max_features = 79,n_estimators=500, random_state=42)

#forest_reg.fit(housing_prepared, housing_labels)



forest_score = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=5)



np.sqrt(-forest_score).mean()/housing_unadjusted['SalePrice'].mean()
from sklearn.model_selection import GridSearchCV



param_grid = [

    # try 12 (3×4) combinations of hyperparameters

    #{'n_estimators': [50], 'max_features': [10]},

    # then try 6 (2×3) combinations with bootstrap set as False

    {'bootstrap': [False], 'n_estimators': [50,500], 'max_features': [10,79]},

  ]



forest_reg = RandomForestRegressor(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error',

                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test_in = test.drop("Id",axis =1)

test_in['TotalSF'] = test_in['TotalBsmtSF'] + test_in['1stFlrSF'] + test_in['2ndFlrSF']
test_prep = full_pipeline.transform(test_in)

housing_predictions = gbrt.predict(test_prep)
result = test[['Id']]

result['SalePrice'] = housing_predictions

result
result.to_csv("/kaggle/working/mysubmission.csv", index = False)