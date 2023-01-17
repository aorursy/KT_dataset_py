import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

import category_encoders as ce

test_filepath = '../input/tmdb-box-office-prediction/test.csv'

train_filepath = '../input/tmdb-box-office-prediction/train.csv'
test_set = pd.read_csv(test_filepath)

train_data = pd.read_csv(train_filepath)

train_data = train_data.replace(0,np.nan)
test_set.head()
test_set.columns.drop(["belongs_to_collection", "id", "tagline", "imdb_id","homepage", "poster_path", "original_title", "overview","crew", "status" ])
y = train_data.revenue

train_set = train_data.drop(["revenue"], axis=1)
X_train, X_val, y_train, y_val = train_test_split(train_set,y,test_size=0.2, random_state=42)
print(train_set.shape,X_train.shape,X_val.shape,y_train.shape,y_val.shape)
%matplotlib inline

train_set.hist(bins=50, figsize =(20,15))
corr = train_data.corr()

corr["revenue"].sort_values(ascending=False)
numeric_features =["budget", "popularity", "runtime"]

pd.plotting.scatter_matrix(train_data[numeric_features], figsize=(12,8))
train_data.plot(kind="scatter", x = "runtime", y = "revenue")
X_test = test_set.copy()
cols_with_missing = [cols for cols in X_train.columns if X_train[cols].isnull().any()]
missing_val_count_by_column = (X_train[numeric_features].isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column >0])
X_train["budget"].skew() #high skew
X_train["runtime"].skew() #high skew
X_train[X_train._get_numeric_data().columns].plot(kind="box", figsize = (10,10))
X_train["budget"].plot(kind="box", figsize = (10,10))
# find number of unique entries



s= train_set.dtypes == "object"

object_cols = list(s[s].index)



for col in object_cols:

    print(str(col) +": " + str(train_set[col].nunique()))
X_train[object_cols].nunique()
# if not using cross-validation



# def get_mae(max_leaf_nodes,X_train, X_val, y_train, y_val):

#     model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state = 42)

#     my_pipeline = Pipeline(steps = [("preprocessor", preprocessor), ("model", model)])

#     my_pipeline.fit(X_train,y_train)

#     val_predictions = my_pipeline.predict(X_val)

#     mae = mean_absolute_error(val_predictions,y_val)

#     print(mae)
object_cols = ["genres", "original_language", 'production_companies',

 'production_countries','spoken_languages','Keywords']
numerical_imputer = SimpleImputer(strategy="median")

imputed_numerical_X_train = pd.DataFrame(numerical_imputer.fit_transform(X_train[numeric_features]))

imputed_numerical_X_val = pd.DataFrame(numerical_imputer.transform(X_val[numeric_features]))

imputed_numerical_X_train.columns = numeric_features

imputed_numerical_X_val.columns = numeric_features

imputed_numerical_X_test = pd.DataFrame(numerical_imputer.fit_transform(X_test[numeric_features]))

imputed_numerical_X_test.columns = numeric_features

categorical_imputer = SimpleImputer(strategy="most_frequent")

imputed_categorical_X_train = pd.DataFrame(categorical_imputer.fit_transform(X_train[object_cols]))

imputed_categorical_X_val = pd.DataFrame(categorical_imputer.transform(X_val[object_cols]))

imputed_categorical_X_train.columns = object_cols

imputed_categorical_X_val.columns = object_cols

imputed_categorical_X_test = pd.DataFrame(categorical_imputer.fit_transform(X_test[object_cols]))

imputed_categorical_X_test.columns = object_cols

count_enc = ce.CountEncoder(cols = object_cols)

count_encoded_train = count_enc.fit_transform(imputed_categorical_X_train)

count_encoded_val = count_enc.transform(imputed_categorical_X_val)

count_encoded_test = count_enc.fit_transform(imputed_categorical_X_test)
X_train = pd.concat([imputed_numerical_X_train,count_encoded_train], axis= 1)

X_val = pd.concat([imputed_numerical_X_val,count_encoded_val], axis= 1)

X_test = pd.concat([imputed_numerical_X_test,count_encoded_test], axis= 1)
X_test
XGBRmodel = XGBRegressor(random_state = 42)
param_grid = {

        "xgbrg__n_estimators": [10,25, 50, 100, 150, 200, 300,400],

    "xgbrg__learning_rate": [0.01,0.05,0.1, 0.75, 1]

}
# fit_params = {"xgbrg__eval_set": [(X_val,y_val)], 

#               "xgbrg__early_stopping_rounds": 5, 

#               "xgbrg__verbose": False   

# }
grid_search = GridSearchCV(XGBRmodel, param_grid,scoring="neg_mean_squared_error", cv=5, n_jobs=1 )
grid_search.fit(X_train, y_train, early_stopping_rounds=5,           eval_set=[(X_val, y_val)],           verbose=True) #
# grid_search.best_estimator_
test_predictions = grid_search.predict(X_test)
output = pd.DataFrame({"id": test_set.id,

                      "revenue":test_predictions})
output.to_csv("submission.csv", index=False)