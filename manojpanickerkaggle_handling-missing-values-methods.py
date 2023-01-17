import pandas as pd

import numpy as np
df = pd.read_csv('../input/melbourne-dataset/Melbourne_Dataset.csv')
df.head()
df.describe().transpose()
df.info()
## columns with missing values

missing_values_cols = [col for col in df.columns

                      if df[col].isnull().any()]

missing_values_cols

df_for_model = df.drop(missing_values_cols,axis=1)
## Categorize columns in Numerical and Categorical

numerical_cols = (df_for_model.dtypes != 'object')

numerical_cols = list(numerical_cols[numerical_cols].index)

categorical_cols = (df_for_model.dtypes == 'object')

categorical_cols = list(categorical_cols[categorical_cols].index)

numerical_cols , categorical_cols
df[numerical_cols].shape , df[categorical_cols].shape
numerical_cols.remove('Price')
numerical_cols
X = df_for_model[numerical_cols]

y = df_for_model['Price']
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error as MAE
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=42)
## function to return model

def return_model(train_X,val_X,train_y,val_y):

    model = RandomForestRegressor(n_estimators=100, random_state=1)

    model.fit(train_X,train_y)

    return model
## score_model

def score_model(val_X):

    predictions = model.predict(val_X)

    error = MAE(val_y,predictions)

    return 'Mean Absolute Error is :' + str(error)
model = return_model(train_X,val_X,train_y,val_y)

error = score_model(val_X)

error
df_test = pd.read_csv('../input/melbourne-datasettest/Melbourne_Dataset-Test.csv')

X_test = df_test[numerical_cols]
predictions = model.predict(X_test)

df_final = pd.DataFrame({'X:':df_test.Price,

                         'y:':predictions})

df_final
from sklearn.impute import SimpleImputer

#     - If "mean", then replace missing values using the mean along

#       each column. Can only be used with numeric data.

#     - If "median", then replace missing values using the median along

#       each column. Can only be used with numeric data.

#     - If "most_frequent", then replace missing using the most frequent

#       value along each column. Can be used with strings or numeric data.

#     - If "constant", then replace missing values with fill_value. Can be

#       used with strings or numeric data.
imputer_mean = SimpleImputer()
## Categorize columns in Numerical and Categorical

numerical_cols = (df.dtypes != 'object')

numerical_cols = list(numerical_cols[numerical_cols].index)
numerical_cols.remove('Price')
categorical_cols = (df.dtypes == 'object')

categorical_cols = list(categorical_cols[categorical_cols].index)
X = df[numerical_cols]

y = df['Price']
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=42)
df_iputed_cols_train = pd.DataFrame(imputer_mean.fit_transform(train_X)) 
df_iputed_cols_val = pd.DataFrame(imputer_mean.transform(val_X)) 
df_iputed_cols_train.columns = train_X.columns

df_iputed_cols_val.columns = val_X.columns
df_iputed_cols_train.shape
df_iputed_cols_val.shape
model = return_model(df_iputed_cols_train,df_iputed_cols_val,train_y,val_y)

error = score_model(df_iputed_cols_val)

error
X_test = df_test[numerical_cols]

df_iputed_cols_test = pd.DataFrame(imputer_mean.transform(X_test)) 

predictions = model.predict(df_iputed_cols_test)
df_final = pd.DataFrame({'X:':df_test.Price,

                         'y:':predictions})

df_final