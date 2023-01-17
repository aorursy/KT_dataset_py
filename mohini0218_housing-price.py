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
input_dir = "/kaggle/input/house-prices-advanced-regression-techniques/"
df_train = pd.read_csv(input_dir+"train.csv",index_col=0)
df_train.head()
X = df_train.drop("SalePrice", axis=1)

y = df_train["SalePrice"]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
categorical_cols = [col for col in df_train.columns if df_train[col].nunique() < 10 and df_train[col].dtype=="object"]

numerical_cols = [col for col in categorical_cols if col not in categorical_cols]



my_cols = categorical_cols + numerical_cols

X_train = X_train[my_cols].copy()

X_test = X_test[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder





numerical_transformer = SimpleImputer(strategy='constant')





categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])





preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
prepared_data = preprocessor.fit_transform(X_train)

prepared_data.toarray()
from sklearn.ensemble import RandomForestRegressor



rnd = RandomForestRegressor(n_estimators=100)
from sklearn.metrics import mean_squared_error as mse



my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),('model',rnd)])
my_pipeline.fit(X_train,y_train)
preds = my_pipeline.predict(X_test)
from sklearn.metrics import mean_absolute_error



mean_absolute_error(y_test, preds)


import numpy as np





np.sqrt(mse(y_test, preds))
df_test = pd.read_csv(input_dir+"train.csv",index_col=0)
df_test.head()
X = df_test.drop("SalePrice", axis=1)

y = df_test["SalePrice"]



preds_test = my_pipeline.predict(X)
val = mse(preds_test, y)

val
np.sqrt(val)
preds_test
df_sub = pd.read_csv(input_dir+"sample_submission.csv")
df_sub
len(preds_test)
len(df_sub)
df_sub = df_sub.loc[1:]

df_sub
my_pipeline.score(X_test, y_test)