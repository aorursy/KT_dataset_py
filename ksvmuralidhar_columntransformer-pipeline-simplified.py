from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

import pandas as pd

import numpy as np

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import MinMaxScaler
df = pd.DataFrame({"col1":["a","b","a","c"],"col2":["a","b","a","c"]})

print(df)
ohe = OneHotEncoder()

df = ohe.fit_transform(df)

print(df.toarray())
df = pd.DataFrame({"col1":["a","b","a","c"],"col2":["a","b","a","c"]})

print(df)
ct = ColumnTransformer(transformers = [('ohe_col1',OneHotEncoder(),[0]),

                                     ('ord_col2',OrdinalEncoder(),[1])])

df = ct.fit_transform(df)

print(df)
df = pd.DataFrame({"col1":["a","b","a","c"],"col2":["a","b","a","c"]})

print(df)
ct = ColumnTransformer(transformers = [('ohe_col1',OneHotEncoder(),[0])])

df = ct.fit_transform(df)

print(df)
df = pd.DataFrame({"col1":["a","b","a","c"],"col2":["a","b","a","c"]})

print(df)
ct = ColumnTransformer(transformers = [('ohe_col1',OneHotEncoder(),[0])],remainder="passthrough")

df = ct.fit_transform(df)

print(df)
df = pd.DataFrame({"col1":["a","b",np.nan,"c"],"col2":[1,2,np.nan,5]})

print(df)
ct = ColumnTransformer(transformers=[('mode_impute1',SimpleImputer(strategy="most_frequent"),[0]),

                                    ('one_hot_encode1',OneHotEncoder(),[0]),

                                    ('median_impute2',SimpleImputer(strategy="median"),[0])])

df = ct.fit_transform(df)

print(df)
df = pd.DataFrame({"col1":["a","b","d","c"],"col2":[1,2,np.nan,5]})

print(df)
ct = ColumnTransformer(transformers=[('ord_encode1',OrdinalEncoder(),[0]),

                                     ('scale1',MinMaxScaler(),[0]),

                                    ('median_impute2',SimpleImputer(strategy="median"),[0])])

df = ct.fit_transform(df)

print(df)
df = pd.DataFrame({"col1":[1,2,np.nan,3],"col2":[1,np.nan,1,5]})

print(df)
pipe = Pipeline(steps=[('imputation',SimpleImputer(strategy="median")),

                      ("scaling",MinMaxScaler())])

df = pipe.fit_transform(df)

print(df)
df = pd.DataFrame({"col1":[1,2,np.nan,3],"col2":[1,np.nan,1,5]})

print(df)
df = SimpleImputer(strategy="median").fit_transform(df)

df = MinMaxScaler().fit_transform(df)

print(df)
df = pd.DataFrame({"col1":["a","b",np.nan,"a"],"col2":[1,2,np.nan,5]})

print(df)
col1_pipe = Pipeline(steps=[('mode_col1',SimpleImputer(strategy="most_frequent")),

                           ("one_hot_encode",OneHotEncoder())])



col_transform = ColumnTransformer(transformers=[("col1",col1_pipe,[0]),

                                               ("col2",SimpleImputer(strategy="median"),[1])])



df = col_transform.fit_transform(df)

print(df)