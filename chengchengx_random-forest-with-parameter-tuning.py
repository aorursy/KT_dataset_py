import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from pandas.api.types import is_string_dtype, is_numeric_dtype

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt

import math
df_raw = pd.read_csv('../input/Automobile_data.csv', low_memory=False)
df_raw.shape
df_raw[:2]
# df_raw.columns = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors',\

#                  'body-style','drive-wheels','engine-location','wheel-base','length','width',\

#                  'height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system',\

#                  'bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg',\

#                  'price']
df_raw[:5]
# convert '?' to None

df_raw = df_raw.replace('?', np.nan)
# Extract all string-type columns

cols_str = []

for col in df_raw:

    if is_string_dtype(df_raw[col]):

        cols_str.append(col)

print(cols_str)
# convert following columns to continuous variables based on data description

# normalized-losses, bore, stroke, horsepower, peak-rpm, price

cols = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]

for col in cols:

    df_raw[col] = pd.to_numeric(df_raw[col], errors='raise')
for col in df_raw:

    if is_string_dtype(df_raw[col]):

        df_raw[col] = df_raw[col].astype('category').cat.as_ordered()
for col in df_raw:

    if is_numeric_dtype(df_raw[col]):

        col_vals = df_raw[col]

        if sum(col_vals.isnull()) != 0:

            df_raw[col+'_na'] = col_vals.isnull()

            df_raw[col] = col_vals.fillna(col_vals.median())
for col in df_raw:

    if str(df_raw[col].dtype) == "category":

        df_raw[col] = df_raw[col].cat.codes + 1
df_raw.shape
X = df_raw.drop('price', axis=1)

y = df_raw['price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 99)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
m = RandomForestRegressor(n_jobs=-1)

m.fit(X_train, y_train)
def rmse(preds, actuals):

    return math.sqrt(((preds-actuals)**2).mean())
[rmse(m.predict(X_train), y_train),rmse(m.predict(X_val), y_val),m.score(X_train, y_train), m.score(X_val, y_val)]
X = df_raw.drop("symboling", axis=1)

y = df_raw["symboling"].astype('category')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 99)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
m = RandomForestClassifier(n_jobs=-1)

m.fit(X_train, y_train)
print(m.score(X_train, y_train))

print(m.score(X_val, y_val))
# Tune three parameters: n_estimators, min_samepls_leaf, and max_features

# It might take some 

numOfestimators = [1,5,10,15,20,25,30]

numOfleafs = [1, 3, 5, 10, 25]

numOffeatures = np.arange(0.1, 1.1, 0.1)

best_result = []

for numOfestimator in numOfestimators:

    for numOfleaf in numOfleafs:

        for numOffeature in numOffeatures:  

            result = [numOfestimator, numOfleaf, numOffeature]

            m = RandomForestClassifier(n_jobs=-1, n_estimators=numOfestimator,\

                                    min_samples_leaf=numOfleaf,\

                                    max_features=numOffeature)

            # print(result)

            m.fit(X_train, y_train)

            result.append(m.score(X_train, y_train))

            result.append(m.score(X_val, y_val))

            if len(best_result) == 0: best_result = result

            elif best_result[4] < result[4]: 

                print(result)

                best_result = result

print(best_result)