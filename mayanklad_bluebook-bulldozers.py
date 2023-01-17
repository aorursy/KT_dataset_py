# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train_val = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv',

                           low_memory=False,

                           parse_dates=['saledate'])



print(df_train_val.shape)
df_train_val.sort_values(by=["saledate"], inplace=True, ascending=True)

df_train_val.head(20)
print(df_train_val.info())
print(df_test.info())
df_train_val.drop('SalesID', axis=1, inplace=True)
removed_features = ['SalesID']

print(removed_features)
print(df_train_val.info())
df_train_val["saleYear"] = df_train_val.saledate.dt.year

df_train_val["saleMonth"] = df_train_val.saledate.dt.month

df_train_val["saleDay"] = df_train_val.saledate.dt.day

df_train_val["saleDayofweek"] = df_train_val.saledate.dt.dayofweek

df_train_val["saleDayofyear"] = df_train_val.saledate.dt.dayofyear

df_train_val.drop("saledate", axis=1, inplace=True)
for label, content in df_train_val.items():

    if pd.api.types.is_string_dtype(content):

        df_train_val[label] = content.astype('category').cat.as_ordered()
df_train_val.info()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(df_train_val['saleYear'][:1000], df_train_val['SalePrice'][:1000])
df_train_val.SalePrice.plot.hist()
df_train_val.head().T
for label, content in df_train_val.items():

    if 100*df_train_val[label].isna().sum()/len(df_train_val) > 70:

        removed_features.append(label)

        print(label,

              '{0:.2f}%'.format(100*df_train_val[label].isna().sum()/len(df_train_val)))
print(removed_features)
list(set(removed_features).intersection(set(df_train_val.columns)))
df_train_val.drop(

    list(set(removed_features).intersection(set(df_train_val.columns))),

    axis=1,

    inplace=True)
df_train_val.info()
for label, content in df_train_val.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isna(content).sum():

            df_train_val[label] = content.fillna(content.median())

    else:

        df_train_val[label] = pd.Categorical(content).codes+1
for label, content in df_train_val.items():

    if df_train_val[label].isna().sum():

        print(label,

              '{0:.2f}%'.format(100*df_train_val[label].isna().sum()/len(df_train_val)))
df_train_val.info()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error, mean_absolute_error
def rmsle(y_test, y_preds):

    return np.sqrt(mean_squared_log_error(y_test, y_preds))



def show_scores(model, X_train, y_train, valid=False, X_valid=None, y_valid=None):

    train_preds = model.predict(X_train)

    

    scores = dict()

    

    scores['Training MAE'] = mean_absolute_error(y_train, train_preds)

    scores['Training RMSLE'] = rmsle(y_train, train_preds)

    scores['Training R^2'] = model.score(X_train, y_train)

    

    if valid:

        val_preds = model.predict(X_valid)

        scores['Valid MAE'] = mean_absolute_error(y_valid, val_preds)

        scores['Valid RMSLE'] = rmsle(y_valid, val_preds)

        scores['Valid R^2'] = model.score(X_valid, y_valid)



    return scores
models = dict()
models['basic'] = RandomForestRegressor(n_jobs=-1)
models['basic'].fit(df_train_val.drop('SalePrice', axis=1), df_train_val.SalePrice)
show_scores(models['basic'],

            df_train_val.drop('SalePrice', axis=1),

            df_train_val.SalePrice)
df_train_val.saleYear.unique()
df_valid = df_train_val[df_train_val.saleYear == 2012]

df_train = df_train_val[df_train_val.saleYear != 2012]



df_train.shape, df_valid.shape
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice

X_valid, y_valid = df_valid.drop("SalePrice", axis=1), df_valid.SalePrice



X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
models['no_tuning'] = RandomForestRegressor(n_jobs=-1)
models['no_tuning'].fit(X_train, y_train)
show_scores(model=models['no_tuning'],

            X_train=X_train,

            y_train=y_train,

            valid=True,

            X_valid=X_valid,

            y_valid=y_valid)
from sklearn.model_selection import RandomizedSearchCV
rf_grid = {"n_estimators": np.arange(10, 100, 10),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2),

           "max_features": [0.5, 1, "sqrt", "auto"],

           "max_samples": [20000]}
%%time

rs_model = RandomizedSearchCV(RandomForestRegressor(),

                              param_distributions=rf_grid,

                              n_iter=20,

                              cv=5,

                              verbose=True)



rs_model.fit(X_train, y_train)
best_params = rs_model.best_params_

best_params
show_scores(model=rs_model,

            X_train=X_train,

            y_train=y_train,

            valid=True,

            X_valid=X_valid,

            y_valid=y_valid)
models['rs'] = RandomForestRegressor(n_jobs=-1,

                                     n_estimators=best_params['n_estimators'],

                                     min_samples_split=best_params['min_samples_split'],

                                     min_samples_leaf=best_params['min_samples_leaf'],

                                     max_features=best_params['max_features'],

                                     max_depth=best_params['max_depth'])

models['rs'].fit(X_train, y_train)
show_scores(model=models['rs'],

            X_train=X_train,

            y_train=y_train,

            valid=True,

            X_valid=X_valid,

            y_valid=y_valid)
df_test = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/Test.csv',

                      low_memory=False,

                      parse_dates=['saledate'])
df_test.head(10)
df_test.head().T
df_test['saleYear'] = df_test.saledate.dt.year

df_test['saleMonth'] = df_test.saledate.dt.month

df_test['saleDay'] = df_test.saledate.dt.day

df_test['saleDayofweek'] = df_test.saledate.dt.dayofweek

df_test['saleDayofyear'] = df_test.saledate.dt.dayofyear

df_test.drop('saledate', axis=1, inplace=True)
df_test.head().T
salesID = df_test.SalesID

salesID.head()
df_test.drop(

    list(set(removed_features).intersection(set(df_test.columns))),

    axis=1,

    inplace=True)
df_test.columns
df_test.isna().sum()/len(df_test)
for label, content in df_test.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isna(content).sum():

            df_test[label] = content.fillna(content.median())



    else:

        df_test[label] = pd.Categorical(content).codes + 1

        
df_test.isna().sum()/len(df_test)
test_preds = dict()

for label, model in models.items():

    test_preds[label] = model.predict(df_test)
for label, preds in test_preds.items():

    output = pd.DataFrame({'SalesID': salesID, 'SalePrice': preds})

    output.to_csv('my_submission_{}.csv'.format(label), index=False)



print('Your submission was successfully saved!')