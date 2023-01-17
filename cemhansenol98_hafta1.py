import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from IPython.display import display

from sklearn import metrics

from sklearn.impute import SimpleImputer

from pandas.api.types import is_string_dtype

from pandas.api.types import is_numeric_dtype

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn import tree

from sklearn.ensemble import forest

from pdpbox import pdp

from plotnine import *
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
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
display_all(train_data.head())
train_data.info()
train_data.shape
display_all(train_data.isna().sum())
drop_col_names = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

train_data.drop(drop_col_names, axis=1, inplace=True)

test_data.drop(drop_col_names, axis=1, inplace=True)
train_data.describe(include='all')
train_data.LotFrontage = train_data.LotFrontage.replace('nan',np.nan)

test_data.LotFrontage = train_data.LotFrontage.replace('nan',np.nan)
cols = train_data.columns

dtype_mapping = dict(train_data.dtypes)

numeric_cols = [ c for c in cols if dtype_mapping[c] != 'object' ]

categoric_cols = [ c for c in cols if dtype_mapping[c] == 'object' ]
cols_t = test_data.columns

dtype_mapping_t = dict(test_data.dtypes)

numeric_cols_t = [ c for c in cols_t if dtype_mapping[c] != 'object' ]

categoric_cols_t = [ c for c in cols_t if dtype_mapping[c] == 'object' ]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(train_data[numeric_cols])

imputer_t = imputer.fit(test_data[numeric_cols_t])
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer2 = imputer2.fit(train_data[categoric_cols])

imputer2_t = imputer2.fit(test_data[categoric_cols_t])
drop_col2 = []

for i in train_data.columns:

    if train_data[i].isna().sum() != 0:

        drop_col2.append(i)

train_data.drop(drop_col2,axis=1,inplace=True)
drop_col2_t = []

for i in test_data.columns:

    if test_data[i].isna().sum() != 0:

        drop_col2_t.append(i)

test_data.drop(drop_col2,axis=1,inplace=True)
def train_cats(df):

    for n,c in df.items():

        if is_string_dtype(c):

            df[n] = c.astype("category").cat.as_ordered()
def apply_cats(df, train):

    for n, c in df.items():

        if train[n].dtype == "category":

            df[n] = pd.Categorical(c, categories = train[n].cat.categories, ordered = True)
train_cats(train_data)
train_cats(test_data)
train_data.MSZoning.cat.categories
def fix_missing(df, col, name):

    if is_numeric_dtype(col):

        if pd.isnull(col).sum:

            df[name+"_na"] = pd.isnull(col)

        df[name] = col.fillna(col.median())
def numericalize(df, col, name):

    if not is_numeric_dtype(col):

        df[name] = col.cat.codes+1
def proc_df(df, y_fld):

    

    y = df[y_fld].values

    df.drop([y_fld], axis = 1, inplace = True)

    

    for n, c in df.items():

        fix_missing(df, c, n)

        

    for n, c in df.items():

        numericalize(df, c, n)

    

    res = [df, y]

    

    return res
train_data_X, train_data_y = proc_df(train_data, 'SalePrice')
m = RandomForestRegressor(n_jobs=-1)

m.fit(train_data_X, train_data_y)

m.score(train_data_X,train_data_y)
def split_train_val(df,n): 

    return df[:n].copy(), df[n:].copy()
n_valid = 200

n_train = len(train_data_X) - n_valid

X_train, X_valid = split_train_val(train_data_X, n_train)

y_train, y_valid = split_train_val(train_data_y, n_train)
def print_score(m):

    

    print(f"RMSE of train set {mean_squared_error(m.predict(X_train), y_train, squared=False)}")

    print(f"RMSE of validation set {mean_squared_error(m.predict(X_valid), y_valid, squared=False)}")

    print(f"R^2 of train set {m.score(X_train, y_train)}")

    print(f"R^2 of validation set {m.score(X_valid, y_valid)}")

    print(f"Adj. R^2 of train set {r2_score(m.predict(X_train), y_train, multioutput='variance_weighted')}")

    print(f"Adj. R^2 of validation set {r2_score(m.predict(X_valid), y_valid, multioutput='variance_weighted')}")
m = RandomForestRegressor(n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)

t = m.fit(X_train, y_train)

print_score(t)
# Bu daha kötü oldu parametreleri sokmadan daha düzgün ayarladı.
estimator = t.estimators_[0]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=300)

tree.plot_tree(estimator, feature_names=train_data.columns,filled=True)
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=10, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
preds = np.array([t.predict(X_valid) for t in m.estimators_])

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])
m = RandomForestRegressor(n_estimators=10, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
preds = np.array([t.predict(X_valid) for t in m.estimators_])

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
preds = np.array([t.predict(X_valid) for t in m.estimators_])

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])
m = RandomForestRegressor(n_estimators=60, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
preds = np.array([t.predict(X_valid) for t in m.estimators_])

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])
def print_score(m):

    

    print(f"RMSE of train set {mean_squared_error(m.predict(X_train), y_train, squared=False)}")

    print(f"RMSE of validation set {mean_squared_error(m.predict(X_valid), y_valid, squared=False)}")

    print(f"R^2 of train set {m.score(X_train, y_train)}")

    print(f"R^2 of validation set {m.score(X_valid, y_valid)}")

    print(f"Adj. R^2 of train set {r2_score(m.predict(X_train), y_train, multioutput='variance_weighted')}")

    print(f"Adj. R^2 of validation set {r2_score(m.predict(X_valid), y_valid, multioutput='variance_weighted')}")

    if hasattr(m, "oob_score_"):

        print(f"OOB score: {m.oob_score_}")
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
def print_score(m):

    

    print(f"RMSE of train set {mean_squared_error(m.predict(X_train), y_train, squared=False)}")

    print(f"RMSE of validation set {mean_squared_error(m.predict(X_valid), y_valid, squared=False)}")

    print(f"R^2 of train set {m.score(X_train, y_train)}")

    print(f"R^2 of validation set {m.score(X_valid, y_valid)}")

    print(f"Adj. R^2 of train set {r2_score(m.predict(X_train), y_train, multioutput='variance_weighted')}")

    print(f"Adj. R^2 of validation set {r2_score(m.predict(X_valid), y_valid, multioutput='variance_weighted')}")
def set_rf_samples(n):

  

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n))
def reset_rf_samples():

   

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n_samples))
set_rf_samples(400)
m = RandomForestRegressor(n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=30, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
reset_rf_samples()
def print_score(m):

    

    print(f"RMSE of train set {mean_squared_error(m.predict(X_train), y_train, squared=False)}")

    print(f"RMSE of validation set {mean_squared_error(m.predict(X_valid), y_valid, squared=False)}")

    print(f"R^2 of train set {m.score(X_train, y_train)}")

    print(f"R^2 of validation set {m.score(X_valid, y_valid)}")

    print(f"Adj. R^2 of train set {r2_score(m.predict(X_train), y_train, multioutput='variance_weighted')}")

    print(f"Adj. R^2 of validation set {r2_score(m.predict(X_valid), y_valid, multioutput='variance_weighted')}")

    if hasattr(m, "oob_score_"):

        print(f"OOB score: {m.oob_score_}")
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=10, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt', n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='log2', n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.3, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
importances = m.feature_importances_

importances
features = train_data.columns

imp = pd.DataFrame({'Features': features, 'Importance': importances})

imp.head()
imp = imp.sort_values(by = 'Importance', ascending = False)

imp.head()
imp['Sum Importance'] = imp['Importance'].cumsum()

imp = imp.sort_values(by = 'Importance')

imp.head()
plt.figure(figsize=(8,8))

plt.barh(imp['Features'], imp['Importance'])

l1 = plt.axhline(len(imp) - (len(imp['Features'][imp['Sum Importance'] < 0.50]) + 1.5), linestyle='-.', color = 'r')

l2 = plt.axhline(len(imp) - (len(imp['Features'][imp['Sum Importance'] < 0.90]) + 1.5), linestyle='--', color = 'r')

l3 = plt.axhline(len(imp) - (len(imp['Features'][imp['Sum Importance'] < 0.99]) + 1.5), linestyle='-', color = 'r')

plt.legend(title = 'Cut-offs of acumulated importance', handles=(l1, l2, l3), labels = ('50%', '90%', '99%'))

plt.title('Feature importance in group assignment')