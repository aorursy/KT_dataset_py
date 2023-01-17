import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import forest

import math
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
PATH = '/kaggle/input/edmonton-property-assessment-data-2020/property_2020.csv'
df_raw = pd.read_csv(PATH)
df_raw.shape
df_raw.head(1)
df_raw["Assessed Value"] = df_raw["Assessed Value"].replace(0, 1)
df_raw["Assessed Value"] = np.log(df_raw["Assessed Value"])
df = df_raw.copy()
# check cardinality
for i in df.columns:
    if df[i].dtype.name == 'object':
        print(i, df[i].nunique(), '[OBJECT]')
    else: 
        print(i, df[i].nunique())
# change column dtypes string, change to cat
for label, col in df.items():
    if label in ['Suite', 'Street Name', 'Neighbourhood']:
        df[label] = col.astype('category').cat.as_ordered()
# check cardinality of categories
for i in df.columns:
    if df[i].dtype.name == 'category':
        print(i, df[i].nunique())
hot_columns = ['Assessment Class', 'Ward', 'Garage']
df = pd.get_dummies(df, columns=hot_columns, dummy_na=True);
df.head()
df.dtypes[0:9]
for i in df.columns:
    if df[i].isna().sum() > 0:
        print(i, df[i].isna().sum(), '\t', 'min value:', df[i].min(), 'max value:', df[i].max())
# turn to codes
cats = ["Suite", "Street Name", "Neighbourhood"]
for name in cats:
    df[name] = pd.Categorical(df[name]).codes+1
df["House Number"] = df["House Number"].fillna(0)
df["Neighbourhood ID"] = df["Neighbourhood ID"].fillna(0)
for i in df.columns:
    if df[i].isna().sum() > 0:
        print(i, df[i].isna().sum())
df.head()
df_y = df["Assessed Value"]
df_X = df.drop("Assessed Value", axis=1)
df_X.head()
X_train, X_valid, y_train, y_valid = train_test_split(df_X, df_y, test_size = 0.3)
y_train.isna().sum()
# Activate Random forest tree subsetting
# from the old fastai 0.7 library*
# https://github.com/fastai/fastai/blob/master/old/fastai/structured.py
def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))


set_rf_samples(20000)
# Score printer from fastML lesson 1
# https://github.com/fastai/fastai/blob/master/courses/ml1/lesson1-rf.ipynb
def rmse(x,y): 
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print('rmse:[ Training  |  Validation ]  |  score: [Train  |  Valid ] ')
    print(res)
m = RandomForestRegressor(n_estimators=100, max_features=0.5, n_jobs=-1)
%time m.fit(X_train, y_train)
#print_score(m)
print_score(m)
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)



fi = rf_feat_importance(m, df_X); fi[:12]
fi.plot('cols', 'imp', figsize=(10, 6), legend=False)
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12, 10), legend=False)

plot_fi(fi[:25]);
to_keep = fi[fi.imp>0.0025].cols; len(to_keep)
df_keep = df[to_keep].copy()
X_train, X_valid, y_train, y_valid = train_test_split(df_keep, df_y, test_size=0.25, random_state=42)
m = RandomForestRegressor(n_estimators=60, min_samples_leaf=10, n_jobs=-1, max_features=math.log(2), oob_score=False)
%time m.fit(X_train, y_train)
%time print_score(m)
fi = rf_feat_importance(m, df_keep)
plot_fi(fi);
import matplotlib.pyplot as plt
house_num = df_keep["House Number"]
price = df_y

fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(house_num[::10], price[::10], alpha=0.3)

ax.set(xlabel='House Number', ylabel='Assessed Value',
       title='House Numbers vs Value')
ax.grid()

plt.show()
house_num = df_keep["Account Number"]
price = df_y

fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(house_num[::10], price[::10], alpha=0.3)

ax.set(xlabel='Account', ylabel='Assessed Value',
       title='Account Number vs Value')
ax.grid()

plt.show()
