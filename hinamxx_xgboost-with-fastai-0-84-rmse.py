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
# Initialization.
# I use these 3 lines of code on top of each Notebooks.
%reload_ext autoreload
%autoreload 2
%matplotlib inline
# Downloading all the necessary Libraries and Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re, math, graphviz, scipy
import seaborn as sns

# I will use XGboost in this Project because the Dataset has Timeseries Data.
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance

# I will also use the Fastai API in this Project for Data Preprocessing and Data Preparation
from pandas.api.types import is_string_dtype, is_numeric_dtype
from IPython.display import display
from sklearn.ensemble import forest
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.cluster import hierarchy as hc
from plotnine import *
from sklearn import metrics
from concurrent.futures import ProcessPoolExecutor
# Loading the Data
# I am using Colab for this Project so accessing the Data might be different in different platforms.
# path = "/content/drive/My Drive/Predict Future Sales"

# Creating the DataFrames using Pandas
transactions = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
item_categories = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
# Looking and Inspecting the Data
## Transactions DataFrame 
display(transactions.head(3)); 
transactions.shape
## Items DataFrame
display(items.head(3)); 
items.shape
## Item Categories DataFrame
display(item_categories.head(3));
item_categories.shape
## Shops DataFrame
display(shops.head(3));
shops.shape
# Test DataFrame
display(test.head());
test.shape
# Merging the Transactions and Items DataFrame on "Item Id" column 
train = pd.merge(transactions, items, on="item_id", how="left")
train.tail()
# Merging the Train, Item Categories and Shops DataFrame as well.
# Merging Train and Item Categories on "Item Category Id" column.
train_df = pd.merge(train, item_categories, on="item_category_id", how="left")
# Merging Train and Shops DataFrame on "Shop Id" column.
train_df = pd.merge(train_df, shops, on="shop_id", how="left")
train_df.head(10)
# Changing the Data column in Datetime Object
train_df["date"] = pd.to_datetime(train_df["date"], format="%d.%m.%Y")
train_df["date"].head()
# Working on Data Leakages
# Checking on Test DataFrame and Removing the Unnecessary Features
test_shops = test["shop_id"].unique()
test_items = test["item_id"].unique()
# Removing the Redundant Features
train_df = train_df[train_df["shop_id"].isin(test_shops)]
train_df = train_df[train_df["item_id"].isin(test_items)]
display(train_df.head()); train_df.shape
# Keeping only the Items whose price is greater than 0
train_df = train_df.query("item_price > 0")
# Creating the new features which contains the Items sold on a particulat month
# Item_cnt_day contains the number of Items sold
train_df["item_cnt_day"] = train_df["item_cnt_day"].clip(0, 20)
train_df = train_df.groupby(["date", "item_category_id", "shop_id", "item_id", "date_block_num"])
train_df = train_df.agg({'item_cnt_day':"sum", 'item_price':"mean"}).reset_index()
train_df = train_df.rename(columns={"item_cnt_day":'item_cnt_month'})
# Using clip(0, 20) to meet the requirements of the Competition
train_df["item_cnt_month"] = train_df["item_cnt_month"].clip(0, 20)
train_df.head()
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None, prepoc_fn=None, max_n_cat=None,
           subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset:
        df = get_sample(df, subset)
    else:
        df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if prepoc_fn: prepoc_fn(df)
    if y_fld is None: y=None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)
    
    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n, c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n, c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res
def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict): 
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict
def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and (max_n_cat is None or col.nunique()>max_n_cat):
        df[name] = col.cat.codes+1

def get_sample(df, n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

def set_rf_samples(n):
    forest._generate_sample_indices = (lambda rs, n_samples:
                                      forest.check_random_state(rs).randit(0, n_samples, n))

def reset_rf_samples():
    forest._generate_sample_indices = (lambda rs, n_samples:
                                      forest.check_random_state(rs).randit(0, n_samples, n_samples))        
def split_vals(a, n):
    return a[:n].copy(), a[n:].copy()

def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def apply_cats(df, trn):
    for n, c in df.items():
        if trn[n].dtype.name == "category":
            df[n] = pd.Categorical(c, categories = trn[n].cat.categories, ordered = True)
def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    if isinstance(fldnames, str):
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64
            
        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub("[Dd]ate$", '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elasped'] = fld.astype(np.int64) // 10**9
        if drop: df.drop(fldname, axis=1, inplace=True)
def scale_vars(df, mapper):
    warnings.filterwarnings("ignore", category = sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n], StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper
def rmse(x, y):
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train),
          rmse(m.predict(X_valid), y_valid),
          m.score(X_train, y_train),
          m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)
# Using add_datepart function 
# This function is very useful while working on Time-Series Data
add_datepart(train_df, "date")
train_df.columns
# Observing the DataFrame again after applying API
train_df.head()
# Dealing with Categorical Features
train_cats(train_df)
# Checking for Null Values in DataFrame
train_df.isnull().sum().sort_index() / len(train_df)
os.makedirs("tmp", exist_ok=True)
train_df.to_feather("tmp/new")
# Loading the Data and Going through simple Exploratory Data Analysis
data = pd.read_feather("tmp/new")
display(data.head(3));
data.shape
data.describe()
new_df, y, nas = proc_df(data, "item_cnt_month")
# Preparing the Validation Data
n_valid = 200000
n_trn = len(data) - n_valid
raw_train, raw_valid = split_vals(data, n_trn)
X_train, X_valid = split_vals(new_df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

# Checking the Shape of Training and Validation Data
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
# Creating the Regressor Model
model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3, 
    seed=42
)

# Fitting the Model
model.fit(
    X_train,
    y_train,
    eval_metric="rmse",
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=10
)
X_test = data[data["date_block_num"] == 33].drop(["item_cnt_month"], axis=1)
Y_test = model.predict(X_test)
submission = pd.DataFrame({
    "ID": test["ID"].iloc[:49531], 
    "item_cnt_month": Y_test.clip(0, 20)
})
submission.to_csv('xgb_submission.csv', index=False)
