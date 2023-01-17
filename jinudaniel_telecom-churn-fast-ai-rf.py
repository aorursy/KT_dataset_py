# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
import os
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
df_raw = pd.read_csv(r"../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_raw.head()
df_raw.shape
df_raw.info()
df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors='coerce')
df_raw.head()
sns.countplot(x='Churn', data=df_raw)
df_raw.isnull().sum().sort_index()/len(df_raw)
#Code taken from fast ai library. Code can be found at github.
#https://github.com/fastai/fastai/blob/master/old/fastai/structured.py
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
train_cats(df_raw)
df_raw.PhoneService.cat.categories
df_raw.PaperlessBilling.cat.categories
#The below functions are taken from fast ai library, code for which can be found at github.
#https://github.com/fastai/fastai/blob/master/old/fastai/structured.py
def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = col.cat.codes+1
        
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res
df, y, nas = proc_df(df_raw, 'Churn', skip_flds=['customerID'], max_n_cat=8)
df.head().T
def split_vals(a,n): return a[:n], a[n:]
n_valid = int(7043 * 0.1)
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
from sklearn import metrics
def print_score(m):
    res = [metrics.accuracy_score(m.predict(X_train), y_train), 
           metrics.accuracy_score(m.predict(X_valid), y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m = RandomForestClassifier(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
#The below functions are taken from fast ai library, code for which can be found at github.
#https://github.com/fastai/fastai/blob/master/old/fastai/structured.py
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, df) 
fi[:10]
def plot_fi(fi): 
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30])
plt.figure(figsize=(9, 4))
plt.title("KDE for Total Charges")
ax0 = sns.kdeplot(df_raw[df_raw['Churn'] == 'No']["TotalCharges"].dropna(), color= 'green', label= 'Churn: No')
ax1 = sns.kdeplot(df_raw[df_raw['Churn'] == 'Yes']["TotalCharges"].dropna(), color= 'red', label= 'Churn: Yes')
plt.figure(figsize=(9, 4))
plt.title("KDE for Monthly Charges")
ax0 = sns.kdeplot(df_raw[df_raw['Churn'] == 'No']["MonthlyCharges"].dropna(), color= 'green', label= 'Churn: No')
ax1 = sns.kdeplot(df_raw[df_raw['Churn'] == 'Yes']["MonthlyCharges"].dropna(), color= 'red', label= 'Churn: Yes')
plt.figure(figsize=(9, 4))
plt.title("KDE for Tenure")
ax0 = sns.kdeplot(df_raw[df_raw['Churn'] == 'No']["tenure"].dropna(), color= 'green', label= 'Churn: No')
ax1 = sns.kdeplot(df_raw[df_raw['Churn'] == 'Yes']["tenure"].dropna(), color= 'red', label= 'Churn: Yes')
#Increase the number of estimator's i.e. the number of decision trees to to 40.
m = RandomForestClassifier(n_estimators=40, n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
fi
to_keep = fi[fi.imp>0.01].cols
len(to_keep)
df_keep = df[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
fi = rf_feat_importance(m, df_keep)
plot_fi(fi);
m = RandomForestClassifier(n_estimators=100, n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
