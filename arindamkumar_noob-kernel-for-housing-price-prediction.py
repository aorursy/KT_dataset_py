%load_ext autoreload
%autoreload 2

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()
train_id=len(train)
test_id=len(test)
train_id
test_id
# since the evaluation metric is RMSLE, so we take log of the dependent variable(here saleprice)
train['SalePrice']=np.log(train['SalePrice'])
train.describe()
train.SalePrice.describe()
train.YearBuilt.describe()
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
plt.plot(train['YearBuilt'],train['SalePrice'])
## finding the missing data
nanpercent=pd.DataFrame({'column':train.columns,'nan%':np.sum(train.isnull(),axis=0)/train_id})
## sorting the above nan percent in descending order
nanpercent=pd.DataFrame({'column':train.columns,'nan%':np.sum(train.isnull(),axis=0)/train_id})
nanpercent.sort_values('nan%',ascending=False,inplace=True)
nanpercent.plot.bar()
## taking the subset of the nan percent
nanpercentsubset=nanpercent[0:35]
nanpercentsubset.plot.bar()
for col in train.columns:
    print(train[col].describe())
train.columns.value_counts()
# count dtypes in the dataframe
train.dtypes.value_counts()
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
## taken from structure.py of fast.ai
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
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
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
train_cats(train)
df_trn,y_trn,nas=proc_df(train,'SalePrice')
def create_features(df):
    df['Age']=df['YrSold']-df['YearBuilt']
    df['total_living_space']=df['GrLivArea']+df['TotalBsmtSF']-df['LowQualFinSF']

df_trn
create_features(df_trn)  ## since the features above mentioned are numeric so we do not have to convert it.
train_id
def split_vals(a,n): return a[:n], a[n:]
## since the size of training set is small we do not waste much of it for validation.
n_valid = 100
n_trn = len(df_trn)-n_valid
X_train, X_val = split_vals(df_trn, n_trn)
y_train, y_val = split_vals(y_trn, n_trn)
import math
def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())
def print_score(m):
    res=[rmse(m.predict(X_train),y_train),rmse(m.predict(X_val),y_val),m.score(X_train,y_train),m.score(X_val,y_val)]
    if hasattr(m,'oob_score_'):
        res.append(m.oob_score_)
    print(res)
%time  # to calculate how much time the process took to run.
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, max_features=0.4, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
%time preds = np.stack([t.predict(X_val) for t in m.estimators_])
preds
preds.shape
preds[:,0]
#taking one of the validation
%time
print(np.mean(preds[:,0]))#mean
print(np.std(preds[:,0])) #confidance
import concurrent
def parallel_trees(m, fn, n_jobs=8):
        return list(concurrent.futures.ProcessPoolExecutor(n_jobs).map(fn, m.estimators_))
def get_preds(m):
    return m.predict(X_val)
%time 
preds = np.stack(parallel_trees(m, get_preds))
np.mean(preds[:,0]), np.std(preds[:,0])
raw_train,raw_valid=split_vals(train,n_trn)
X_copy=raw_valid.copy()
## variatiion of confidance in the data set.
X_copy['pred_std']=np.std(preds,axis=0) ## axis=0 as we are adding a column as the std is taken along the column
X_copy['mean']=np.mean(preds,axis=0)
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)
fi=rf_feat_importance(m,df_trn)
fi[:10]
def plot_fi(fi):
    return fi.plot('cols','imp','barh',figsize=(12,9),legend=False)
plot_fi(fi[:30])
fi[:28]
to_keep=fi[fi.imp>0.0035].cols
len(to_keep)
df_keep=df_trn[to_keep].copy()
X_train,X_val=split_vals(df_keep,n_trn)
## the y_train doesnot change as it is the price which is same as before
## using the same model from above
%time  
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, max_features=0.4, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
fi = rf_feat_importance(m, df_keep)
plot_fi(fi);
df_trn2, y_trn, nas = proc_df(train, 'SalePrice', max_n_cat=7)
X_train1, X_val1 = split_vals(df_trn2, n_trn)
y_train1=y_train
%time  
m2 = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, max_features=0.4, n_jobs=-1, oob_score=True)
m2.fit(X_train1, y_train1)
print_score(m2)
import scipy
## Taken from course taught by Jeremy Howard.

from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
def get_oob(df):
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, max_features=0.4, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_
def get_rmse(df):
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, max_features=0.4, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return [rmse(m.predict(X_val),y_val)]
#baseline model
print(get_oob(df_keep))
print(get_rmse(df_keep))
for c in('GarageYrBlt','YearBuilt','GrLivArea','total_living_space',
         'GarageArea','GarageCars','Fireplaces','FireplaceQu','1stFlrSF','TotalBsmtSF'):
    print(c, get_oob(df_keep.drop(c, axis=1)))
to_drop = ['GrLivArea','GarageYrBlt','TotalBsmtSF','GarageCars','Fireplaces']
print(get_oob(df_keep.drop(to_drop,axis=1)))
#print(get_rmse(df_keep.drop(to_drop,axis=1)))
np.save('keep_cols.npy', np.array(df_keep.columns))
df_result=df_keep.copy()
columns=df_result.columns
len(columns)
df_result.drop(to_drop, axis=1, inplace=True)
X_train, X_val = split_vals(df_result, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=1, max_features=0.4, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=120, min_samples_leaf=1, max_features=0.4, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
print(len(X_train),train_id,len(df_trn),len(y_train),len(y_trn))
X=X_train.append(X_val)
y=y_trn
m = RandomForestRegressor(n_estimators=120, min_samples_leaf=1, max_features=0.4, n_jobs=-1, oob_score=True)
m.fit(X, y)
#print_score(m)
df_test = pd.read_csv('../input/test.csv')
create_features(df_test) # add "ageSold" and "TotalLivingSF" to the set.
train_cats(df_test) 
df_test,_,_ = proc_df(df_test,na_dict = nas)
Id = df_test.Id
df_test = df_test[columns]
ans = np.exp(m.predict(df_test))
sub = pd.DataFrame({
    'Id':test['Id'],
    'SalePrice':np.exp(m.predict(df_test))
})

sub.head()
sub.to_csv('prediction_housing-prediction.csv',index=False) 
