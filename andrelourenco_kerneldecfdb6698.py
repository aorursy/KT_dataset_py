 !pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887 2>/dev/null 1>/dev/null
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import math 

import pandas as pd

import numpy as np 

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

#Xgboost

import xgboost as xgb

#CatBoost 

from catboost import CatBoostRegressor

import scipy as scipy

import matplotlib.pyplot as plt

from pandas_summary import DataFrameSummary

from IPython.display import display

from sklearn.model_selection import GridSearchCV

from pdpbox import pdp



from scipy import stats

from scipy.stats import norm, skew #for some statistics

from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler







from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.SalePrice = np.log(train.SalePrice)

train.shape
Id = test['Id']

test_copy = test.copy()

test_copy["SalePrice"] = np.nan



train_set_data = [train,test_copy]

train_set_data = pd.concat(train_set_data)

len(train_set_data) == len(train)+len(test)



train_set_data.drop("Id", axis = 1, inplace = True)
# Missing Value Count Function

def show_missing():

    missing = train_set_data.columns[train_set_data.isnull().any()].tolist()

    return missing



# Missing data counts and percentage

print('Missing Data Count')

print(train_set_data[show_missing()].isnull().sum().sort_values(ascending = False))

print('--'*40)

print('Missing Data Percentage')

print(round(train_set_data[show_missing()].isnull().sum().sort_values(ascending = False)/len(train_set_data)*100,2))
numeric_feats = train_set_data.dtypes[train_set_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = train_set_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.85]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    train_set_data[feat] = boxcox1p(train_set_data[feat], lam)

    
numeric_feats = train_set_data.dtypes[train_set_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = train_set_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
Id = test['Id']

test_copy = test.copy()

test_copy["SalePrice"] = np.nan



train_set_data = [train,test_copy]

train_set_data = pd.concat(train_set_data)

len(train_set_data) == len(train)+len(test)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)



#################################



def display_all(df):

    with pd.option_context("display.max_rows", 1000): 

        with pd.option_context("display.max_columns", 1000): 

            display(df)



################################



def add_datepart(df, fldname, drop=True):

    fld = df[fldname]

    if not np.issubdtype(fld.dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, 

                                     infer_datetime_format=True)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 

            'Dayofyear', 'Is_month_end', 'Is_month_start', 

            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 

            'Is_year_start'):

        df[targ_pre+n] = getattr(fld.dt,n.lower())

    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9

    if drop: df.drop(fldname, axis=1, inplace=True)

        

################################



def train_cats(df):

    for n,c in df.items():

        if pd.api.types.is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

            

################################



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

        if not pd.api.types.is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes

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



#################################



def numericalize(df, col, name, max_n_cat):

     if not pd.api.types.is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):

        df[name] = pd.Categorical(col).codes+1

        

#################################



def fix_missing(df, col, name, na_dict):

    

    if pd.api.types.is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict



##################################



def rmse(x,y): return math.sqrt(((x-y)**2).mean())



##################################



def print_score(m):

    res = [rmse(m.predict(X_train), y_train),

           rmse(m.predict(X_valid), y_valid),

           m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)

    

##################################



def get_oob(df):

    m = RandomForestRegressor(n_estimators=80, min_samples_leaf=1, 

           max_features=0.5, n_jobs=-1, oob_score=True)

    x, _ = split_vals(df, n_trn)

    m.fit(x, y_train)

    return m.oob_score_
train_cats(train_set_data)
df, y, nas = proc_df(train_set_data, 'SalePrice',max_n_cat=8)
# Missing Value Count Function

def show_missing():

    missing = df.columns[df.isnull().any()].tolist()

    return missing



# Missing data counts and percentage

print('Missing Data Count')

print(df[show_missing()].isnull().sum().sort_values(ascending = False))

print('--'*40)

print('Missing Data Percentage')

print(round(df[show_missing()].isnull().sum().sort_values(ascending = False)/len(df)*100,2))
test_df = df[1460:2919]

df = df[0:1460]

y=y[0:1460]
m = RandomForestRegressor(n_jobs=-1)

m.fit(df, y)

m.score(df,y)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 400  

n_trn = len(df)-n_valid

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
m = RandomForestRegressor( n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=80, min_samples_leaf=1, max_features=0.5, n_jobs=-1,oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)
preds = np.stack([t.predict(X_valid) for t in m.estimators_])

print(preds[:,0], np.mean(preds[:,0]), y_valid[0])



print(preds.shape)
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(80)])
fi = rf_feat_importance(m, df); fi[:10]
def plot_fi(fi): return fi.plot('cols','imp','barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);
to_keep = fi[fi.imp>0.005].cols; 

print(len(to_keep))

df_keep=df[to_keep].copy()

X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=80, min_samples_leaf=1, max_features=0.5, n_jobs=-1,oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)
fi = rf_feat_importance(m, df_keep)

def plot_fi(fi): return fi.plot('cols','imp','barh', figsize=(12,7), legend=False)

plot_fi(fi[:18]);
from scipy.cluster import hierarchy as hc

corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,10))

dendrogram = hc.dendrogram(z, labels=df_keep.columns, 

      orientation='left', leaf_font_size=16)

plt.show()
get_oob(df_keep)
for c in ['GarageYrBlt', 'GarageCars']:

    print(c, get_oob(df_keep.drop(c, axis=1)))
to_drop = ['GarageYrBlt']

get_oob(df_keep.drop(to_drop, axis=1))
df_keep.drop(to_drop, axis=1, inplace=True)


X_train, X_valid = split_vals(df_keep, n_trn)

m = RandomForestRegressor(n_estimators=1000, min_samples_leaf=1, max_features=0.5, n_jobs=-1,oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
xgboost = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
from lightgbm import LGBMRegressor

lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )
xgb_classifier = xgb.XGBRegressor()
gbm_param_grid = {

    'n_estimators': range(1,100),

    'max_depth': range(1, 15),

    'learning_rate': [.1,.13, .16, .19,.3,.6],

    'colsample_bytree': [.6, .7, .8, .9, 1]

}
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 

                                    estimator = xgb_classifier, 

                                    verbose = 0, n_iter = 50, cv = 4)
lightgbm.fit(X_train,y_train)
print_score(lightgbm)
xgboost.fit(X_train,y_train)
print_score(xgboost)
test_df = test_df[list(X_train.columns)]

y_pred = xgboost.predict(test_df)
submission = pd.DataFrame({"Id": Id,"SalePrice": y_pred})
submission.SalePrice = np.exp(submission.SalePrice)
submission.to_csv('submission.csv', index=False)