import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', 100)

%matplotlib inline
"""

Note: 

may functions included here are from Jeremy's MI courses (www.fastai.ai).

"""



import os, io, platform, itertools, warnings



import pandas as pd

import numpy as np

import scipy

from scipy.cluster import hierarchy as hc



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

import plotly.offline as py

py.init_notebook_mode(connected=True)



from IPython.display import SVG,display

from IPython.display import Image

from PIL import  Image





from sklearn.preprocessing import Imputer, PolynomialFeatures, StandardScaler

from sklearn.feature_selection import VarianceThreshold, SelectFromModel

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV, learning_curve, train_test_split

from sklearn.metrics import classification_report, confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

import lightgbm as lgb

import gc

import re



from datetime import datetime

from pandas.api.types import is_string_dtype, is_numeric_dtype



from sklearn.tree import DecisionTreeClassifier, export_graphviz

from graphviz import Source





pd.set_option('display.max_columns', 100)





def missing_data(df):

    """df: panda data frame"""

    total = df.isnull().sum().sort_values(ascending=False)

    percent = total / len(df) 

    return pd.concat([total,percent], axis=1, keys =['Total', 'Percent'])



def split_time(df, fldname, drop=True, date=True, time=False, errors='raise'):

    fld = df[fldname]

    fld_dtype = fld.dtype

    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)

    targ_pre = re.sub('[Dd]ate$','',fldname)

    attr = []

    if date:

        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    if time:

        attr = attr + ['Hour', 'Minute', 'Second']

    for n in attr:

        df[targ_pre+n] = getattr(fld.dt, n.lower())

    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9

    if drop:

        df.drop(fldname, axis=1, inplace=True)



def is_date(x): return np.issubdtype(x.dtype, np.datetime64)



def cat_train(df):

    for n, c in df.items():

        if is_string_dtype(c):

            df[n] = c.astype('category').cat.as_ordered()



def apply_cats(df, trn):    

    for n,c in df.items():

        if (n in trn.columns) and (trn[n].dtype.name=='category'):

            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)





def prep_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,

            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):    

    if not ignore_flds: 

        ignore_flds=[]

        

    if not skip_flds: 

        skip_flds=[]

        

    if subset: 

        df = get_sample(df,subset)

    else: 

        df = df.copy()

        

    ignored_flds = df.loc[:, ignore_flds]

    df.drop(ignore_flds, axis=1, inplace=True)

    

    if preproc_fn: 

        preproc_fn(df)

        

    if y_fld is None: 

        y = None

    else:

        if not is_numeric_dtype(df[y_fld]): 

            df[y_fld] = df[y_fld].cat.codes

            

        y = df[y_fld].values

        skip_flds += [y_fld]

        

    df.drop(skip_flds, axis=1, inplace=True)



    if na_dict is None: 

        na_dict = {}

    else: 

        na_dict = na_dict.copy()

        

    na_dict_initial = na_dict.copy()

    

    for n,c in df.items(): 

        na_dict = fix_missing(df, c, n, na_dict)

    

    if len(na_dict_initial.keys()) > 0:

        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

    

    if do_scale: 

        mapper = scale_vars(df, mapper)

        

    for n,c in df.items(): 

        numericalize(df, c, n, max_n_cat)

    

    df = pd.get_dummies(df, dummy_na=True)

    df = pd.concat([ignored_flds, df], axis=1)

    res = [df, y, na_dict]

    

    if do_scale: 

        res = res + [mapper]

    

    return res



def fix_missing(df, col, name, na_dict):    

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict



def scale_vars(df, mapper):

    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)

    if mapper is None:

        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]

        mapper = DataFrameMapper(map_f).fit(df)

    df[mapper.transformed_names_] = mapper.transform(df)

    return mapper



def numericalize(df, col, name, max_n_cat):

    

    if not is_numeric_dtype(col) and (max_n_cat is None or len(col.cat.categories)>max_n_cat):

        df[name] = col.cat.codes+1





def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)
churn_df = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')

churn_df.head()
print ("Num of rows: " + str(churn_df.shape[0])) # row count

print ("Num of columns: " + str(churn_df.shape[1])) # col count
churn_percent = churn_df['churn'].sum().astype(float)/len(churn_df['churn'])

'{:2.2%} of all customers tend to leave'.format(churn_percent)
missing_data(churn_df)
churn_df.dtypes
churn_df['area code'] = churn_df['area code'].astype(object)

churn_df['churn'] = np.where(churn_df['churn'] == True,1,0)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sb



sb.distplot(churn_df['total intl charge'], kde=False)
# Select the numeric columns

cols = ['account length', 'number vmail messages',

       'total day minutes', 'total day calls', 'total day charge',

       'total eve minutes', 'total eve calls', 'total eve charge',

       'total night minutes', 'total night calls', 'total night charge',

       'total intl minutes', 'total intl calls', 'total intl charge',

       'customer service calls','churn']      



# Find correlations with the sale price 

correlations = churn_df[cols].corr()
sb.heatmap(correlations)
cat_train(churn_df)
X, y, na_dict = prep_df(churn_df, 'churn')
X.shape, na_dict
## keep 20% for test later

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_index, test_index = next(kfold.split(X, y))

len(train_index), len(test_index)
def split_test(X, y):

    X_train = X.iloc[train_index]

    y_train = y[train_index]



    X_test = X.iloc[test_index]

    y_test = y[test_index]

    return X_train, y_train, X_test, y_test
from sklearn.ensemble import RandomForestClassifier

X_train, y_train, X_test, y_test = split_test(X, y)

m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
def confusion_df(y_test,pred):

    return pd.DataFrame(confusion_matrix(y_test,pred),

             columns=["Predicted Class " + str(class_name) for class_name in [0,1]],

             index = ["Class " + str(class_name) for class_name in [0,1]])
m.fit(X_train, y_train)

pred = m.predict(X_test)

print(confusion_df(y_test, pred))
fi = rf_feat_importance(m, X); fi[:10]
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi);
to_keep = fi[fi.imp>0.005].cols; len(to_keep)

X_keep = X[to_keep].copy()
X2, y, _ = prep_df(churn_df, 'churn', max_n_cat=7)



X_train2, y_train, X_test2, y_test = split_test(X2, y)



m.fit(X_train2, y_train)
pred = m.predict(X_test2)

print(confusion_df(y_test, pred))
fi = rf_feat_importance(m, X2)

plot_fi(fi);
corr = np.round(scipy.stats.spearmanr(X_keep).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,10))

dendrogram = hc.dendrogram(z, labels=X_keep.columns, orientation='left', leaf_font_size=16)

plt.show()
def get_oob(X, y):

    m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)   

    m.fit(X, y)

    return m.oob_score_
get_oob(X_keep, y)
for c in ('total day minutes','total day charge', 'total eve minutes','total eve charge',

          'total intl minutes','total intl charge', 'total night minutes','total night charge', 'voice mail plan',

       'number vmail messages'):

    print(c, get_oob(X_keep.drop(c, axis=1), y))
to_drop = ['total day minutes', 'total eve minutes', 'total intl minutes', 'total night minutes', 'voice mail plan']

print(get_oob(X_keep.drop(to_drop, axis=1), y))
X_keep.drop(to_drop, axis=1, inplace=True)

X_keep.columns
X_train, y_train, X_test, y_test = split_test(X_keep, y)
X_test.shape, X_train.shape
### Find Optimal Parameters: Hyperopt

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,  space_eval



param_space = {

    'max_depth': hp.choice('max_depth', range(1,10)),

    'max_features': hp.choice('max_features', ['auto', 'sqrt', 0.5, 0.6, None]),

    'n_estimators': 100,

    'criterion': hp.choice('criterion', ["gini", "entropy"]),

    # Minimum number of samples required to split a node

    'min_samples_split': hp.choice('min_samples_split', range(5, 25)),

     # Minimum number of samples required at each leaf node

    'min_samples_leaf': hp.choice('min_samples_leaf', range(4, 25)),

    'bootstrap': hp.choice('bootstrap', [True, False])

    }



best_score = 0



def RF_score(params):

    global best_score

    clf = RandomForestClassifier(**params)

    score = cross_val_score(clf, X_train, y_train, cv=3, scoring='recall_weighted').mean()    

    if score > best_score:

        best_score = score

        print('score:', score)

        print('params:', params)

    return {'loss': -score, 'status': STATUS_OK}   





trials = Trials()

best = fmin(RF_score, param_space, algo=tpe.suggest, max_evals=100, trials=trials)

print('best parameters:')

best_params = space_eval(param_space, best)

print(best_params)  
model = RandomForestClassifier(**best_params)

g = plot_learning_curve(model, "Learning curve", X_train, y_train, cv=5)
model.fit(X_train, y_train)



prediction = model.predict(X_test)

print(confusion_df(y_test, prediction))
plot_fi(rf_feat_importance(model, X_keep));
rf_feat_importance(model, X_keep)