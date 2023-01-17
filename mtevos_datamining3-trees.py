# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/bankdefaultsinrussia/dataset.csv')
df.columns = ['license',

 'date',

 'net_assets',

 'ROA',

 'liquid',

 'ibl',

 'stocks',

 'bond',

 'oth_cap',

 'sunk_retail_credit',

 'NI',

 'organization_credit',

 'sunk_organization_credit',

 'credit_portf',

 'sunk_credit_portf',

 'organization_deposit',

 'retail_deposit',

 'security_tot',

 'ROE',

 'retail_credit',

 'reserv_credit_perc',

 'zalog_credit_perc',

 'foreign_na_fr',

 'retail_deposit_fr',

 'N3',

 'N2',

 'N1',

 'capital',

 'msk_spb',

 'INF_SA',

 'NX_growth',

 'micex_std',

 'miacr_std',

 'miacr_amount',

 'usd_rub_std_diff',

 'micex_return',

 'net_foreign_assets_diff',

 'net_gov_debt_diff',

 'other_fin_debt_diff',

 'retail_debt_SA_DETREND_diff',

 'stocks_capital_diff',

 'i_retail_spread_diff',

 'usd_rub_return',

 'miacr_diff',

 'default']
print(f'THERE ARE {df.license.nunique()} UNIQUE BANKS IN THIS DATASET')
defaulted = pd.DataFrame(df.groupby(['license'])['default'].max()).reset_index()

print(f'OUT OF WHICH {dict(defaulted.groupby(["default"])["license"].count())[1]} EVENTUALLY DEFAULTED')
from matplotlib import pyplot as plt

import seaborn as sns
df['DATE'] =  pd.to_datetime(df.date)
df['YEAR'] = pd.DatetimeIndex(df['DATE']).year

df['MONTH'] = pd.DatetimeIndex(df['DATE']).month

df['DOF'] = pd.DatetimeIndex(df['DATE']).dayofweek
df[['YEAR', 'MONTH']].head()
df['YEARMONTH'] = df.apply(lambda row: 100 * row['YEAR'] + row['MONTH'], axis = 1)

df['YEARMONTH'].head()
df.head()
defaulted = df[df.default == 1].license.unique()

_df = df[df.license.isin(defaulted)]

plt.scatter(_df.YEARMONTH, _df.default);
from collections import Counter

freq = pd.DataFrame(Counter(df.license), index = ['cnt']).T.sort_values('cnt', ascending = False).reset_index()
freq.head()
freq.tail()
from pylab import rcParams

rcParams['figure.figsize'] = 20, 10

_df = df[df.license.isin([1144., 2696., 1067., 3296., 2398., 3292., 2664., 2961., 2271.,

       1948., 2995., 1411., 2103., 2093., 2649., 3265., 2645., 2609.,

        702.,  704.])].sort_values('DATE')

plt.plot(_df.DATE, _df.net_assets)

plt.title('NET ASSETS OF BANKS \nWITH MOST FREQUENT REPORTING');
df = df[~(df.DATE.isna()|df.license.isna())].copy(deep = True)
df['N_REPORTS'] = df.groupby('license')['license'].transform('count')

df['OCCURENCE'] = df.groupby(['license']).cumcount()+1
df[df.license == 2696.]
colors = ['blue', 'red']

for defaulted in [0,1]:

    sns.distplot(df[df.default == defaulted].N_REPORTS, kde = True, hist= False, bins = 100)

plt.title('WE SEE THAT THERE IS BALANCE BETWEEN\n REPORTING FREQUENCY AND DEFAULT');
pd.DataFrame(df.groupby('license')['OCCURENCE'].max())['OCCURENCE'].describe()
df.sort_values(['license', 'OCCURENCE'], inplace = True)

df['next_day'] = df.DATE.shift(-1)

df['next_id'] = df.license.shift(-1)
df['GAP'] = (df['next_day'] - df['DATE']).dt.days
df['GAP'] = df.GAP.apply(lambda x: x if x > 0 else 0)

df['GAP'].describe()
df['INV_OCCURENCCE'] = df.N_REPORTS - df.OCCURENCE
#TAKING LAST 50 VALUES

ndf = df[df.INV_OCCURENCCE < 50]
df[df.GAP > 0].groupby('license')['GAP'].describe()
cols = list(ndf)
ndf.head()
featcols = cols[cols.index('net_assets') : cols.index('default')]

tdf = ndf.groupby('license')[featcols].mean().reset_index()
#ASSIGNING LABELS

labels = dict(ndf.groupby('license')['default'].max())

tdf['Y'] = tdf.license.apply(lambda x: labels[x])
tdf.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



X_train, X_test, y_train, y_test = train_test_split(tdf[featcols], tdf['Y'])
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 0)

dt.fit(X_train, y_train)
def get_scores(_yhat: np.array, _y_true: np.array) -> dict:



    """

    Costumizes the confusion matrix,

    and calculates recall and precision for both classes.

    """



    _bmx = confusion_matrix(_yhat, _y_true).flatten()



    _OO = _bmx[0]

    _OI = _bmx[2]

    _IO = _bmx[1]

    _II = _bmx[3]



    # FOR CLASS 1

    _per1 = _II / (_II + _OI)

    _rec1 = _II / (_II + _IO)



    # FOR CLASS 0

    _per0 = _OO / (_OO + _IO)

    _rec0 = _OO / (_OO + _OI)



    # F1 SCORES

    _f11 = 2 * _per1 * _rec1 / (_per1 + _rec1)

    _f10 = 2 * _per0 * _rec0 / (_per0 + _rec0)



    # F1 TOTAL

    _ftot = 2 * _f11 * _f10 / (_f11 + _f10)



    _cc = Counter(_y_true.flatten())

    _res = {"PREC_DEFAULT": _per1,

            "REC_DEFAULT": _rec1,

            "PRE_NOT_DEFAULT": _per0,

            "REC_NOT_DEFAULT": _rec0,

            "F1_DEFAULT": _f11,

            "F1_NOT_DEFAULT": _f10,

            "F1_ALL": _ftot,

            "NOT_DEFAULT": _cc[0],

            "DEFAULT": _cc[1],

            "NOT_as_NOT": _OO,

            "NOT_as_CLS": _OI,

            "CLS_as_NOT": _IO,

            "CLS_as_CLS": _II,

            "BENCHMARK PRECISION FOR DEFAULT": (_II+_IO)/sum(_cc.values())}

    return _res
# A NICE EXAMPLE OF OVERFITTING

pd.DataFrame([get_scores(dt.predict(X_test), y_test.values), get_scores(dt.predict(X_train), y_train.values)], index = ['TESTING', 'TRAINING']).T
from datetime import datetime

from sklearn.model_selection import GridSearchCV

def search_grids(X, y, clf, params_grid, cros_val=5):

    grid_search = GridSearchCV(clf,

                               param_grid=params_grid,

                               cv=cros_val)

    start = datetime.now()

    grid_search.fit(X, y)

    end = datetime.now()

    print (f"TOOK {(end - start).seconds} SECONDS")

    return  grid_search.best_params_, grid_search.best_estimator_
tree_param_grid = {"criterion": ["gini", "entropy"],

              "min_samples_split": [2, 4, 16],

              "max_depth": [None, 2, 4, 16],

              "min_samples_leaf": [5, 20],

              "max_leaf_nodes": [None, 5, 20],

              }
dt.__dict__
dtc = DecisionTreeClassifier(random_state = 0)

prm, tr = search_grids(X_train, y_train, dtc, tree_param_grid, cros_val = 3)
prm
# !conda uninstall python-graphviz
# !conda uninstall graphviz
# !pip install dtreeviz
# from sklearn.datasets import *

# from sklearn import tree

# from dtreeviz.trees import *
# %matplotlib inline%matplotlib inline

# viz = dtreeviz(tr,

#                tdf,

#                tdf.Y,

#                target_name='DEFAULTED',

#                feature_names=featcols)

              

# viz.view()              

# viz = dtreeviz(tr,

#                tdf,

#                tdf.Y,

#                target_name='DEFAULTED',

#                feature_names=featcols)

              

# viz.view()              
# rft.predict_proba(X_test)
# A CLASSIC EXAMPLE OF OVERFITTING

pd.DataFrame([get_scores(tr.predict(X_test), y_test.values), get_scores(dt.predict(X_test), y_test.values)], index = ['GRIDSEARCH', 'BASE PARAMS']).T
# Bagging, Boosting, Stacking
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
rf_grid = {'bootstrap': [True, False],

             'max_depth': [10, 20, None],

             'max_features': ['auto', 'sqrt'],

             'min_samples_split': [2, 20],

             'n_estimators': [50, 100]}
rf1 = RandomForestClassifier(random_state = 0)

prm, rf = search_grids(X_train, y_train, rf1, rf_grid, cros_val = 3)
et1 = ExtraTreesClassifier(random_state = 0)

prm, et = search_grids(X_train, y_train, et1, rf_grid, cros_val = 3)
rft.__dict__
rft = RandomForestClassifier(random_state = 0)

rft.fit(X_train, y_train)

ett = ExtraTreesClassifier(random_state = 0)

ett.fit(X_train, y_train)
rf.estimators_[0]
# rft.estimators_
pd.DataFrame([

    get_scores(rf.predict(X_test), y_test.values), 

    get_scores(rft.predict(X_test), y_test.values),

    get_scores(et.predict(X_test), y_test.values), 

    get_scores(ett.predict(X_test), y_test.values),



], index = ['GRIDSEARCH RF', 'BASE PARAMS RF', 'GRIDSEARCH EXTT', 'BASE PARAMS EXTT']).T
#WHY? SMALL DATA ISSUE
from xgboost import XGBClassifier as xgb

from lightgbm import LGBMClassifier as lg

from catboost import CatBoostClassifier as cat
xx = xgb(random_state= 0)

lgbm = lg(random_state = 0)

ctt = cat(random_state = 0)
xx.fit(X_train, y_train)

lgbm.fit(X_train, y_train)

ctt.fit(X_train, y_train, verbose = False)
pd.DataFrame([get_scores(tr.predict(X_test), y_test.values), 

              get_scores(rft.predict(X_test), y_test.values), 

              get_scores(ett.predict(X_test), y_test.values), 

              get_scores(xx.predict(X_test), y_test.values),

              get_scores(lgbm.predict(X_test), y_test.values),

              get_scores(ctt.predict(X_test), y_test.values),

             ],

             index = ['SINGLE_TREE', '100 RF', '100 EXTT', 'XGBOOST', 'LIGHTGBM', 'CAT']).T
cat.__dict__['__init__']
#NEXT POTENTIAL STEPS

#- META CLASSIFER TRAINED ON VARIOUS TIMEFRAMES OF THE DATASET

#- FEATURE TRANSFORMATION USING DOIMAIN KNOWLEDGE (RATIOS/ETC)
#FIN
#P.S ONE MORE ALGORITHM
!pip install skope-rules
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from skrules import SkopeRules
# rft.feauture_importance
rule = SkopeRules(random_state=0, feature_names = featcols)
rule.fit(X_train, y_train)

get_scores(rule.predict_top_rules(X_test, 5), y_test.values), 
rule.rules_[:3]
tdf.query('INF_SA > 0.009013020433485508 and net_gov_debt_diff > -37380.984375 and other_fin_debt_diff <= 73263.51953125').groupby('Y')['license'].nunique()
!pip install rfpimp
from sklearn.metrics import r2_score

from rfpimp import permutation_importances



def get_r2(_rf, _X_train, _y_train):

    return r2_score(_y_train, _rf.predict(X_train))



perm = perm_imp_rfpimp = permutation_importances(rft, X_train, y_train, get_r2)
%matplotlib inline

perm