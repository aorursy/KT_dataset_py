# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame, Series

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gc

import warnings

warnings.filterwarnings('ignore')



import scipy as sp



from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss, roc_curve, confusion_matrix, plot_roc_curve

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from sklearn.preprocessing import LabelEncoder



from tqdm import tqdm_notebook as tqdm

from category_encoders import OrdinalEncoder



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMClassifier
#データ読込

df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, parse_dates=['issue_d'])

df_test =  pd.read_csv('../input/homework-for-students4plus/test.csv', index_col=0, parse_dates=['issue_d'])

#オプションデータ

df_zipcode = pd.read_csv('../input/homework-for-students4plus/free-zipcode-database.csv', dtype={'Zipcode':np.int64})

df_gdp = pd.read_csv('../input/homework-for-students4plus/US_GDP_by_State.csv')

df_statelatlong = pd.read_csv('../input/homework-for-students4plus/statelatlong.csv')

df_zipcode = df_zipcode[['Zipcode','Xaxis','Yaxis','Zaxis']]

df_gdp = df_gdp[df_gdp.year>2013]
#zip_code整形

df_zipcode = df_zipcode.rename(columns={'Zipcode':'zip_code'})

df_zipcode['zip_code'] = df_zipcode.zip_code.astype(str).str[:3].astype(np.int64)

df_zipcode = df_zipcode[~df_zipcode.duplicated('zip_code')]

df_zipcode.head()
#zip_code結合

df_train['zip_code'] = df_train.zip_code.str[:3].astype(np.int64)

df_test['zip_code'] = df_test.zip_code.str[:3].astype(np.int64)

df_train = pd.merge(df_train,df_zipcode,on='zip_code',how='left')

df_test = pd.merge(df_test,df_zipcode,on='zip_code',how='left')

df_train.shape
# State、GDP結合

df_statelatlong = df_statelatlong.rename(columns={'State':'addr_state'})

df_gdp = df_gdp.rename(columns={'State':'City'})

df_train = pd.merge(df_train,df_statelatlong,on='addr_state',how='left')

df_test = pd.merge(df_test,df_statelatlong,on='addr_state',how='left')
# 2014年以降絞り

df_train = df_train[df_train['issue_d'] > '2013-12-31']
#元データから特徴量生成 ///

df_train["earliest_cr_line2"]=pd.to_datetime(df_train["earliest_cr_line"])

df_test["earliest_cr_line2"]=pd.to_datetime(df_test["earliest_cr_line"])



df_train["issue_d_unix"] = df_train["issue_d"].view('int64') // 10**9

df_test["issue_d_unix"] = df_test["issue_d"].view('int64') // 10**9



df_train["earliest_cr_line_unix"] = df_train["earliest_cr_line2"].view('int64') // 10**9

df_test["earliest_cr_line_unix"] = df_test["earliest_cr_line2"].view('int64') // 10**9



df_train["period"]=df_train["issue_d_unix"]-df_train["earliest_cr_line_unix"]

df_test["period"]=df_test["issue_d_unix"]-df_test["earliest_cr_line_unix"]



df_train["period"]=df_train["period"].fillna(0)

df_test["period"]=df_test["period"].fillna(0) 



df_train['loan_amnt__installment']=round(df_train['loan_amnt']/df_train['installment'],5)

df_test['loan_amnt__installment']=round(df_test['loan_amnt']/df_test['installment'],5)



df_train['loan_amnt__annual_inc']=round(df_train['loan_amnt']/df_train['annual_inc'],5)

df_test['loan_amnt__annual_inc']=round(df_test['loan_amnt']/df_test['annual_inc'],5)



df_train['revol_bal__revol_util']=round(df_train['revol_bal']/df_train['revol_util'],5)

df_test['revol_bal__revol_util']=round(df_test['revol_bal']/df_test['revol_util'],5)



df_train['revol_bal__total_acc']=round(df_train['revol_bal']/df_train['total_acc'],5)

df_test['revol_bal__total_acc']=round(df_test['revol_bal']/df_test['total_acc'],5)



df_train['loan_amnt__open_acc']=round(df_train['loan_amnt']/df_train['open_acc'],5)

df_test['loan_amnt__open_acc']=round(df_test['loan_amnt']/df_test['open_acc'],5)
# target取り出し

y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
#意味なさそうなので削除

X_train.drop('issue_d', axis=1, inplace=True)

X_test.drop('issue_d', axis=1, inplace=True)

X_train.drop('earliest_cr_line2', axis=1, inplace=True)

X_test.drop('earliest_cr_line2', axis=1, inplace=True)

#X_train.drop('earliest_cr_line', axis=1, inplace=True)

#X_test.drop('earliest_cr_line', axis=1, inplace=True)

#X_train.drop('grade', axis=1, inplace=True)

#X_test.drop('grade', axis=1, inplace=True)

#X_train.drop('sub_grade', axis=1, inplace=True)

#X_test.drop('sub_grade', axis=1, inplace=True)
#交互作用　数値

cols1 = ['loan_amnt','installment','installment','loan_amnt','loan_amnt']

cols2 = ['revol_bal','revol_bal','revol_util','open_acc','revol_util']



#add_cols = []

for col1,col2 in zip(cols1,cols2):

    col12m = col1+'&'+col2

    X_train[col12m]=X_train[col1]*X_train[col2]

    X_test[col12m]=X_test[col1]*X_test[col2]
#数値用

X_train.fillna(X_train.median(), inplace=True)

X_test.fillna(X_test.median(), inplace=True)
#dtypeがobject(数値でない)のカラム名とユニーク数を確認

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
#文字列用

X_train.fillna('#', inplace=True)

X_test.fillna('#', inplace=True)
# Target Encoding

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



for col in cats:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test[col] = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=99, shuffle=True)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train[col]  = enc_train
#Top2

X_train['grade_subgrade'] = X_train['grade'] * X_train['sub_grade']

X_test['grade_subgrade'] = X_test['grade'] * X_test['sub_grade']



#特徴量追加 交互作用 文字



cols1 = ['grade','grade','grade','grade','grade','grade']

cols3 = ['sub_grade','sub_grade','sub_grade','sub_grade','sub_grade','sub_grade']

cols2 = ['dti','emp_title','loan_amnt','annual_inc','tot_cur_bal','home_ownership']



cols1.extend(cols3)

cols2.extend(cols2)



add_cols = []

for col1,col2 in zip(cols1,cols2):

    col12m = col1+'&'+col2

    X_train[col12m]=X_train[col1]*X_train[col2]

    X_test[col12m]=X_test[col1]*X_test[col2]

    add_cols.append(col12m)
#一応

X_train.fillna(X_train.median(), inplace=True)

X_test.fillna(X_test.median(), inplace=True)
# 学習用と検証用に分割する

X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.05, random_state=71)
from hyperopt import fmin, tpe, hp, rand, Trials

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



from lightgbm import LGBMClassifier
def objective(space):

    scores = []



    skf = StratifiedKFold(n_splits=5, random_state=99, shuffle=True)



    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        clf = LGBMClassifier(n_estimators=9999, **space) 



        clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

        y_pred = clf.predict_proba(X_val)[:,1]

        score = roc_auc_score(y_val, y_pred)

        scores.append(score)

        

    scores = np.array(scores)

    print(scores.mean())

    

    return -scores.mean()
space ={

        'max_depth': hp.choice('max_depth', np.arange(1, 50, dtype=int)),

        'subsample': hp.uniform ('subsample', 0.8, 1),

        'num_leaves':hp.choice('num_leaves', np.arange(100, 1000)),

        'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),

        'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)

    }
trials = Trials()



best = fmin(fn=objective,

              space=space, 

              algo=tpe.suggest,

              max_evals=20, 

              trials=trials, 

              rstate=np.random.RandomState(99) 

             )
"""

#パラメータチューニング

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

gkf = GroupKFold(n_splits=5, )



param_grid = {'learning_rate': [0.05],

                    'max_depth':  np.linspace(1,21,3,dtype = int),

                    'num_leaves': [100,300,900,1200],#np.linspace(100,1000,200),

                    'colsample_bytree': np.linspace(0.5, 1.0, 3),

                    'random_state': [71]}



fit_params = {"early_stopping_rounds": 20,

                    "eval_metric": 'auc',

                    "eval_set": [(X_val, y_val)]}



clf  = LGBMClassifier(n_estimators=9999, n_jobs=1)



gs = GridSearchCV(clf, param_grid, scoring='roc_auc',  

                              n_jobs=-1, cv=skf, verbose=True)



gs.fit(X_train_, y_train_, **fit_params,)

"""
# 全データで再学習

#clf_best =  LGBMClassifier(**gs.best_params_)

#clf_best.fit(X_train, y_train)

clf_best =  LGBMClassifier(**best)
%%time

scores = []

y_pred_test = np.zeros(len(X_test)) # テストデータに対する予測格納用array

skf = StratifiedKFold(n_splits=10, random_state=99, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

    X_val, y_val = X_train.iloc[test_ix], y_train.iloc[test_ix]



    clf_best.fit(X_train_, y_train_)

    y_pred = clf_best.predict_proba(X_val)[:,1]

    scores.append(roc_auc_score(y_val, y_pred))

    y_pred_test += clf_best.predict_proba(X_test)[:,1] # テストデータに対する予測値を足していく



scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())

y_pred_test /= 10 # 最後にfold数で割る
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/homework-for-students4plus/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred_test

submission.to_csv('submission.csv')
LGBMClassifier(**best)
#変数重要度

clf_best.booster_.feature_importance(importance_type='gain')

imp = DataFrame(clf_best.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp
fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(clf_best, max_num_features=50, ax=ax, importance_type='gain')