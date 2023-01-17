import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler





import lightgbm as lgb

from lightgbm import LGBMClassifier
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。

# df_train = pd.read_csv('train.csv', index_col=0, skiprows=lambda x: x%20!=0)

df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0)

#df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0)

df_test = pd.read_csv('../input/homework-for-students4plus/test.csv', index_col=0)
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
X_train.shape,X_test.shape
X_train.head()
X_train['home_ownership'].describe()
# このあとに必要なカラムがcats送りにされてしまうので、ここで先に欲しいカラムを捌きます
X_train = X_train.drop(['emp_title', 'title'], axis=1)

X_test = X_test.drop(['emp_title', 'title'], axis=1)
from sklearn.preprocessing import LabelEncoder
# 借入目的のエンコーディング

encoder = LabelEncoder()

X_train['purpose'] = encoder.fit_transform(X_train['purpose'].values)

X_test['purpose'] = encoder.transform(X_test['purpose'].values)
# サブグレードのエンコーディング

encoder = LabelEncoder()

X_train['sub_grade'] = encoder.fit_transform(X_train['sub_grade'].values)

X_test['sub_grade'] = encoder.transform(X_test['sub_grade'].values)
def map_emp_length(x):

    dic = {

        np.nan: np.nan,

        '< 1 year': 0.5,

        '1 year': 1,

        '2 years': 2,

        '3 years': 3,

        '4 years': 4,

        '5 years': 5,

        '6 years': 6,

        '7 years': 7,

        '8 years': 8,

        '9 years': 9,

        '10+ years': 20,

    }

    return dic[x]





for xs in [X_train, X_test]:

    del xs['grade']

    xs['emp_length'] = xs['emp_length'].map(map_emp_length)
# dtypeがobject（数値でないもの）のカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
oe = OrdinalEncoder(cols=cats, return_df=False)



X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])
f = 'annual_inc'

scaler = StandardScaler()

scaler.fit(X_train[[f]])
scaler.transform(X_test[['loan_amnt']])
from datetime import datetime as dt

from datetime import timedelta
# 年収に対する預金比率

# revo_depo = X_train['revol_bal'] / X_train['tot_cur_bal']

# revo_depo = X_test['revol_bal'] / X_test['tot_cur_bal']
X_train2 = X_train

X_test2 = X_test
X_train2['incom_depo']=np.nan

X_test2['incom_depo']=np.nan
def kakezan2(x):

    try:

        return x.loc['annual_inc'] / x.loc['tot_cur_bal'].round(7)

    except:ValueError
def kakezan3(x):

    try:

        return x.loc['annual_inc'] / x.loc['tot_cur_bal'].round(7)

    except:ValueError
X_train2['incom_depo'] = X_train2.apply(kakezan2, axis=1)

X_test2['incom_depo'] = X_test2.apply(kakezan3, axis=1)
X_train2['incom_depo']=X_train2['incom_depo'].replace([np.inf, -np.inf], np.nan).fillna(3000)

X_test2['incom_depo']=X_test2['incom_depo'].replace([np.inf, -np.inf], np.nan).fillna(3000)
X_train2['incom_depo']
X_train['incom_depo']
X_test2['incom_depo']
# 返済比率 2

# revo_depo = X_train['revol_bal'] / X_train['tot_cur_bal']

# revo_depo = X_test['revol_bal'] / X_test['tot_cur_bal']
X_train2['dti2']=np.nan

X_test2['dti2']=np.nan
# ( dti * ( annual_inc / 12) + installment ) / annual_inc * 12
# ( X_train2['dti'] * ( X_train2['annual_inc'] / 12) + X_train2['installment'] ) / X_train2['annual_inc'] * 12
def kakezan4(x):

    try:

        return ((x.loc['dti'] * x.loc['annual_inc'] / 12) + x.loc['installment']) / x.loc['annual_inc'] * 12 

    except:ValueError
def kakezan5(x):

    try:

        return ((x.loc['dti'] * x.loc['annual_inc'] / 12) + x.loc['installment']) / x.loc['annual_inc'] * 12

    except:ValueError
X_train2['dti2'] = X_train2.apply(kakezan4, axis=1)

X_test2['dti2'] = X_test2.apply(kakezan5, axis=1)
X_train = X_train2

X_test = X_test2
X_train['loan_amnt'] = X_train['loan_amnt'].apply(np.log1p)

X_test['loan_amnt'] = X_test['loan_amnt'].apply(np.log1p)
# まずは数値特徴量だけを取り出してみます。

X_train_num = X_train.drop(cats, axis=1).fillna(-9999)

X_test_num = X_test.drop(cats, axis=1).fillna(-9999)
X_train
# 以下を参考に自分で書いてみましょう 

X_train = X_train.fillna(X_train.median())

X_test = X_test.fillna(X_test.median())



# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
X_train.fillna(X_train.mean(), axis=0, inplace=True)

X_test.fillna(X_train.mean(), axis=0, inplace=True)
X_train.describe()
# まずは数値特徴量だけでモデリングしてスコアを見てみます。

# 交差検定 (Cross Validation = CV) の詳細については次回より詳しく取り扱います。

# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。



# scores = []



# skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



# for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train_num, y_train))):

#     X_train_, y_train_ = X_train_num.values[train_ix], y_train.values[train_ix]

#      X_val, y_val = X_train_num.values[test_ix], y_train.values[test_ix]

    

    

#     clf = GradientBoostingClassifier()

    

#     clf.fit(X_train_, y_train_)

#     y_pred = clf.predict_proba(X_val)[:,1]

#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)

    

#     print('CV Score of Fold_%d is %f' % (i, score))
# 平均スコアを算出

# np.array(scores).mean()
target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

# ここでは理解のために自分で交差検定的に実施するが、xfeatなどを用いても良い

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)
enc_train.describe()
enc_test.describe()
# この下のやつは回さんで良くないか
# 以下を参考に自分で書いてみましょう 

X_train = X_train.fillna(X_train.median())

X_test = X_test.fillna(X_test.median())



# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
X_test
# 今度はTarget Encoding

target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



for col in cats:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test[col] = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train[col]  = enc_train
X_train.fillna(X_train.mean(), axis=0, inplace=True)

X_test.fillna(X_train.mean(), axis=0, inplace=True)
# 今度はカテゴリ特徴量も追加してモデリングしてみましょう。

# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

scores = []

y_pred_test = np.zeros(len(X_test))



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    

    clf = LGBMClassifier()

    

    

    clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(roc_auc_score(y_val, y_pred))

    y_pred_test +=  clf.predict_proba(X_test)[:,1]



scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())

y_pred_test /= 5 # 最後にfold数で割る
# 平均スコアを算出

np.array(scores).mean()
y_pred_test
y_pred
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/homework-for-students4plus/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred_test

submission.to_csv('submission.csv')
import eli5



eli5.show_weights(clf, feature_names=list(X_train.columns))