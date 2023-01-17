import numpy as np

import scipy as sp

import pandas as pd

import category_encoders as ce

from pandas import DataFrame, Series

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.dummy import DummyClassifier



import lightgbm as lgb

from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from keras.layers import Input, Dense, Dropout, BatchNormalization

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
#df_train = pd.read_csv('C:/Users/big/Desktop/DRアカデミー/train_small.csv', index_col=0)

#df_test = pd.read_csv('C:/Users/big/Desktop/DRアカデミー/test.csv', index_col=0)

#st_data = pd.read_csv('C:/Users/big/Desktop/DRアカデミー/statelatlong.csv')

#gdp_data = pd.read_csv('C:/Users/big/Desktop/DRアカデミー/US_GDP_by_State.csv')

#spi_data = pd.read_csv('C:/Users/big/Desktop/DRアカデミー/spi.csv')

df_train = pd.read_csv('../input/train.csv', index_col=0)

df_test = pd.read_csv('../input/test.csv', index_col=0)

st_data = pd.read_csv('../input/statelatlong.csv')

gdp_data = pd.read_csv('../input/US_GDP_by_State.csv')

spi_data = pd.read_csv('../input/spi.csv')
df_train.info(), df_test.info()
# df_train = df_train.fillna(df_train.mode().iloc[0])

# df_train.info()

df_train.shape
df_train.head()
df_train.info(), df_test.info()
# x,yへの分割

#y_train = df_train.loan_condition

#X_train = df_train.drop(['loan_condition', 'issue_d'], axis=1)



#X_test = df_test.drop(['issue_d'], axis=1)
# 外部データのマージ（州関係同志）

gdp_data_1 =  gdp_data.groupby('State').mean()
gdp_data.head()
gdp_data_1.head()
spi_data.head()
df_train.issue_d.head()
# 外部データのマージ（メインと州関係）

st_gdp = pd.merge(st_data, gdp_data_1, left_on='City', right_on='State')



df_train = pd.merge(df_train ,st_gdp, left_on='addr_state', right_on='State', how='left')

df_test = pd.merge(df_test ,st_gdp, left_on='addr_state', right_on='State', how='left')



#df_train = pd.merge(df_train ,spi_data, left_on='issue_d', right_on='date', how='left')

#df_test = pd.merge(df_test ,spi_data, left_on='issue_d', right_on='date', how='left')



#X_train = pd.merge(X_train ,st_gdp, left_on='addr_state', right_on='State', how='left')

#X_test = pd.merge(X_test ,st_gdp, left_on='addr_state', right_on='State', how='left')



#X_train = pd.merge(X_train ,spi_data, left_on='issue_d', right_on='date', how='left')

#X_test = pd.merge(X_test ,spi_data, left_on='issue_d', right_on='date', how='left')
df_train.head()
# x,yへの分割

y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition', 'issue_d'], axis=1)



X_test = df_test.drop(['issue_d'], axis=1)
X_train.head()
X_train.drop(['year'], axis=1, inplace=True)

X_test.drop(['year'], axis=1, inplace=True)
X_train.head()
# カテゴリ特徴量

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        print(col, X_train[col].nunique())
cats.remove('emp_title')

cats
#X_train[cats] = X_train[cats].fillna('NA')

#X_test[cats] = X_test[cats].fillna('NA')

X_train[cats] = X_train[cats].fillna(X_train[cats].mode().iloc[0])

X_test[cats] = X_test[cats].fillna(X_test[cats].mode().iloc[0])
# 数値特徴量

nums = []

for col in X_train.columns:

    if X_train[col].dtype != 'object':

        nums.append(col)

        print(col, X_train[col].nunique())
for i in nums: 

    #X_train[i] = X_train[i].fillna(X_train[i].mode().iloc[0])

    #X_test[i] = X_test[i].fillna(X_test[i].mode().iloc[0])

    X_train[i] = X_train[i].fillna(X_train[i].median())

    X_test[i] = X_test[i].fillna(X_test[i].median())
# 対数変換

X_train.annual_inc = X_train.annual_inc.apply(np.log1p)

X_test.annual_inc = X_test.annual_inc.apply(np.log1p)
X_train.annual_inc.head()
X_train.head()
# 標準化

scaler = StandardScaler()

scaler.fit(X_train[nums])



X_train[nums] = scaler.transform(X_train[nums])

X_test[nums] = scaler.transform(X_test[nums])
X_train.head()
# emp_titleの除外

TXT_train = X_train.emp_title

X_train.drop(['emp_title'], axis=1, inplace=True)



TXT_test = X_test.emp_title

X_test.drop(['emp_title'], axis=1, inplace=True)
## emp_titleのTFIDF

#TXT_train.fillna('#', inplace=True)

#TXT_test.fillna('#', inplace=True)



#tfidf = TfidfVectorizer(max_features=1000)



#TXT_train = tfidf.fit_transform(TXT_train)

#TXT_test = tfidf.fit_transform(TXT_test)



#TXT_train.todense()
X_train.head()
oe = ce.OrdinalEncoder(cols=cats, return_df = False)

X_train[cats] = oe.fit_transform(X_train[cats])
for i in cats:

    print(X_train[i].dtype)
X_test[cats] = oe.fit_transform(X_test[cats])
df_train.loan_amnt.count()*0.01
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=71)



clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

        importance_type='split', learning_rate=0.05, max_depth=-1,

        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

        n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

        random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

        subsample=0.9, subsample_for_bin=200000, subsample_freq=0)

 

clf.fit(X_train, y_train, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])

#y_pred = clf.predict_proba(X_val)[:,1]

#score = roc_auc_score(y_val, y_pred)

y_pred = clf.predict_proba(X_test)[:,1]
#scores = []



#skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



#for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

#    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

#    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

#        importance_type='split', learning_rate=0.05, max_depth=-1,

#        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#        n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

#        random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

#        subsample=0.9, subsample_for_bin=200000, subsample_freq=0)

 

#    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])

#    y_pred = clf.predict_proba(X_val)[:,1]

#    score = roc_auc_score(y_val, y_pred)

#    scores.append(score)



#print(sum(scores)/len(scores))    
# 全データで再学習し、testに対して予測する

#parameters = {CV Score of Fold_0 is 0.651084

#  "loss":["deviance"],

# "learning_rate": [0.1],

# "min_samples_split": np.linspace(0.1, 0.5, 12),

#   "min_samples_leaf": np.linspace(0.1, 0.5, 12),

#   "max_depth":[8],

#   "max_features":["sqrt"],

#   "n_estimators":[100]

#   }

    

#clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

#clf.fit(X_train, y_train)



#y_pred = clf.predict_proba(X_test)[:,1] # predict_probaで確率を出力する
y_pred.shape
# sample submissionを読み込んで、予測値を代入の後、保存する

#submission = pd.read_csv('C:/Users/big/Desktop/DRアカデミー/sample_submission.csv', index_col=0)

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')
X_train.shape
X_test.shape