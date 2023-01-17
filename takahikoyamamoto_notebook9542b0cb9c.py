import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.preprocessing import StandardScaler, MinMaxScaler, quantile_transform

import seaborn as sns

import datetime



import gc

import warnings

warnings.filterwarnings('ignore')



from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss, roc_curve, confusion_matrix, plot_roc_curve

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from sklearn.preprocessing import LabelEncoder



import lightgbm as lgb

from lightgbm import LGBMClassifier



import lightgbm as lgb

from lightgbm import LGBMRegressor



# 必要なライブラリのインポート

from sklearn.ensemble import RandomForestRegressor



# NN

# from tensorflow.keras.layers import Dense ,Dropout, BatchNormalization, Input, Embedding, SpatialDropout1D, Reshape, Concatenate

# from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.models import Model

# from tensorflow.keras.callbacks import EarlyStopping

# from tensorflow.keras.metrics import AUC



from hyperopt import fmin, tpe, hp, rand, Trials

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



from lightgbm import LGBMClassifier

# import xgboost as xgb
# df_train = pd.read_csv('../input/exam-for-students20200923/train.csv', index_col=0, skiprows=lambda x: x%20!=0)

df_train = pd.read_csv('../input/exam-for-students20200923/train.csv', index_col=0)

df_test = pd.read_csv('../input/exam-for-students20200923/test.csv', index_col=0)
df_country = pd.read_csv('../input/exam-for-students20200923/country_info.csv', index_col=0)
df_train.shape
df_test.shape
df_country.shape
#  結合

df_train_country=pd.merge(df_train, df_country, left_on='Country', right_on='Country', how='left')

df_test_country=pd.merge(df_test, df_country, left_on='Country', right_on='Country', how='left')
df_train_country.shape
df_test_country.shape
df_train=df_train_country

df_test=df_test_country
pd.set_option('display.max_rows', 150)

pd.set_option('display.max_columns', 150)
df_train.dtypes
df_train.head(2)
df_test.head(2)
y_train = df_train.ConvertedSalary.apply(np.log1p) #目的変数

X_train = df_train.drop(['ConvertedSalary'], axis=1) #説明変数



X_test = df_test
y_train.head()
# dtypeがobject（数値でないもの）のカラム名とユニーク数を確認してみましょう。

cats = []

nums = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

    else:

        nums.append(col)

    

        print(col, X_train[col].nunique())
cats
nums
df_train.describe()
df_test.describe()
# f = 'AssessJob1'

f = nums



for f in nums:

    plt.figure(figsize=[5,5])

    df_train[f].hist(density=True, alpha=0.5, bins=50,color='r') # α:透過率 colorがなくてもいい

    df_test[f].hist(density=True, alpha=0.5, bins=50,color='b') # colorがなくてもいい

    # testデータに対する可視化を記入してみましょう

    plt.xlabel(f)

    plt.ylabel('density') #　density:絶対値でなく相対値

    plt.show()
# # まずは数値特徴量だけを取り出してみます。

# X_train_num = X_train.drop(cats, axis=1).fillna(-9999)

# X_test_num = X_test.drop(cats, axis=1).fillna(-9999)
# y_train
# X_train=X_train_num

# X_test=X_test_num
# #数値の欠損値を中央値で埋める

# X_train.fillna(X_train.median(), inplace=True)

# X_test.fillna(X_train.median(), inplace=True)
# X_train.head(2)
#target enc

target = 'ConvertedSalary'

X_temp = pd.concat([X_train, y_train], axis=1)



for col in cats:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test[col] = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    skf = KFold(n_splits=5, random_state=71, shuffle=True)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train[col]  = enc_train
X_train.describe()
#カテゴリの欠損値を平均値で埋める

#数値の欠損値も平均値で埋める

X_train.fillna(X_train.mean(), axis=0, inplace=True)

X_test.fillna(X_train.mean(), axis=0, inplace=True)



# # -9999で埋める

# X_train.fillna(-9999, axis=0, inplace=True)

# X_test.fillna(-9999, axis=0, inplace=True)
X_train.describe()
X_train
fold=10
# LightGB CV Averaging random1

scores = []

y_pred_test1 = np.zeros(len(X_test)) # テストデータに対する予測格納用array



skf = KFold(n_splits=fold, random_state=91, shuffle=True)



# for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

#     X_train_, y_train_, text_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, y_val, text_val = X_train.iloc[test_ix], y_train.iloc[test_ix ]

for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



#     clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7000000000000001,

#                                 importance_type='split', learning_rate=0.05, max_depth=10,

#                                 min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#                                 n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

#                                 random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#                                 subsample=0.9929417385040324, subsample_for_bin=200000, subsample_freq=0)

    clf = LGBMRegressor(boosting_type='gbdt', class_weight=None,

                                importance_type='split')

#     clf = RandomForestRegressor()



    

    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='rmse', eval_set=[(X_val, y_val)])

#     clf.fit(X_train_, y_train_)

    y_pred = clf.predict(X_val)

    scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    

    y_pred_test1 += clf.predict(X_test) # テストデータに対する予測値を足していく
y_pred_test1
scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())

score_m1 = scores.mean()



y_pred_test1 /= fold # 最後にfold数で割る

y_pred_m1 = y_pred_test1
y_pred_m1
y_pred_m1 = np.exp(y_pred_m1) - 1
y_pred_m1
# LightGB CV Averaging random2

scores = []

y_pred_test2 = np.zeros(len(X_test)) # テストデータに対する予測格納用array



skf = KFold(n_splits=fold, random_state=101, shuffle=True)



# for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

#     X_train_, y_train_, text_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

#     X_val, y_val, text_val = X_train.iloc[test_ix], y_train.iloc[test_ix ]

for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]





    clf = LGBMRegressor(boosting_type='gbdt', class_weight=None,

                                importance_type='split')

    

    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='rmse', eval_set=[(X_val, y_val)])

#     clf.fit(X_train_, y_train_)

    y_pred = clf.predict(X_val)

    scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    

    y_pred_test2 += clf.predict(X_test) # テストデータに対する予測値を足していく
scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())

score_m2 = scores.mean()



y_pred_test2 /= fold # 最後にfold数で割る

y_pred_m2 = y_pred_test2
y_pred_m2 = np.exp(y_pred_m2) - 1
y_pred_m2
# LightGB CV Averaging random3

scores = []

y_pred_test3 = np.zeros(len(X_test)) # テストデータに対する予測格納用array



skf = KFold(n_splits=fold, random_state=111, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]





    clf = LGBMRegressor(boosting_type='gbdt', class_weight=None,

                                importance_type='split')

    

    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='rmse', eval_set=[(X_val, y_val)])



    y_pred = clf.predict(X_val)

    scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    

    y_pred_test3 += clf.predict(X_test) # テストデータに対する予測値を足していく
scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())

score_m3 = scores.mean()



y_pred_test3 /= fold # 最後にfold数で割る

y_pred_m3 = y_pred_test3
y_pred_m3 = np.exp(y_pred_m3) - 1
y_pred_m3
# 平均スコアを算出 

# np.array(scores).mean()

(score_m1+score_m2+score_m3)/3
# アンサンブル

y_pred_m_all = (y_pred_m1+y_pred_m2+y_pred_m3)/3
submission = pd.read_csv('../input/exam-for-students20200923/sample_submission.csv', index_col=0)
submission.ConvertedSalary = y_pred_m_all

submission.to_csv('submission.csv')

submission.head() # まずは初回submitしてみましょう！これからこのモデルの改善を進めていくことになります。
submission.head() # まずは初回submitしてみましょう！これからこのモデルの改善を進めていくことになります。
s1 = pd.DataFrame(y_pred_m1)
s1
s2 = pd.DataFrame(y_pred_m2)
s2
s3 = pd.DataFrame(y_pred_m3)
s3
s_all = pd.DataFrame(y_pred_m_all)
s_all