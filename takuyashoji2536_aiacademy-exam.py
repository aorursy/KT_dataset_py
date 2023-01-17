import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import scipy as sp

from pandas import DataFrame, Series

import re



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



# Encord

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, quantile_transform

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



# Model

from sklearn.linear_model  import LinearRegression, ElasticNet, Lasso, Ridge

from xgboost import XGBRegressor 

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor



# Validation

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
base_path = '/kaggle/input/exam-for-students20200923'

train_path = f'{base_path}/train.csv'

test_path = f'{base_path}/test.csv'

country_info_path = f'{base_path}/country_info.csv'

sample_submission_path = f'{base_path}/sample_submission.csv'



TARGET = 'ConvertedSalary'

SEED = 71
train_df = pd.read_csv(train_path)

test_df = pd.read_csv(test_path)

country_info_df = pd.read_csv(country_info_path)
# country_info_df 数値カラム加工

for col in country_info_df.columns:

    if col == 'Country' or col == 'Region' or country_info_df[col].dtype != 'object':

        continue

    else:

        country_info_df[col] = country_info_df[col].str.replace(',', '.').astype(float)



country_info_df.head()
train_df_1 = pd.merge(train_df, country_info_df, on=['Country'], how='left')

test_df_1 = pd.merge(test_df, country_info_df, on=['Country'], how='left')
train_df_1["Region"]
train_df_1 = train_df_1[train_df_1["Region"] == 'WESTERN EUROPE                     ']
train_df_1.head()
# X,y へ分割

y_train = []

X_train = []



y_train = train_df_1[TARGET]

X_train = train_df_1.drop([TARGET], axis=1)

X_test = test_df_1



X_all = pd.concat([X_train,X_test], axis=0)
X_all['WakeTime'].unique()
# Age の序列をマッピング

X_all=X_all.replace({'Age':{'Nan': 0,'Under 18 years old':1, '18 - 24 years old':2,

                                 '25 - 34 years old':3, '35 - 44 years old':4,

                                 '45 - 54 years old':5, '55 - 64 years old':6,

                                 '65 years or older':7}})

X_all['Age'] = X_all['Age'].astype('object')



# CompanySize の序列をマッピング

X_all=X_all.replace({'CompanySize':{'Nan': 0, 'Fewer than 10 employees':1, '10 to 19 employees':2,

                                    '20 to 99 employees':3,'100 to 499 employees':4, '500 to 999 employees':5, 

                                    '1,000 to 4,999 employees':6,'5,000 to 9,999 employees':7,'10,000 or more employees':8}})

X_all['CompanySize'] = X_all['CompanySize'].astype('object')



# YearsCoding の序列をマッピング

X_all=X_all.replace({'YearsCoding':{'Nan': 0,'0-2 years':1,'3-5 years':2, '6-8 years':3, '9-11 years':4, 

                                    '12-14 years':5,'15-17 years':6, '18-20 years':7,'21-23 years':8,

                                    '24-26 years':9, '27-29 years':10,'30 or more years':11}})

X_all['YearsCoding'] = X_all['YearsCoding'].astype('object')



# YearsCodingProf の序列をマッピング

X_all=X_all.replace({'YearsCodingProf':{'Nan': 0,'0-2 years':1,'3-5 years':2, '6-8 years':3, '9-11 years':4, 

                                    '12-14 years':5,'15-17 years':6, '18-20 years':7,'21-23 years':8,

                                    '24-26 years':9, '27-29 years':10,'30 or more years':11}})

X_all['YearsCodingProf'] = X_all['YearsCodingProf'].astype('object')



# LastNewJob の序列をマッピング

X_all=X_all.replace({'LastNewJob':{'Nan': 0,"I've never had a job":1, 'Less than a year ago':2,'Between 1 and 2 years ago':3,

                                   'Between 2 and 4 years ago':4, 'More than 4 years ago':5}})

X_all['LastNewJob'] = X_all['LastNewJob'].astype('object')



# WakeTime の序列をマッピング

X_all=X_all.replace({'WakeTime':{'Nan': 0, 'Before 5:00 AM':1, 'Between 5:00 - 6:00 AM':2,'Between 6:01 - 7:00 AM':3,

                                 'Between 7:01 - 8:00 AM':4, 'Between 8:01 - 9:00 AM':5, 'Between 9:01 - 10:00 AM':6,

                                 'Between 10:01 - 11:00 AM':7, 'Between 11:01 AM - 12:00 PM':8, 'After 12:01 PM':9,

                                 'I do not have a set schedule':10, 'I work night shifts':11}})

X_all['WakeTime'] = X_all['WakeTime'].astype('object')

X_all.dtypes
X_all.head()
# 各カラムでの欠損値の有無を保持

for col in X_all.columns:

    X_all[col+"_null"] = (X_all[col].isnull()*1).astype('object')
# ;の数をカウント

count_col = ['DevType', 'CommunicationTools', 'FrameworkWorkedWith']



for col in count_col:

    X_all[col] = X_all[col].astype(str)

    X_all[col + '_count'] = X_all[col].apply(lambda x: len(re.split(';', x)))
X_all
cat = []

num = []



for col in X_all.columns:

    if X_all[col].dtype == 'object':

        if X_all[col].nunique() != 1: # ユニーク値しかないデータは除外

            cat.append(col)

            print(col, X_all[col].nunique())

    else:

        if col != 'DevType':

            num.append(col)



cat.remove('DevType')



cat_all = X_all[cat]

num_all = X_all[num].fillna(-9999)



DevType_all = X_all.DevType
cat_all
def rank_gauss(df, col):

    df[col] = quantile_transform(df[col], n_quantiles=100, random_state=0, output_distribution='normal')

    return df



def ordinal_encoding(df, col):

    for col in tqdm(col):

        oe = OrdinalEncoder(return_df=False)

        df[col] = oe.fit_transform(df[col])

    return df



def count_encoder(df, col):

    for col in tqdm(col):

        summary = df[col].value_counts()

        df[col+"_cnt"] = df[col].map(summary).astype('object')

    return df



def text_encoder(df, df_txt):

    # DevType

    df_txt.fillna('#', inplace=True)

    tfidf1 = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=300, ngram_range=(1,2))

    df_txt = tfidf1.fit_transform(df_txt.fillna('#'))

    df = pd.concat([df, pd.DataFrame(df_txt.todense(), index=df.index)], axis=1)

    return df
def feature_engineering(df_num, df_cat, df_txt, num, cat):

    df_num = rank_gauss(df_num, num)

    df_cat = ordinal_encoding(df_cat, cat)

    df_cat = count_encoder(df_cat, cat)

    df = pd.concat([df_num, df_cat], axis=1)

    df = text_encoder(df, df_txt)

    return df
X_all = feature_engineering(num_all, cat_all, DevType_all, num, cat)



X_all
X_all.fillna(-9999, inplace=True)
# トレーニングデータ・テストデータに分割

X_train = X_all.iloc[:X_train.shape[0], :]

X_test = X_all.iloc[X_train.shape[0]:, :]



del X_all

gc.collect()
# 学習用と検証用に分割する

X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.05, random_state=71)



clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                     importance_type='split', learning_rate=0.05, max_depth=-1,

                     min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                     n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                     random_state=SEED, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                     subsample=0.9, subsample_for_bin=200000, subsample_freq=0)
clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='RMSE', eval_set=[(X_val, y_val)])
# 特徴量インパクト集計

imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp
特徴量インパクトをしきい値としてカラムを限定

use_col = imp[imp['importance'] > 10000].index

use_col = imp.index[:100] # 変数重要度で特徴量を絞り込んでみましょう

use_col
# トレーニングデータ・テストデータに分割

X_train = X_train[use_col]

X_test = X_test[use_col]
# # 学習用と検証用に分割する

# X_train_, X_val, y_train_, y_val= train_test_split(X_train, y_train, test_size=0.05, random_state=71)



# clf_names = ["LinearRegression",

#              "ElasticNet",

#              "Lasso",

#              "Ridge",

#              "LGBMRegressor",

#              "CatBoostRegressor",

#              "XGBRegressor",

#             ]

 

# clf_params = ["",

#               "max_iter=1000, tol=0.0001",

#               "max_iter=1000, tol=0.0001",

#               "",

#               "boosting_type='gbdt', num_leaves=31, learning_rate=0.1, n_estimators=100",

#               "logging_level='Silent'",

#               "max_depth=3, learning_rate=0.1, n_estimators=100",

#              ]



# def sklearn_model(X_train, y_train):    

#     models = list()

#     model = None # もっと精度が高いモデル

#     total = 0.0

#     name  = ""

#     for i in range(len(clf_names)):

#         clf = eval("%s(%s)" % (clf_names[i], clf_params[i]))

#         clf.fit(X_train, y_train)

#         score = clf.score(X_train, y_train)

#         print('%s Accuracy:' % clf_names[i], score)

#         models.append(clf)

#         if total <= score:

#              total = score

#              model = clf

#              name  = clf_names[i]

#     print('%s was selected' % name)

#     return models, model



# # 学習データとテストデータを統合

# models, model = sklearn_model(X_train, y_train)
# LightGBM

scores = []



skf = StratifiedKFold(n_splits=5, random_state=72, shuffle=True)

y_tests = np.zeros(len(X_test.index))



for SEED in [9, 42, 71, 88, 99]:

    clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                         importance_type='split', learning_rate=0.05, max_depth=-1,

                         min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                         n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                         random_state=SEED, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                         subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='rmse', eval_set=[(X_val, np.log1p(y_val))])

        score = mean_squared_error(np.log1p(y_val), np.log1p(y_pred))**0.5

        y_pred = np.expm1(clf.predict(X_val))

        scores.append(score)

        y_tests += np.expm1(clf.predict(X_test))
y_pred = (y_tests/len(scores)).round(-2)

y_pred
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv(sample_submission_path, index_col=0)



submission.ConvertedSalary = y_pred

submission.to_csv('submission.csv')