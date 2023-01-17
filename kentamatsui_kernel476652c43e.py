import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

from pandas.plotting import scatter_matrix



from sklearn.metrics import mean_squared_log_error, mean_squared_error

from sklearn.model_selection import KFold, GridSearchCV, TimeSeriesSplit,cross_val_predict

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.feature_extraction import text

from sklearn.decomposition import SparsePCA

from lightgbm import LGBMRegressor

pd.set_option('display.max_columns', 500)
df_train = pd.read_csv('../input/exam-for-students20200129/train.csv', index_col=0)

df_test = pd.read_csv('../input/exam-for-students20200129/test.csv', index_col=0)

# df_info = pd.read_csv('../input/exam-for-students20200129/country_info.csv', index_col=0)



# ターゲットログ変換

df_train.ConvertedSalary = np.log1p(df_train.ConvertedSalary)
# df_train = pd.merge(df_train,df_info,how='left',on='Country')

# df_test = pd.merge(df_test,df_info,how='left',on='Country')
len(df_train)
len(df_test)
# トレーニングデータ先頭

df_train.head()
# トレーニングデータ最後

df_train.tail()
# テストデータ先頭

df_test.head()
# テストデータ最後

df_test.tail()
# トレーニングデータ基本統計量

df_train.describe()
# テストデータ基本統計量

df_test.tail()
# トレーニングデータ型確認

df_train.dtypes
# テストデータ型確認

df_test.dtypes
# トレーニングデータnull

df_train.isnull().sum()
# テストデータnull

df_test.isnull().sum()
# plt.figure(figsize=(15,15))

# cmap = sns.color_palette("coolwarm", 200)

# sns.heatmap(df_train.corr(), square=True, annot=True, cmap=cmap)

# plt.figure(figsize=(15,15))

# cmap = sns.color_palette("coolwarm", 200)

# sns.heatmap(df_test.corr(), square=True, annot=True, cmap=cmap)
# # トレーニングデータヒストグラム

# df_train.hist(figsize=(20,15))

# plt.show()
# # テストデータヒストグラム

# df_test.hist(figsize=(20,15))

# plt.show()
# ユニーク値確認

u = 'Hobby'

df_train[u].unique()
# オブジェクト型のユニーク数確認

cats = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, df_train[col].nunique())
# ユニーク値確認

u = 'Hobby'

df_train[u].unique()
df_train.groupby(u).size()
# # ヒストグラム色分け

# f1 = 'Hobby'

# f2 = 'ConvertedSalary'



# plt.figure(figsize=[7,7])

# df_train.loc[(df_train[f1]=='No'),f2].hist(density=True, alpha=0.5, bins=50)

# df_train.loc[(df_train[f1]=='Yes'),f2].hist(density=True, alpha=0.5, bins=50)

# plt.xlabel(f2)

# plt.ylabel('density')

# plt.show()
# ユニーク値確認

u = 'OpenSource'

df_train[u].unique()
df_train.groupby(u).size()
# # ヒストグラム色分け

# f2 = 'ConvertedSalary'



# plt.figure(figsize=[7,7])

# df_train.loc[(df_train[u]=='No'),f2].hist(density=True, alpha=0.5, bins=50)

# df_train.loc[(df_train[u]=='Yes'),f2].hist(density=True, alpha=0.5, bins=50)

# plt.xlabel(f2)

# plt.ylabel('density')

# plt.show()
# ユニーク値確認

u = 'Country'

df_train[u].unique()
df_train.groupby(u).size().sort_values()

# 少数をまとめるかどうか
df_test.groupby(u).size().sort_values()
# FranceとGermanyがトレーニングデータにないためドロップ

df_train.drop('Country',axis=1,inplace=True)

df_test.drop('Country',axis=1,inplace=True)
# ユニーク値確認

u = 'Student'

df_train[u].unique()

# nanの扱い
df_train.groupby(u).size()
# # ヒストグラム色分け

# f2 = 'ConvertedSalary'



# plt.figure(figsize=[7,7])

# df_train.loc[(df_train[u]=='No'),f2].hist(density=True, alpha=0.5, bins=50)

# df_train.loc[(df_train[u].isnull()),f2].hist(density=True, alpha=0.5, bins=50)

# df_train.loc[(df_train[u]=='Yes, part-time'),f2].hist(density=True, alpha=0.5, bins=50)

# df_train.loc[(df_train[u]=='Yes, full-time'),f2].hist(density=True, alpha=0.5, bins=50)

# plt.xlabel(f2)

# plt.ylabel('density')

# plt.show()
# ユニーク値確認

u = 'Employment'

df_train[u].unique()

# nanの扱い
df_train.groupby(u).size()
# f = 'Employment'

# nf = 'target_Employment'

# skf = KFold(n_splits=5, random_state=51, shuffle=True)

# # skf = TimeSeriesSplit(n_splits=5)

# df_train[nf] = np.nan

# for i, (train_ix, val_ix) in enumerate(skf.split(df_train,df_train)):

# #     df_train.sort_values('issue_d')[nf].iloc[val_ix] = df_train.sort_values('issue_d').iloc[val_ix][f].map(df_train.sort_values('issue_d').iloc[train_ix].groupby(f).loan_condition.mean())

#     df_train[nf].iloc[val_ix] = df_train.iloc[val_ix][f].map(df_train.iloc[train_ix].groupby(f).ConvertedSalary.mean())

    

# df_test[nf] = df_test[f].map(df_train.groupby(f).ConvertedSalary.mean())
nf = 'count_Employment'



df_train[nf] = df_train[u].map(df_train.groupby(u).ConvertedSalary.count())    

df_test[nf] = df_test[u].map(df_train.groupby(u).ConvertedSalary.count())
# ユニーク値確認

u = 'FormalEducation'

df_train[u].unique()

# nanの扱い
df_train.groupby(u).size()
u = 'CompanySize'

df_train.groupby(u).size()
df_test.groupby(u).size()
# df_train[u].replace({'1,000 to 4,999 employees':1000,

#                      '10 to 19 employees':10,

#                     '10,000 or more employees':10000,

#                     '100 to 499 employees':100,

#                     '20 to 99 employees':20,

#                     '5,000 to 9,999 employees':5000,

#                     '500 to 999 employees':500,

#                     'Fewer than 10 employees':5},inplace=True)
df_train.groupby(u).size()
# df_test[u].replace({'1,000 to 4,999 employees':1000,

#                      '10 to 19 employees':10,

#                     '10,000 or more employees':10000,

#                     '100 to 499 employees':100,

#                     '20 to 99 employees':20,

#                     '5,000 to 9,999 employees':5000,

#                     '500 to 999 employees':500,

#                     'Fewer than 10 employees':5},inplace=True)
u = 'YearsCoding'

df_train.groupby(u).size()
df_test.groupby(u).size()
df_train[u].replace({'0-2 years':0,

                     '12-14 years':12,

                    '15-17 years':15,

                    '18-20 years':18,

                    '21-23 years':21,

                    '24-26 years':24,

                    '27-29 years':27,

                    '3-5 years':3,

                    '30 or more years':30,

                    '6-8 years':6,

                    '9-11 years':9},inplace=True)
df_train.groupby(u).size()
df_test[u].replace({'0-2 years':0,

                     '12-14 years':12,

                    '15-17 years':15,

                    '18-20 years':18,

                    '21-23 years':21,

                    '24-26 years':24,

                    '27-29 years':27,

                    '3-5 years':3,

                    '30 or more years':30,

                    '6-8 years':6,

                    '9-11 years':9},inplace=True)
u = 'YearsCodingProf'

df_train.groupby(u).size()
# df_train[u].replace({'0-2 years':0,

#                      '12-14 years':12,

#                     '15-17 years':15,

#                     '18-20 years':18,

#                     '21-23 years':21,

#                     '24-26 years':24,

#                     '27-29 years':27,

#                     '3-5 years':3,

#                     '30 or more years':30,

#                     '6-8 years':6,

#                     '9-11 years':9},inplace=True)
df_train.groupby(u).size()
# df_test[u].replace({'0-2 years':0,

#                      '12-14 years':12,

#                     '15-17 years':15,

#                     '18-20 years':18,

#                     '21-23 years':21,

#                     '24-26 years':24,

#                     '27-29 years':27,

#                     '3-5 years':3,

#                     '30 or more years':30,

#                     '6-8 years':6,

#                     '9-11 years':9},inplace=True)
u = 'Age'

df_train.groupby(u).size()
df_test.groupby(u).size()
# df_train[u].replace({'18 - 24 years old':18,

#                      '25 - 34 years old':25,

#                     '35 - 44 years old':35,

#                     '45 - 54 years old':45,

#                     '55 - 64 years old':55,

#                     '65 years or older':65,

#                     'Under 18 years old':15},inplace=True)
df_train.groupby(u).size()
# df_test[u].replace({'18 - 24 years old':18,

#                      '25 - 34 years old':25,

#                     '35 - 44 years old':35,

#                     '45 - 54 years old':45,

#                     '55 - 64 years old':55,

#                     '65 years or older':65,

#                     'Under 18 years old':15},inplace=True)
u = 'LastNewJob'

df_train.groupby(u).size()
df_train.groupby(u)['ConvertedSalary'].mean()
# f = 'LastNewJob'

# nf = 'target_LastNewJob'

# skf = KFold(n_splits=5, random_state=51, shuffle=True)

# # skf = TimeSeriesSplit(n_splits=5)

# df_train[nf] = np.nan

# for i, (train_ix, val_ix) in enumerate(skf.split(df_train,y_train)):

# #     df_train.sort_values('issue_d')[nf].iloc[val_ix] = df_train.sort_values('issue_d').iloc[val_ix][f].map(df_train.sort_values('issue_d').iloc[train_ix].groupby(f).loan_condition.mean())

#     df_train[nf].iloc[val_ix] = df_train.iloc[val_ix][f].map(df_train.iloc[train_ix].groupby(f).ConvertedSalary.mean())

    

# df_test[nf] = df_test[f].map(df_train.groupby(f).ConvertedSalary.mean())
u = 'Currency'

df_train.groupby(u).size()
df_test.groupby(u).size().index
# nf = 'count_Currency'



# df_train[nf] = df_train[u].map(df_train.groupby(u).ConvertedSalary.count())    

# df_test[nf] = df_test[u].map(df_train.groupby(u).ConvertedSalary.count())
# df_train.loc[~df_train[u].isin(df_test.groupby(u).size().index),u] = 'other'
df_train[u].nunique()
df_test[u].nunique()
u = 'CurrencySymbol'

df_train.groupby(u).size()
df_test.groupby(u).size().index
# df_train.loc[~df_train['CurrencySymbol'].isin(df_test.groupby(u).size().index),u] = 'other'
df_train.groupby(u)['ConvertedSalary'].mean()
nf = 'count_CurrencySymbol'



df_train[nf] = df_train[u].map(df_train.groupby(u).ConvertedSalary.count())    

df_test[nf] = df_test[u].map(df_train.groupby(u).ConvertedSalary.count())
# ヒストグラム色分け

f2 = 'ConvertedSalary'



plt.figure(figsize=[7,7])

df_train.loc[(df_train[u]=='AUD'),f2].hist(density=True, alpha=0.5, bins=50)

df_train.loc[(df_train[u]=='BTC'),f2].hist(density=True, alpha=0.5, bins=50)

df_train.loc[(df_train[u]=='CAD'),f2].hist(density=True, alpha=0.5, bins=50)

plt.xlabel(f2)

plt.ylabel('density')

plt.show()
# f = 'CurrencySymbol'

# nf = 'target_CurrencySymbol'

# skf = KFold(n_splits=5, random_state=51, shuffle=True)

# # skf = TimeSeriesSplit(n_splits=5)

# df_train[nf] = np.nan

# for i, (train_ix, val_ix) in enumerate(skf.split(df_train,df_train)):

# #     df_train.sort_values('issue_d')[nf].iloc[val_ix] = df_train.sort_values('issue_d').iloc[val_ix][f].map(df_train.sort_values('issue_d').iloc[train_ix].groupby(f).loan_condition.mean())

#     df_train[nf].iloc[val_ix] = df_train.iloc[val_ix][f].map(df_train.iloc[train_ix].groupby(f).ConvertedSalary.mean())

    

# df_test[nf] = df_test[f].map(df_train.groupby(f).ConvertedSalary.mean())
# # テキストtfidf

# f = 'emp_title'

# TXT_train = df_train[f]

# TXT_test = df_test[f]

# TXT_train.fillna('#',inplace=True)

# TXT_test.fillna('#',inplace=True)

# TXT_temp = TXT_train.append(TXT_test)

# x = 15

# tfidf = TfidfVectorizer(max_features=x,stop_words=['of','inc','and','title'],ngram_range=(1,2))

# tfidf.fit(TXT_temp)

# TXT_train = tfidf.transform(TXT_train)

# TXT_test = tfidf.transform(TXT_test)
# オブジェクト型のユニーク数確認

cats = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, df_train[col].nunique())

encoder = OrdinalEncoder(cols=cats)

df_train[cats] = encoder.fit_transform(df_train[cats])

df_test[cats] = encoder.transform(df_test[cats])
# 上位95%の特徴量

feature = ['CompanySize',

 'LastNewJob',

 'CurrencySymbol',

 'YearsCoding',

 'count_Employment',

 'SalaryType',

 'Currency',

 'Employment',

 'MilitaryUS',

 'YearsCodingProf',

 'count_CurrencySymbol',

 'Age',

 'RaceEthnicity',

 'DevType',

 'Student',

 'AssessBenefits2',

 'JobContactPriorities3',

 'CareerSatisfaction',

 'FrameworkWorkedWith',

 'NumberMonitors',

 'CheckInCode',

 'AssessBenefits6',

 'AssessJob1',

 'AssessBenefits11',

 'JobEmailPriorities5',

 'AssessBenefits8',

 'OperatingSystem',

 'AssessBenefits9',

 'FormalEducation',

 'AssessBenefits7',

 'AssessJob5',

 'AssessJob4',

 'AssessBenefits4',

 'EducationParents',

 'WakeTime',

 'AssessBenefits1',

 'JobEmailPriorities6',

 'AssessBenefits10',

 'AssessJob10',

 'CommunicationTools',

 'JobContactPriorities4',

 'JobContactPriorities1',

 'JobContactPriorities2',

 'AdsPriorities5',

 'UndergradMajor',

 'AssessJob3',

 'AssessJob8',

 'AssessJob6',

 'AssessJob2',

 'AssessJob7',

 'AssessBenefits5',

 'AdsActions',

 'AssessBenefits3',

 'Exercise',

 'AdsPriorities2',

 'AdsPriorities3',

 'JobEmailPriorities1',

 'AdsAgreeDisagree3',

 'AssessJob9',

 'StackOverflowVisit',

 'JobEmailPriorities2',

 'AdsPriorities1',

 'AdsPriorities6',

 'Gender',

 'JobEmailPriorities7',

 'JobEmailPriorities3',

 'AdsPriorities7',

 'UpdateCV',

 'AdsPriorities4',

 'StackOverflowDevStory',

 'HopeFiveYears',

 'ErgonomicDevices']


# 説明変数・目的変数分割

y_train = df_train.ConvertedSalary

X_train = df_train.drop(['ConvertedSalary'],axis=1)

X_test = df_test.copy()

# X_train = df_train[feature]

# X_test = df_test[feature]

# 交差検定

scores = []

y_pred_test = 0

skf = KFold(n_splits=5, random_state=60, shuffle=True)

for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    # clf = GradientBoostingClassifier(random_state=71,n_estimators=170,max_depth=3,learning_rate=0.1)

    clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    clf.fit(X_train_, y_train_, early_stopping_rounds=50, eval_metric='rmse', eval_set=[(X_val, y_val)])

    y_pred = clf.predict(X_val)

    y_pred_test += clf.predict(X_test)

    score = mean_squared_error(y_val, y_pred)

    scores.append(score)

    print('CV Score of Fold_%d is %f' % (i, score))
skf = KFold(n_splits=5, random_state=40, shuffle=True)

for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    # clf = GradientBoostingClassifier(random_state=71,n_estimators=170,max_depth=3,learning_rate=0.1)

    clf = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    clf.fit(X_train_, y_train_, early_stopping_rounds=50, eval_metric='rmse', eval_set=[(X_val, y_val)])

    y_pred = clf.predict(X_val)

    y_pred_test += clf.predict(X_test)

    score = mean_squared_error(y_val, y_pred)

    scores.append(score)

    print('CV Score of Fold_%d is %f' % (i, score))
# 交差検定確認

print(np.mean(scores))

print(scores)

# 1.638953053626206

# 1.658142347739124

# 1.6828459262477922

# 1.67482726579517

# 1.6695032760072563

# 1.614970454233402

# 1.6102697591205206

# 1.6258962003536748

# 1.6237746357435543

# 1.6198626982797617
# 特徴量重要度確認

fti = clf.booster_.feature_importance(importance_type='gain')

df_fti = pd.DataFrame()

df_fti['feature'] = X_train.columns

df_fti['importance'] = fti

plt.figure(figsize=[60,7])

df_fti.plot(x='feature',y='importance',kind='bar')
df_fti.sort_values('importance',ascending=False)
df_fti['importance'].sum()*0.95
df_fti.sort_values('importance',ascending=False)[['importance']].head(72).sum()
df_fti.sort_values('importance',ascending=False).head(72).feature.to_list()
y_pred_test
# sample submissionを読み込んで、予測値を代入の後、保存する

y_pred = y_pred_test/10

y_pred = np.expm1(y_pred)

submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)



submission.ConvertedSalary = y_pred

submission.to_csv('submission.csv')