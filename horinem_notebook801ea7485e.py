# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gc

import numpy as np

import pandas as pd

import scipy as sp

import lightgbm as lgb

import xgboost as xgb

import seaborn as sns

import math

import matplotlib.pyplot as plt

import os



from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_squared_error

from category_encoders import OrdinalEncoder



from pandas import DataFrame, Series

from tqdm import tqdm_notebook as tqdm



from sklearn import metrics

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GroupKFold

from sklearn.model_selection import StratifiedKFold, KFold



#表示設定

pd.set_option('display.max_columns', 10000)

pd.set_option('display.max_rows', 10000)
path='..input/exam-for-students20200923'
df_train = pd.read_csv('/kaggle/input/exam-for-students20200923/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/exam-for-students20200923/test.csv', index_col=0)
"""

/kaggle/input/exam-for-students20200923/survey_dictionary.csv

/kaggle/input/exam-for-students20200923/sample_submission.csv

/kaggle/input/exam-for-students20200923/test.csv

/kaggle/input/exam-for-students20200923/country_info.csv

/kaggle/input/exam-for-students20200923/train.csv

"""

df_sub = pd.read_csv('/kaggle/input/exam-for-students20200923/sample_submission.csv')

country_info = pd.read_csv('/kaggle/input/exam-for-students20200923/country_info.csv', index_col=0)
#ConvertedSalary
#　要素数確認

display(df_train.shape)

display(df_test.shape)
# 列のデータ型を確認

df_train.dtypes
display(df_train.head(10))

display(df_test.head(10))
display(country_info.head(3))
# 外れ値を確認

df_train.boxplot('ConvertedSalary')
# カテゴリ変数のユニーク数を確認

# ユニーク数が多いカテゴリ変数はテキストとして処理するなど検討

cat_col = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cat_col.append(col)

        print(col, df_train[col].nunique())
# カテゴリ変数のユニーク数を確認

# ユニーク数が多いカテゴリ変数はテキストとして処理するなど検討

cat_col = []

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        cat_col.append(col)

        print(col, df_test[col].nunique())
# 欠損値を確認

df_train.isnull().sum()
# 相関行列を確認

df_corr = df_train.corr()

fig, ax = plt.subplots(figsize=(12, 9)) 

sns.heatmap(df_corr, square=True, vmax=1, vmin=-1, center=0)
# trainとtestの比較

def hist_train_vs_test(feature,bins,clip = False):

    plt.figure(figsize=(16, 8))

    if clip:

        th_train = np.percentile(df_train[feature], 99)

        th_test = np.percentile(df_test[feature], 99)

        plt.hist(x=[df_train[df_train[feature]<th_train][feature], df_test[df_test[feature]<th_test][feature]])

    else:

        plt.hist(x=[df_train[feature], df_test[feature]])

    plt.legend(['train', 'test'])

    plt.show()
# 特徴量ごとの可視化

f = 'MilitaryUS'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=20)

#df_test[f].hist(density=True, alpha=0.5, bins=20)

# testデータに対する可視化を記入してみましょう

plt.xlabel(f)

plt.ylabel('density')

plt.show()
# 特徴量ごとの可視化

f = 'MilitaryUS'



plt.figure(figsize=[7,7])

df_test[f].hist(density=True, alpha=0.5, bins=20)

#df_test[f].hist(density=True, alpha=0.5, bins=20)

# testデータに対する可視化を記入してみましょう

plt.xlabel(f)

plt.ylabel('density')

plt.show()
#hist_train_vs_test('ConvertedSalary',50,False)

"""

ConvertedSalary                    0

Hobby                              0

OpenSource                         0

Country                            0

Student                          362



Employment                       130

FormalEducation                  571

UndergradMajor                  4442

CompanySize                     6090

DevType                          306

YearsCoding                       18

YearsCodingProf                  644

JobSatisfaction                 2266

CareerSatisfaction               644

HopeFiveYears                    748

JobSearchStatus                    1

LastNewJob                        78

AssessJob1                      1969

AssessJob2                      1969

AssessJob3                      1969

AssessJob4                      1969

AssessJob5                      1969

AssessJob6                      1969

AssessJob7                      1969

AssessJob8                      1969

AssessJob9                      1969

AssessJob10                     1969

AssessBenefits1                 2594

AssessBenefits2                 2594

AssessBenefits3                 2594

AssessBenefits4                 2594

AssessBenefits5                 2594

AssessBenefits6                 2594

AssessBenefits7                 2594

AssessBenefits8                 2594

AssessBenefits9                 2594

AssessBenefits10                2594

AssessBenefits11                2595

JobContactPriorities1          12557

JobContactPriorities2          12557

JobContactPriorities3          12557

JobContactPriorities4          12557

JobContactPriorities5          12557

JobEmailPriorities1            13611

JobEmailPriorities2            13611

JobEmailPriorities3            13611

JobEmailPriorities4            13611

JobEmailPriorities5            13611

JobEmailPriorities6            13611

JobEmailPriorities7            13611

UpdateCV                        1651

Currency                        4527

SalaryType                      5441

CurrencySymbol                     0

CommunicationTools              4601

TimeFullyProductive             7098

TimeAfterBootcamp              38706

AgreeDisagree1                  1165

AgreeDisagree2                  1124

AgreeDisagree3                  1118

FrameworkWorkedWith            13400

OperatingSystem                 1355

NumberMonitors                  1197

CheckInCode                     2400

AdBlocker                       1222

AdBlockerDisable               12203

AdsAgreeDisagree1               1456

AdsAgreeDisagree2               1482

AdsAgreeDisagree3               1477

AdsActions                      8720

AdsPriorities1                  7139

AdsPriorities2                  7139

AdsPriorities3                  7139

AdsPriorities4                  7139

AdsPriorities5                  7139

AdsPriorities6                  7139

AdsPriorities7                  7139

AIDangerous                     6552

AIInteresting                   5096

AIResponsible                   4661

AIFuture                        2200

EthicsChoice                    1773

EthicsReport                    1865

EthicsResponsible               5044

EthicalImplications             2088

StackOverflowRecommend          1265

StackOverflowVisit              1097

StackOverflowHasAccount         1129

StackOverflowParticipate        6014

StackOverflowJobs               1305

StackOverflowDevStory           6025

StackOverflowJobsRecommend     19533

StackOverflowConsiderMember     1137

HypotheticalTools1              2361

HypotheticalTools2              2386

HypotheticalTools3              2401

HypotheticalTools4              2365

HypotheticalTools5              2371

WakeTime                        1517

HoursComputer                   1560

HoursOutside                    1610

SkipMeals                       1615

ErgonomicDevices               23115

Exercise                        1536

Gender                          2865

SexualOrientation               4692

EducationParents                3741

RaceEthnicity                   5910

Age                             2603

Dependents                      3305

MilitaryUS                     30465

SurveyTooLong                   2356

SurveyEasy                      2362

dtype: int64











# trainとtestの比較





"""
# Country カテゴリで後で一気の処理

display(df_train['Country'].value_counts())

display(df_test['Country'].value_counts())
#Student

display(df_train['Student'].value_counts())

display(df_test['Student'].value_counts())



df_train['Student'] = df_train['Student'].fillna('MISSING')

df_test['Student'] = df_test['Student'].fillna('MISSING')
#Employment

display(df_train['Employment'].value_counts())

display(df_test['Employment'].value_counts())



df_train['Employment'] = df_train['Employment'].fillna('MISSING')

df_test['Employment'] = df_test['Employment'].fillna('MISSING')
#FormalEducation                  571

display(df_train['FormalEducation'].value_counts())

display(df_test['FormalEducation'].value_counts())



df_train['FormalEducation'] = df_train['FormalEducation'].fillna('MISSING')

df_test['FormalEducation'] = df_test['FormalEducation'].fillna('MISSING')
#UndergradMajor                  571

display(df_train['UndergradMajor'].value_counts())

display(df_test['UndergradMajor'].value_counts())



df_train['UndergradMajor'] = df_train['UndergradMajor'].fillna('MISSING')

df_test['UndergradMajor'] = df_test['UndergradMajor'].fillna('MISSING')
#CompanySize                  571

display(df_train['CompanySize'].value_counts())

display(df_test['CompanySize'].value_counts())



df_train['CompanySize'] = df_train['CompanySize'].fillna('MISSING')

df_test['CompanySize'] = df_test['CompanySize'].fillna('MISSING')



dict_Company = {

    '20 to 99 employees':20,

    '100 to 499 employees':100,

    '10,000 or more employees':10000,

    '1,000 to 4,999 employees':1000,

    '10 to 19 employees':10,

    'Fewer than 10 employees':5,

    '500 to 999 employees':500,

    '5,000 to 9,999 employees':5000,

    'MISSING': -1

}

df_train['CompanySize'] = df_train['CompanySize'].apply(

    lambda x: dict_Company[x] if x in dict_Company.keys() else int(x)

    ).astype(int)

df_test['CompanySize'] = df_test['CompanySize'].apply(

    lambda x: dict_Company[x] if x in dict_Company.keys() else int(x)

    ).astype(int)
#YearsCoding

#YearsCodingProf

display(df_train['YearsCoding'].value_counts())

display(df_test['YearsCoding'].value_counts())

display(df_train['YearsCodingProf'].value_counts())

display(df_test['YearsCodingProf'].value_counts())



df_train['YearsCoding'] = df_train['YearsCoding'].fillna('MISSING')

df_test['YearsCoding'] = df_test['YearsCoding'].fillna('MISSING')

df_train['YearsCodingProf'] = df_train['YearsCodingProf'].fillna('MISSING')

df_test['YearsCodingProf'] = df_test['YearsCodingProf'].fillna('MISSING')





dict_year = {



    '0-2 years':1,

    '3-5 years':4,

    '6-8 years':7,

    '9-11 years':10,

    '12-14 years':13,

    '15-17 years':16,

    '18-20 years':19,

    '21-23 years':22,

    '24-26 years':25,

    '27-29 years':28,

    '30 or more years':31,

    'MISSING': -1

}

df_train['YearsCoding'] = df_train['YearsCoding'].apply(

    lambda x: dict_year[x] if x in dict_year.keys() else int(x)

    ).astype(int)

df_test['YearsCoding'] = df_test['YearsCoding'].apply(

    lambda x: dict_year[x] if x in dict_year.keys() else int(x)

    ).astype(int)

df_train['YearsCodingProf'] = df_train['YearsCodingProf'].apply(

    lambda x: dict_year[x] if x in dict_year.keys() else int(x)

    ).astype(int)

df_test['YearsCodingProf'] = df_test['YearsCodingProf'].apply(

    lambda x: dict_year[x] if x in dict_year.keys() else int(x)

    ).astype(int)
"""

#LastNewJob                   ＝＝＝＝精度落ちた＝＝＝＝＝

display(df_train['LastNewJob'].value_counts())

display(df_test['LastNewJob'].value_counts())



df_train['LastNewJob'] = df_train['LastNewJob'].fillna('MISSING')

df_test['LastNewJob'] = df_test['LastNewJob'].fillna('MISSING')



dict_last_job = {

    "Less than a year ago":0.5,

    "Between 1 and 2 years ago":1,

    "Between 2 and 4 years ago":3,

    "More than 4 years ago":5,

    "I've never had a job":0,

    "MISSING": -1

}

df_train['LastNewJob'] = df_train['LastNewJob'].apply(

    lambda x: dict_last_job[x] if x in dict_last_job.keys() else int(x)

    ).astype(int)

df_test['LastNewJob'] = df_test['LastNewJob'].apply(

    lambda x: dict_last_job[x] if x in dict_last_job.keys() else int(x)

    ).astype(int)





"""





display(df_train['Age'].value_counts())

display(df_test['Age'].value_counts())

df_train['Age'] = df_train['Age'].fillna('MISSING')

df_test['Age'] = df_test['Age'].fillna('MISSING')



dict_age = {

    'Under 18 years old':17,

    '18 - 24 years old':22,

    '25 - 34 years old':29,

    '35 - 44 years old':39,

    '45 - 54 years old':49,

    '55 - 64 years old':59,

    '65 years or older':65,

    "MISSING": -99

}

df_train['Age'] = df_train['Age'].apply(

    lambda x: dict_age[x] if x in dict_age.keys() else int(x)

    ).astype(int)

df_test['Age'] = df_test['Age'].apply(

    lambda x: dict_age[x] if x in dict_age.keys() else int(x)

    ).astype(int)



df_train['Age'] = df_train['Age'].replace([-99], np.nan)

df_test['Age'] = df_test['Age'].replace([-99], np.nan)
df_train_ = pd.merge(df_train, country_info, on=['Country'], how='left')

df_test_ = pd.merge(df_test, country_info, on=['Country'], how='left')
display(df_train_['Region'].value_counts())

display(df_test_['Region'].value_counts())
country_info.head()
# check histgram

f='Population'



plt.figure(figsize=[7,7])

df_train_[f].hist(density=True, alpha=0.5, bins=20)

df_test_[f].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
#Region                  571

display(df_train_['Region'].value_counts())

display(df_test_['Region'].value_counts())

"""

df_train['Region'] = df_train['Region'].fillna('MISSING')

df_test['Region'] = df_test['Region'].fillna('MISSING')



dict_region = {

    '':,

    '':,

    '':,

    '':,

    '':,

    'MISSING': -1

}

df_train['Region'] = df_train['Region'].apply(

    lambda x: dict_region[x] if x in dict_region.keys() else int(x)

    ).astype(int)

df_test['Region'] = df_test['Region'].apply(

    lambda x: dict_region[x] if x in dict_region.keys() else int(x)

    ).astype(int)







"""



target = 'ConvertedSalary'

X_temp = df_train_

y_train=df_train_[target]

ff=['CurrencySymbol','Region','DevType', 'CommunicationTools', 'FrameworkWorkedWith'] 



fff = []

for col in train.columns:

    if train[col].dtype == 'object':

        fff.append(col)

        

fff.remove('SalaryType')

fff.remove('Currency')



for f in fff:

    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([f])[target].mean()

    enc_test = df_test_[f].map(summary) 



    

    # X_trainのカテゴリ変数をoofでエンコーディングする

    # ここでは理解のために自分で交差検定的に実施するが、xfeatなどを用いても良い

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    enc_train = Series(np.zeros(len(df_train_)), index=df_train_.index)



    for i, (train_ix, val_ix) in enumerate((skf.split(df_train_, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)



    df_train_[f]  = enc_train

    df_test_[f]  = enc_test
df_train_.head()
train=df_train_

test=df_test_
"""

#試し

train=df_train_[df_train_['Region'].isin(['WESTERN EUROPE'])]

test=df_test_

"""

#数値用

train.fillna(train.median(), inplace=True)

test.fillna(test.median(), inplace=True)
#dtypeがobject(数値でない)のカラム名とユニーク数を確認

cats = []

for col in train.columns:

    if train[col].dtype == 'object':

        cats.append(col)

        

        

#文字列用

train.fillna('#', inplace=True)

test.fillna('#', inplace=True)
oe = OrdinalEncoder(cols=cats)

train = oe.fit_transform(train)

test = oe.fit_transform(test)
# 使わない列を消す

#df_train.drop(['', '', '', '',''], axis=1, inplace=True)

#df_test.drop(['', '', '', '',''], axis=1, inplace=True)

train.drop(['Country','MilitaryUS','SurveyTooLong','SurveyEasy'], axis=1, inplace=True)

test.drop(['Country','MilitaryUS','SurveyTooLong','SurveyEasy'], axis=1, inplace=True)
Y_train = np.log1p(train['ConvertedSalary']) # 評価指標がRMSLEなのでlogとっておく

X_train = train.drop('ConvertedSalary', axis=1)

X_test = test
n_sp = 5

#gkf = GroupKFold(n_splits=n_sp)





seeds = [51,61,71,81,91]

scores = []

y_pred_cvavg = np.zeros(len(X_test))



scores = []



for seed in seeds:

    kf = KFold(n_splits=n_sp, random_state=seed, shuffle=True)

    for i, (train_ix, test_ix) in enumerate(kf.split(X_train, Y_train)):

        # 学習データ

        X_train_, Y_train_ = X_train.values[train_ix], Y_train.values[train_ix]

        # 検定データ

        X_val, Y_val = X_train.values[test_ix], Y_train.values[test_ix]

    

        clf = LGBMRegressor(n_estimators=9999, random_state=seed, colsample_bytree=0.9,

                            learning_rate=0.05, subsample=0.9, num_leaves=31) 



        

        clf.fit(X_train_, Y_train_, early_stopping_rounds=50, eval_metric='rmse', eval_set=[(X_val, Y_val)])  



        y_pred = clf.predict(X_val)

        score = np.sqrt(mean_squared_error(Y_val, y_pred)) #元データをlog取っているのでRMSLE 

        scores.append(score)



        y_pred_cvavg += clf.predict(X_test)

        print(clf.predict(X_test))

            

        

print(np.mean(scores))

print(scores)



y_pred_cvavg /= n_sp * len(seeds) # fold数 * seed数



y_sub = np.expm1(y_pred_cvavg) # RMSLEなのでexpとる
scores = np.array(scores)

print('Ave. CV score is %f' % scores.mean())
# 予測結果サンプル

y_sub[:10]
submission = pd.read_csv('/kaggle/input/exam-for-students20200923/sample_submission.csv', index_col=0)



submission.ConvertedSalary = y_sub
submission.to_csv('submission.csv')

submission
#変数重要度

clf.booster_.feature_importance(importance_type='gain')

imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp
fig, ax = plt.subplots(figsize=(20, 10))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')