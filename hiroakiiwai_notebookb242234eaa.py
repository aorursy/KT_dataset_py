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
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series

import re



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from sklearn.preprocessing import StandardScaler



import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
df_train = pd.read_csv('/kaggle/input/exam-for-students20200923/train.csv')

df_test = pd.read_csv('/kaggle/input/exam-for-students20200923/test.csv')

df_country = pd.read_csv('/kaggle/input/exam-for-students20200923/country_info.csv')
df_train.head()
df_test.head()
df_country.head()
df_train.describe()
df_test.describe()
f = 'ConvertedSalary'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
# 対数変換(評価指標がRMSLEのため、目的変数に対して対数変換を実施)

log_trans_cols = ['ConvertedSalary']



for col in log_trans_cols:

    df_train[col] = df_train[col].apply(np.log1p)
f = 'ConvertedSalary'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
# df_country を使えるデータのみに限定する

df_country = df_country[['Country', 'Region', 'GDP ($ per capita)']]



# df_country と結合

df_train = pd.merge(df_train, df_country, how='left', left_on='Country', right_on='Country').set_index(df_train.index)

df_test = pd.merge(df_test, df_country, how='left', left_on='Country', right_on='Country').set_index(df_test.index)
df_train.head()
df_test.head()
f = 'GDP ($ per capita)'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=20)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
#各列の構成要素の確認

for i in df_train.columns:

    print(df_train[i].value_counts())

    print('\n')
CompanySize_mapping = {"Fewer than 10 employees": 9,

                       "10 to 19 employees": 10,

                       "20 to 99 employees": 20,

                       "100 to 499 employees": 100,

                       "500 to 999 employees": 500,

                       "1,000 to 4,999 employees": 1000,

                       "5,000 to 9,999 employees": 5000,

                       "10,000 or more employees": 10000

              }



YearsCoding_mapping = {"0-2 years": 1,

                        "3-5 years": 3,

                        "6-8 years": 6,

                        "9-11 years": 9,

                        "12-14 years": 12,

                        "15-17 years": 15,

                        "18-20 years": 18,

                        "21-23 years": 21,

                        "24-26 years": 24,

                        "27-29 years": 27,

                        "30 or more years": 30

              }



YearsCodingProf_mapping = {"15-17 years": 15,

                           "18-20 years": 18,

                           "21-23 years": 21,

                           "24-26 years": 24,

                           "27-29 years": 27,

                           "30 or more years": 30

                          }



JobSatisfaction_mapping = {"Extremely satisfied": 3,

                      "Moderately satisfied": 2,

                      "Slightly satisfied": 1,

                      "Neither satisfied nor dissatisfied": 0,

                      "Slightly dissatisfied": -1,

                      "Moderately dissatisfied":-2,

                      "Extremely dissatisfied": -3

              }

CareerSatisfaction_mapping = JobSatisfaction_mapping



TimeFullyProductive_mapping = {"Less than a month": 0.5,

                        "One to three months": 1,

                        "Three to six months": 3,

                        "Six to nine months": 6,

                        "Nine months to a year": 9,

                        "More than a year": 13

              }



LastNewJob_mapping = {"Less than a year ago": 0.5,

                      "Between 1 and 2 years ago": 1,

                      "Between 2 and 4 years ago": 2,

                      "More than 4 years ago": 4,

                      "I've never had a job": 9999

              }



Age_mapping = {"Under 18 years old": 17,

               "18 - 24 years old": 18,

               "25 - 34 years old": 25,

               "35 - 44 years old": 35,

               "45 - 54 years old": 45,

               "55 - 64 years old": 55,

               "65 years or older": 65

              }



StackOverflowRecommend_mapping = {"10 (Very Likely)": 10,

               "9": 9,

               "8": 8,

               "7": 7,

               "6": 6,

               "5": 5,

               "4": 4,

               "3": 3,

               "2": 2,

               "1": 1,

               "0 (Not Likely)": 0

              }

StackOverflowJobsRecommend_mapping = StackOverflowRecommend_mapping



HoursOutside_mapping = {"Less than 30 minutes": 0.1,

                        "30 - 59 minutes": 0.5,

                        "1 - 2 hours": 1,

                        "3 - 4 hours": 3,

                        "Over 4 hours": 5

              }



HoursComputer_mapping = {"Less than 1 hour": 0.5,

                        "1 - 4 hours": 1,

                        "5 - 8 hours": 5,

                        "9 - 12 hours": 9,

                        "Over 12 hours": 15

              }



SkipMeals_mapping = {"Never": 0,

                      "1 - 2 times per week": 1,

                      "3 - 4 times per week": 3,

                      "Daily or almost every day": 7

              }

Exercise_mapping = {"I don't typically exercise": 0,

                      "1 - 2 times per week": 1,

                      "3 - 4 times per week": 3,

                      "Daily or almost every day": 7

              }



AgreeDisagree1_mapping = {"Strongly agree": 2,

                      "Agree": 1,

                      "Neither Agree nor Disagree": 0,

                      "Disagree":-1,

                      "Strongly disagree": -2

              }

AgreeDisagree2_mapping = AgreeDisagree1_mapping

AgreeDisagree3_mapping = AgreeDisagree1_mapping



AdsAgreeDisagree1_mapping = {"Strongly agree": 2,

                      "Somewhat agree": 1,

                      "Neither agree nor disagree": 0,

                      "Somewhat disagree": -1,

                      "Strongly disagree": -2

              }

AdsAgreeDisagree2_mapping = AdsAgreeDisagree1_mapping

AdsAgreeDisagree3_mapping = AdsAgreeDisagree1_mapping



HypotheticalTools1_mapping = {"Extremely interested": 4,

                      "Very interested": 3,

                      "Somewhat interested": 2,

                      "A little bit interested": 1,

                      "Not at all interested": 0

                             }

HypotheticalTools2_mapping = HypotheticalTools1_mapping

HypotheticalTools3_mapping = HypotheticalTools1_mapping

HypotheticalTools4_mapping = HypotheticalTools1_mapping

HypotheticalTools5_mapping = HypotheticalTools1_mapping



NumberMonitors_mapping = {"1": 1,

                      "2": 2,

                      "3": 3,

                      "4": 4,

                      "More than 4": 5

                             }



CheckInCode_mapping = {"Never": 0,

                      "Less than once per month": 1,

                      "Weekly or a few times per month": 4,

                      "A few times per week": 8,

                      "Once a day": 20,

                      "Multiple times per day": 40

                             }



StackOverflowVisit_mapping = {"I have never visited Stack Overflow (before today)": 0,

                      "Less than once per month or monthly ": 1,

                      "A few times per month or weekly": 4,

                      "A few times per week": 8,

                      "Daily or almost daily": 20,

                      "Multiple times per day": 40

                             }

StackOverflowParticipate_mapping = {"I have never participated in Q&A on Stack Overflow": 0,

                      "Less than once per month or monthly": 1,

                      "A few times per month or weekly": 4,

                      "A few times per week": 8,

                      "Daily or almost daily": 20,

                      "Multiple times per day": 40

                             }



WakeTime_mapping = {"Before 5:00 AM": 4,

                      "Between 5:00 - 6:00 AM": 5,

                      "Between 6:01 - 7:00 AM": 6,

                      "Between 7:01 - 8:00 AM": 7,

                      "Between 8:01 - 9:00 AM": 8,

                      "Between 9:01 - 10:00 AM": 9,

                      "Between 10:01 - 11:00 AM": 10,

                      "Between 11:01 AM - 12:00 PM": 11,

                      "After 12:01 PM": 12,

                      "I work night shifts": 18,

                      "I do not have a set schedule": 9999

                  }



def mapping(map_col, mapping):

    df_train[map_col] = df_train[map_col].map(mapping)

    df_test[map_col] = df_test[map_col].map(mapping)

    

mapping('CompanySize', CompanySize_mapping)

mapping('YearsCoding', YearsCoding_mapping)

mapping('YearsCodingProf', YearsCodingProf_mapping)

mapping('JobSatisfaction', JobSatisfaction_mapping)

mapping('CareerSatisfaction', CareerSatisfaction_mapping)

mapping('TimeFullyProductive', TimeFullyProductive_mapping)

mapping('LastNewJob', LastNewJob_mapping)

mapping('Age', Age_mapping)

mapping('StackOverflowRecommend', StackOverflowRecommend_mapping)

mapping('StackOverflowJobsRecommend', StackOverflowJobsRecommend_mapping)

mapping('HoursOutside', HoursOutside_mapping)

mapping('HoursComputer', HoursComputer_mapping)

mapping('SkipMeals', SkipMeals_mapping)

mapping('Exercise', Exercise_mapping)

mapping('AgreeDisagree1', AgreeDisagree1_mapping)

mapping('AgreeDisagree2', AgreeDisagree2_mapping)

mapping('AgreeDisagree3', AgreeDisagree3_mapping)

mapping('AdsAgreeDisagree1', AdsAgreeDisagree1_mapping)

mapping('AdsAgreeDisagree2', AdsAgreeDisagree2_mapping)

mapping('AdsAgreeDisagree3', AdsAgreeDisagree3_mapping)

mapping('HypotheticalTools1', HypotheticalTools1_mapping)

mapping('HypotheticalTools2', HypotheticalTools2_mapping)

mapping('HypotheticalTools3', HypotheticalTools3_mapping)

mapping('HypotheticalTools4', HypotheticalTools4_mapping)

mapping('HypotheticalTools5', HypotheticalTools5_mapping)

mapping('NumberMonitors', NumberMonitors_mapping)

mapping('CheckInCode', CheckInCode_mapping)

mapping('StackOverflowVisit', StackOverflowVisit_mapping)

mapping('StackOverflowParticipate', StackOverflowParticipate_mapping)

mapping('WakeTime', WakeTime_mapping)
# 特徴量の削除

def drop_col(col):

    df_train.drop([col], axis=1, inplace=True)

    df_test.drop([col], axis=1, inplace=True)

    

drop_col('Hobby')

drop_col('SurveyTooLong')

drop_col('SurveyEasy')
#各列の構成要素の確認

for i in df_train.columns:

    print(df_train[i].value_counts())

    print('\n')
# x と y の分離

X_train = df_train.drop(['Respondent'], axis=1)

X_train = X_train.drop(['ConvertedSalary'], axis=1)

y_train = df_train.ConvertedSalary



X_test = df_test.drop(['Respondent'], axis=1)
X_train.shape, X_test.shape
cats = []

nums = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

    else:

        nums.append(col)
cats
# Ordinal Encoder

oe_cols = cats

encoder = OrdinalEncoder(cols=oe_cols)

X_train[oe_cols] = encoder.fit_transform(X_train[oe_cols])

X_test[oe_cols] = encoder.transform(X_test[oe_cols])
X_train.isnull().sum()
X_test.isnull().sum()
# Target Encoding

target = 'ConvertedSalary'

X_temp = pd.concat([X_train, y_train], axis=1)



te_cols = ['Age','YearsCoding', 'YearsCodingProf','JobSatisfaction', 'CareerSatisfaction',

           'CurrencySymbol', 'TimeFullyProductive','HoursComputer', 'HoursOutside', 

           'SkipMeals', 'Employment', 'DevType', 'LastNewJob', 'NumberMonitors']



for col in te_cols:



    # X_testはX_trainでエンコーディングする

    summary = X_temp.groupby([col])[target].mean()

    X_test['te_' + col] = X_test[col].map(summary) 





    # X_trainのカテゴリ変数をoofでエンコーディングする

    kf = KFold(n_splits=5, random_state=71, shuffle=False)

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



    for i, (train_ix, val_ix) in enumerate((kf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

    X_train['te_' + col]  = enc_train
X_train.fillna(-9999, inplace = True)

X_test.fillna(-9999, inplace = True)
fig=plt.figure(figsize=[20,100])



i = 0

for col in X_train.columns:

    i=i+1



    plt.subplots_adjust(wspace=0.2, hspace=0.8)

    ax_name = fig.add_subplot(120,2,i)

    ax_name.hist(X_train[col],bins=30,density=True, alpha=0.5,color = 'r')

    ax_name.hist(X_test[col],bins=30,density=True, alpha=0.5, color = 'b')

    ax_name.set_title(col)
scores = []

scores2 = []

y_pred_avg = np.zeros(len(X_test))



iter_num = 5 # seed averagingで繰り返す回数を設定

num_split = 10 # cvの分割数



for random_state in range(0, iter_num):

    

    kf = KFold(n_splits=num_split, random_state=random_state, shuffle=True)

    for i, (train_ix, test_ix) in enumerate(kf.split(X_train, y_train)):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        lgbmr = LGBMRegressor(

                             objective='regression',

                             n_estimators=10000, 

                             boosting_type='gbdt', 

                             subsample = 0.8,

                             reg_alpha = 0.85, 

                             reg_lambda = 0.85,

                             random_state=random_state, 

                             importance_type='gain',

                             silent = 1,

                             n_jobs=-1)



        lgbmr.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='rmse', verbose=100, eval_set=[(X_val, y_val)])



        #y_pred = lgbmr.predict(X_val)

        score2 = np.sqrt(mean_squared_error(y_train_, lgbmr.predict(X_train_)))

        scores2.append(score2)

        score = np.sqrt(mean_squared_error(y_val, lgbmr.predict(X_val)))

        scores.append(score)



        print('===================================================')

        print('train CV Score of Fold_%d is %f' % (i, score2))

        print('---------------------------------------------------')

        print('val CV Score of Fold_%d is %f' % (i, score))

        print('---------------------------------------------------')

        print('diff', score - score2)

        print('===================================================')

        

        # 予測確率の平均値を求める

        y_pred_avg += lgbmr.predict(X_test)



y_pred_avg = y_pred_avg / (num_split * iter_num )

y_pred1 = np.expm1(y_pred_avg)



print('Average')

print('val avg:', np.mean(scores))

print(scores)

print('=============================')

print('train avg:', np.mean(scores2))

print(scores2)

print('=============================')

print('diff',np.mean(scores) - np.mean(scores2))
# Plot feature importance

importances = lgbmr.feature_importances_



indices = np.argsort(importances)[::-1]

feat_labels = X_train.columns[0:]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" % (f +1,30, feat_labels[indices[f]], importances[indices[f]]))
fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(lgbmr, max_num_features=30, ax=ax, importance_type='gain')
submission = pd.read_csv('/kaggle/input/exam-for-students20200923/sample_submission.csv')

submission.ConvertedSalary = y_pred1

submission.to_csv('./submission.csv', index=False)
submission