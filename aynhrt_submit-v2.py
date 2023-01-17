# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import scipy as sp

import numpy as np # linear algebra

from pandas import DataFrame, Series

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#データの読み込み

df_train = pd.read_csv("../input/train.csv", index_col = 0)

df_test = pd.read_csv("../input/test.csv", index_col = 0)
df_train
df_test
df_train.ConvertedSalary.mean()
#データ内容の確認

df_test.head()
#データ型の確認

cols_full = []

for col in df_train.columns:

    cols_full.append(col)

        

    print(col, df_train[col].dtype)
#欠損有無チェック

print(df_train.isnull().any())
#効かなそうな特徴量を削除

df_train = df_train.drop('SurveyTooLong', axis = 1)

df_train = df_train.drop('SurveyEasy', axis = 1)

df_test = df_test.drop('SurveyTooLong', axis = 1)

df_test = df_test.drop('SurveyEasy', axis = 1)
#説明変数と目的変数にデータ分割

X_train = df_train.drop('ConvertedSalary', axis = 1)

X_test = df_test

y_train = df_train.ConvertedSalary
#行ごとの欠損値数を特徴量として追加

X_train['nan_sum'] = X_train.isnull().sum(axis = 1)

X_test['nan_sum'] = X_test.isnull().sum(axis = 1)
#カテゴリ型特徴量のエンジニアリング：One-hot Encoding

X_train = pd.get_dummies(columns = ['Hobby'], data = X_train)

X_train = pd.get_dummies(columns = ['OpenSource'], data = X_train)

#X_train = pd.get_dummies(columns = ['Country'], data = X_train)

X_train = pd.get_dummies(columns = ['Student'], data = X_train)

X_train = pd.get_dummies(columns = ['Employment'], data = X_train)

X_train = pd.get_dummies(columns = ['FormalEducation'], data = X_train)

X_train = pd.get_dummies(columns = ['UndergradMajor'], data = X_train)

X_train = pd.get_dummies(columns = ['CompanySize'], data = X_train)

#X_train = pd.get_dummies(columns = ['DevType'], data = X_train)

X_train = pd.get_dummies(columns = ['YearsCoding'], data = X_train)

X_train = pd.get_dummies(columns = ['YearsCodingProf'], data = X_train)

#X_train = pd.get_dummies(columns = ['JobSatisfaction'], data = X_train)

#X_train = pd.get_dummies(columns = ['CareerSatisfaction'], data = X_train)

X_train = pd.get_dummies(columns = ['HopeFiveYears'], data = X_train)

X_train = pd.get_dummies(columns = ['JobSearchStatus'], data = X_train)

X_train = pd.get_dummies(columns = ['LastNewJob'], data = X_train)

X_train = pd.get_dummies(columns = ['UpdateCV'], data = X_train)

X_train = pd.get_dummies(columns = ['Currency'], data = X_train)

X_train = pd.get_dummies(columns = ['SalaryType'], data = X_train)

#X_train = pd.get_dummies(columns = ['CurrencySymbol'], data = X_train)

#X_train = pd.get_dummies(columns = ['CommunicationTools'], data = X_train)

X_train = pd.get_dummies(columns = ['TimeFullyProductive'], data = X_train)

X_train = pd.get_dummies(columns = ['TimeAfterBootcamp'], data = X_train)

X_train = pd.get_dummies(columns = ['AgreeDisagree1'], data = X_train)

X_train = pd.get_dummies(columns = ['AgreeDisagree2'], data = X_train)

X_train = pd.get_dummies(columns = ['AgreeDisagree3'], data = X_train)

X_train = pd.get_dummies(columns = ['FrameworkWorkedWith'], data = X_train)

X_train = pd.get_dummies(columns = ['OperatingSystem'], data = X_train)

X_train = pd.get_dummies(columns = ['NumberMonitors'], data = X_train)

X_train = pd.get_dummies(columns = ['CheckInCode'], data = X_train)

X_train = pd.get_dummies(columns = ['AdBlocker'], data = X_train)

X_train = pd.get_dummies(columns = ['AdBlockerDisable'], data = X_train)

X_train = pd.get_dummies(columns = ['AdsAgreeDisagree1'], data = X_train)

X_train = pd.get_dummies(columns = ['AdsAgreeDisagree2'], data = X_train)

X_train = pd.get_dummies(columns = ['AdsAgreeDisagree3'], data = X_train)

X_train = pd.get_dummies(columns = ['AdsActions'], data = X_train)

X_train = pd.get_dummies(columns = ['AIDangerous'], data = X_train)

X_train = pd.get_dummies(columns = ['AIInteresting'], data = X_train)

X_train = pd.get_dummies(columns = ['AIResponsible'], data = X_train)

X_train = pd.get_dummies(columns = ['AIFuture'], data = X_train)

X_train = pd.get_dummies(columns = ['EthicsChoice'], data = X_train)

X_train = pd.get_dummies(columns = ['EthicsReport'], data = X_train)

X_train = pd.get_dummies(columns = ['EthicsResponsible'], data = X_train)

X_train = pd.get_dummies(columns = ['EthicalImplications'], data = X_train)

#X_train = pd.get_dummies(columns = ['StackOverflowRecommend'], data = X_train)

X_train = pd.get_dummies(columns = ['StackOverflowVisit'], data = X_train)

X_train = pd.get_dummies(columns = ['StackOverflowHasAccount'], data = X_train)

X_train = pd.get_dummies(columns = ['StackOverflowParticipate'], data = X_train)

X_train = pd.get_dummies(columns = ['StackOverflowJobs'], data = X_train)

X_train = pd.get_dummies(columns = ['StackOverflowDevStory'], data = X_train)

X_train = pd.get_dummies(columns = ['StackOverflowJobsRecommend'], data = X_train)

X_train = pd.get_dummies(columns = ['StackOverflowConsiderMember'], data = X_train)

X_train = pd.get_dummies(columns = ['HypotheticalTools1'], data = X_train)

X_train = pd.get_dummies(columns = ['HypotheticalTools2'], data = X_train)

X_train = pd.get_dummies(columns = ['HypotheticalTools3'], data = X_train)

X_train = pd.get_dummies(columns = ['HypotheticalTools4'], data = X_train)

X_train = pd.get_dummies(columns = ['HypotheticalTools5'], data = X_train)

X_train = pd.get_dummies(columns = ['WakeTime'], data = X_train)

X_train = pd.get_dummies(columns = ['HoursComputer'], data = X_train)

X_train = pd.get_dummies(columns = ['HoursOutside'], data = X_train)

X_train = pd.get_dummies(columns = ['SkipMeals'], data = X_train)

X_train = pd.get_dummies(columns = ['ErgonomicDevices'], data = X_train)

X_train = pd.get_dummies(columns = ['Exercise'], data = X_train)

X_train = pd.get_dummies(columns = ['Gender'], data = X_train)

X_train = pd.get_dummies(columns = ['SexualOrientation'], data = X_train)

X_train = pd.get_dummies(columns = ['EducationParents'], data = X_train)

X_train = pd.get_dummies(columns = ['RaceEthnicity'], data = X_train)

X_train = pd.get_dummies(columns = ['Age'], data = X_train)

X_train = pd.get_dummies(columns = ['Dependents'], data = X_train)

X_train = pd.get_dummies(columns = ['MilitaryUS'], data = X_train)

#X_train = pd.get_dummies(columns = ['SurveyTooLong'], data = X_train)

#X_train = pd.get_dummies(columns = ['SurveyEasy'], data = X_train)



X_test = pd.get_dummies(columns = ['Hobby'], data = X_test)

X_test = pd.get_dummies(columns = ['OpenSource'], data = X_test)

#X_test = pd.get_dummies(columns = ['Country'], data = X_test)

X_test = pd.get_dummies(columns = ['Student'], data = X_test)

X_test = pd.get_dummies(columns = ['Employment'], data = X_test)

X_test = pd.get_dummies(columns = ['FormalEducation'], data = X_test)

X_test = pd.get_dummies(columns = ['UndergradMajor'], data = X_test)

X_test = pd.get_dummies(columns = ['CompanySize'], data = X_test)

#X_test = pd.get_dummies(columns = ['DevType'], data = X_test)

X_test = pd.get_dummies(columns = ['YearsCoding'], data = X_test)

X_test = pd.get_dummies(columns = ['YearsCodingProf'], data = X_test)

#X_test = pd.get_dummies(columns = ['JobSatisfaction'], data = X_test)

#X_test = pd.get_dummies(columns = ['CareerSatisfaction'], data = X_test)

X_test = pd.get_dummies(columns = ['HopeFiveYears'], data = X_test)

X_test = pd.get_dummies(columns = ['JobSearchStatus'], data = X_test)

X_test = pd.get_dummies(columns = ['LastNewJob'], data = X_test)

X_test = pd.get_dummies(columns = ['UpdateCV'], data = X_test)

X_test = pd.get_dummies(columns = ['Currency'], data = X_test)

X_test = pd.get_dummies(columns = ['SalaryType'], data = X_test)

#X_test = pd.get_dummies(columns = ['CurrencySymbol'], data = X_test)

#X_test = pd.get_dummies(columns = ['CommunicationTools'], data = X_test)

X_test = pd.get_dummies(columns = ['TimeFullyProductive'], data = X_test)

X_test = pd.get_dummies(columns = ['TimeAfterBootcamp'], data = X_test)

X_test = pd.get_dummies(columns = ['AgreeDisagree1'], data = X_test)

X_test = pd.get_dummies(columns = ['AgreeDisagree2'], data = X_test)

X_test = pd.get_dummies(columns = ['AgreeDisagree3'], data = X_test)

X_test = pd.get_dummies(columns = ['FrameworkWorkedWith'], data = X_test)

X_test = pd.get_dummies(columns = ['OperatingSystem'], data = X_test)

X_test = pd.get_dummies(columns = ['NumberMonitors'], data = X_test)

X_test = pd.get_dummies(columns = ['CheckInCode'], data = X_test)

X_test = pd.get_dummies(columns = ['AdBlocker'], data = X_test)

X_test = pd.get_dummies(columns = ['AdBlockerDisable'], data = X_test)

X_test = pd.get_dummies(columns = ['AdsAgreeDisagree1'], data = X_test)

X_test = pd.get_dummies(columns = ['AdsAgreeDisagree2'], data = X_test)

X_test = pd.get_dummies(columns = ['AdsAgreeDisagree3'], data = X_test)

X_test = pd.get_dummies(columns = ['AdsActions'], data = X_test)

X_test = pd.get_dummies(columns = ['AIDangerous'], data = X_test)

X_test = pd.get_dummies(columns = ['AIInteresting'], data = X_test)

X_test = pd.get_dummies(columns = ['AIResponsible'], data = X_test)

X_test = pd.get_dummies(columns = ['AIFuture'], data = X_test)

X_test = pd.get_dummies(columns = ['EthicsChoice'], data = X_test)

X_test = pd.get_dummies(columns = ['EthicsReport'], data = X_test)

X_test = pd.get_dummies(columns = ['EthicsResponsible'], data = X_test)

X_test = pd.get_dummies(columns = ['EthicalImplications'], data = X_test)

#X_test = pd.get_dummies(columns = ['StackOverflowRecommend'], data = X_test)

X_test = pd.get_dummies(columns = ['StackOverflowVisit'], data = X_test)

X_test = pd.get_dummies(columns = ['StackOverflowHasAccount'], data = X_test)

X_test = pd.get_dummies(columns = ['StackOverflowParticipate'], data = X_test)

X_test = pd.get_dummies(columns = ['StackOverflowJobs'], data = X_test)

X_test = pd.get_dummies(columns = ['StackOverflowDevStory'], data = X_test)

X_test = pd.get_dummies(columns = ['StackOverflowJobsRecommend'], data = X_test)

X_test = pd.get_dummies(columns = ['StackOverflowConsiderMember'], data = X_test)

X_test = pd.get_dummies(columns = ['HypotheticalTools1'], data = X_test)

X_test = pd.get_dummies(columns = ['HypotheticalTools2'], data = X_test)

X_test = pd.get_dummies(columns = ['HypotheticalTools3'], data = X_test)

X_test = pd.get_dummies(columns = ['HypotheticalTools4'], data = X_test)

X_test = pd.get_dummies(columns = ['HypotheticalTools5'], data = X_test)

X_test = pd.get_dummies(columns = ['WakeTime'], data = X_test)

X_test = pd.get_dummies(columns = ['HoursComputer'], data = X_test)

X_test = pd.get_dummies(columns = ['HoursOutside'], data = X_test)

X_test = pd.get_dummies(columns = ['SkipMeals'], data = X_test)

X_test = pd.get_dummies(columns = ['ErgonomicDevices'], data = X_test)

X_test = pd.get_dummies(columns = ['Exercise'], data = X_test)

X_test = pd.get_dummies(columns = ['Gender'], data = X_test)

X_test = pd.get_dummies(columns = ['SexualOrientation'], data = X_test)

X_test = pd.get_dummies(columns = ['EducationParents'], data = X_test)

X_test = pd.get_dummies(columns = ['RaceEthnicity'], data = X_test)

X_test = pd.get_dummies(columns = ['Age'], data = X_test)

X_test = pd.get_dummies(columns = ['Dependents'], data = X_test)

X_test = pd.get_dummies(columns = ['MilitaryUS'], data = X_test)

#X_test = pd.get_dummies(columns = ['SurveyTooLong'], data = X_test)

#X_test = pd.get_dummies(columns = ['SurveyEasy'], data = X_test)
#列数・列順を揃える

X_test_tmp = X_test #データ退避

X_test = pd.DataFrame(index=[])

#X_trainに存在しているが、X_testに存在しないカラムは全値0で列追加

for col in X_train.columns:

    if col in X_test_tmp.columns:

        X_test[col] = X_test_tmp[col]

    else:

        X_test[col] = int(0)



#X_testに存在しているが、X_trainに存在しないカラムは列削除        

#for col in X_test.columns:

#    if not col in X_train.columns:

#        X_test.drop([col], axis = 1, inplace = True)
#カテゴリ型特徴量のエンジニアリング：Count (Frequency) Encoding

summary = X_train.Country.value_counts()

X_train['Country'] = X_train.Country.map(summary)

summary = X_test.Country.value_counts()

X_test['Country'] = X_test.Country.map(summary)



summary = X_train.CurrencySymbol.value_counts()

X_train['CurrencySymbol'] = X_train.CurrencySymbol.map(summary)

summary = X_test.CurrencySymbol.value_counts()

X_test['CurrencySymbol'] = X_test.CurrencySymbol.map(summary)
X_train['JobSatisfaction'] = X_train['JobSatisfaction'].str.replace('Extremely satisfied', '1')

X_train['JobSatisfaction'] = X_train['JobSatisfaction'].str.replace('Moderately satisfied', '2')

X_train['JobSatisfaction'] = X_train['JobSatisfaction'].str.replace('Slightly satisfied', '3')

X_train['JobSatisfaction'] = X_train['JobSatisfaction'].str.replace('Neither satisfied nor dissatisfied', '4')

X_train['JobSatisfaction'] = X_train['JobSatisfaction'].str.replace('Slightly dissatisfied', '5')

X_train['JobSatisfaction'] = X_train['JobSatisfaction'].str.replace('Moderately dissatisfied', '6')

X_train['JobSatisfaction'] = X_train['JobSatisfaction'].str.replace('Extremely dissatisfied', '7')

X_train.JobSatisfaction = X_train.JobSatisfaction.fillna(99)



X_train['CareerSatisfaction'] = X_train['CareerSatisfaction'].str.replace('Extremely satisfied', '1')

X_train['CareerSatisfaction'] = X_train['CareerSatisfaction'].str.replace('Moderately satisfied', '2')

X_train['CareerSatisfaction'] = X_train['CareerSatisfaction'].str.replace('Slightly satisfied', '3')

X_train['CareerSatisfaction'] = X_train['CareerSatisfaction'].str.replace('Neither satisfied nor dissatisfied', '4')

X_train['CareerSatisfaction'] = X_train['CareerSatisfaction'].str.replace('Slightly dissatisfied', '5')

X_train['CareerSatisfaction'] = X_train['CareerSatisfaction'].str.replace('Moderately dissatisfied', '6')

X_train['CareerSatisfaction'] = X_train['CareerSatisfaction'].str.replace('Extremely dissatisfied', '7')

X_train.CareerSatisfaction = X_train.CareerSatisfaction.fillna(99)



X_test['JobSatisfaction'] = X_test['JobSatisfaction'].str.replace('Extremely satisfied', '1')

X_test['JobSatisfaction'] = X_test['JobSatisfaction'].str.replace('Moderately satisfied', '2')

X_test['JobSatisfaction'] = X_test['JobSatisfaction'].str.replace('Slightly satisfied', '3')

X_test['JobSatisfaction'] = X_test['JobSatisfaction'].str.replace('Neither satisfied nor dissatisfied', '4')

X_test['JobSatisfaction'] = X_test['JobSatisfaction'].str.replace('Slightly dissatisfied', '5')

X_test['JobSatisfaction'] = X_test['JobSatisfaction'].str.replace('Moderately dissatisfied', '6')

X_test['JobSatisfaction'] = X_test['JobSatisfaction'].str.replace('Extremely dissatisfied', '7')

X_test.JobSatisfaction = X_test.JobSatisfaction.fillna(99)



X_test['CareerSatisfaction'] = X_test['CareerSatisfaction'].str.replace('Extremely satisfied', '1')

X_test['CareerSatisfaction'] = X_test['CareerSatisfaction'].str.replace('Moderately satisfied', '2')

X_test['CareerSatisfaction'] = X_test['CareerSatisfaction'].str.replace('Slightly satisfied', '3')

X_test['CareerSatisfaction'] = X_test['CareerSatisfaction'].str.replace('Neither satisfied nor dissatisfied', '4')

X_test['CareerSatisfaction'] = X_test['CareerSatisfaction'].str.replace('Slightly dissatisfied', '5')

X_test['CareerSatisfaction'] = X_test['CareerSatisfaction'].str.replace('Moderately dissatisfied', '6')

X_test['CareerSatisfaction'] = X_test['CareerSatisfaction'].str.replace('Extremely dissatisfied', '7')

X_test.CareerSatisfaction = X_test.CareerSatisfaction.fillna(99)
#テキスト型特徴量のエンジニアリング：TFIDF

#テキスト列のみ抜き出す

TXT1_train = X_train['DevType']

TXT2_train = X_train['CommunicationTools']

TXT1_test = X_test['DevType']

TXT2_test = X_test['CommunicationTools']
#欠損値補間

TXT1_train.fillna('#', inplace = True)

TXT2_train.fillna('#', inplace = True)

TXT1_test.fillna('#', inplace = True)

TXT2_test.fillna('#', inplace = True)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features = 100)

TXT1_train = tfidf.fit_transform(TXT1_train)

TXT1_test = tfidf.transform(TXT1_test)

TXT2_train = tfidf.fit_transform(TXT2_train)

TXT2_test = tfidf.transform(TXT2_test)
X_train_idx = X_train.reset_index()

TXT1_train2 = pd.DataFrame(TXT1_train.todense())

X_train = pd.concat([X_train_idx, TXT1_train2], axis = 1)

X_test_idx = X_test.reset_index()

TXT1_test2 = pd.DataFrame(TXT1_test.todense())

X_test = pd.concat([X_test_idx, TXT1_test2], axis = 1)



X_train_idx = X_train.reset_index()

TXT2_train2 = pd.DataFrame(TXT2_train.todense())

X_train = pd.concat([X_train_idx, TXT2_train2], axis = 1)

X_test_idx = X_test.reset_index()

TXT2_test2 = pd.DataFrame(TXT2_test.todense())

X_test = pd.concat([X_test_idx, TXT2_test2], axis = 1)
#テキスト列を削除する

X_train.drop(['DevType' ,'CommunicationTools'], axis = 1, inplace = True)

X_test.drop(['DevType' ,'CommunicationTools'], axis = 1, inplace = True)
#数値と文字列が混在しているデータから、数値のみ抜粋

X_train['StackOverflowRecommend'] = X_train['StackOverflowRecommend'].str.extract('([0-9]+)').astype(float)

X_test['StackOverflowRecommend'] = X_test['StackOverflowRecommend'].str.extract('([0-9]+)').astype(float)
X_train.head(10)
#欠損を平均値で補間

X_train.AssessJob1 = X_train.AssessJob1.fillna(X_train.AssessJob1.mean())

X_train.AssessJob2 = X_train.AssessJob2.fillna(X_train.AssessJob2.mean())

X_train.AssessJob3 = X_train.AssessJob3.fillna(X_train.AssessJob3.mean())

X_train.AssessJob4 = X_train.AssessJob4.fillna(X_train.AssessJob4.mean())

X_train.AssessJob5 = X_train.AssessJob5.fillna(X_train.AssessJob5.mean())

X_train.AssessJob6 = X_train.AssessJob6.fillna(X_train.AssessJob6.mean())

X_train.AssessJob7 = X_train.AssessJob7.fillna(X_train.AssessJob7.mean())

X_train.AssessJob8 = X_train.AssessJob8.fillna(X_train.AssessJob8.mean())

X_train.AssessJob9 = X_train.AssessJob9.fillna(X_train.AssessJob9.mean())

X_train.AssessJob10 = X_train.AssessJob10.fillna(X_train.AssessJob10.mean())

X_train.AssessBenefits1 = X_train.AssessBenefits1.fillna(X_train.AssessBenefits1.mean())

X_train.AssessBenefits2 = X_train.AssessBenefits2.fillna(X_train.AssessBenefits2.mean())

X_train.AssessBenefits3 = X_train.AssessBenefits3.fillna(X_train.AssessBenefits3.mean())

X_train.AssessBenefits4 = X_train.AssessBenefits4.fillna(X_train.AssessBenefits4.mean())

X_train.AssessBenefits5 = X_train.AssessBenefits5.fillna(X_train.AssessBenefits5.mean())

X_train.AssessBenefits6 = X_train.AssessBenefits6.fillna(X_train.AssessBenefits6.mean())

X_train.AssessBenefits7 = X_train.AssessBenefits7.fillna(X_train.AssessBenefits7.mean())

X_train.AssessBenefits8 = X_train.AssessBenefits8.fillna(X_train.AssessBenefits8.mean())

X_train.AssessBenefits9 = X_train.AssessBenefits9.fillna(X_train.AssessBenefits9.mean())

X_train.AssessBenefits10 = X_train.AssessBenefits10.fillna(X_train.AssessBenefits10.mean())

X_train.AssessBenefits11 = X_train.AssessBenefits11.fillna(X_train.AssessBenefits11.mean())

X_train.JobContactPriorities1 = X_train.JobContactPriorities1.fillna(X_train.JobContactPriorities1.mean())

X_train.JobContactPriorities2 = X_train.JobContactPriorities2.fillna(X_train.JobContactPriorities2.mean())

X_train.JobContactPriorities3 = X_train.JobContactPriorities3.fillna(X_train.JobContactPriorities3.mean())

X_train.JobContactPriorities4 = X_train.JobContactPriorities4.fillna(X_train.JobContactPriorities4.mean())

X_train.JobContactPriorities5 = X_train.JobContactPriorities5.fillna(X_train.JobContactPriorities5.mean())

X_train.JobEmailPriorities1 = X_train.JobEmailPriorities1.fillna(X_train.JobEmailPriorities1.mean())

X_train.JobEmailPriorities2 = X_train.JobEmailPriorities2.fillna(X_train.JobEmailPriorities2.mean())

X_train.JobEmailPriorities3 = X_train.JobEmailPriorities3.fillna(X_train.JobEmailPriorities3.mean())

X_train.JobEmailPriorities4 = X_train.JobEmailPriorities4.fillna(X_train.JobEmailPriorities4.mean())

X_train.JobEmailPriorities5 = X_train.JobEmailPriorities5.fillna(X_train.JobEmailPriorities5.mean())

X_train.JobEmailPriorities6 = X_train.JobEmailPriorities6.fillna(X_train.JobEmailPriorities6.mean())

X_train.JobEmailPriorities7 = X_train.JobEmailPriorities7.fillna(X_train.JobEmailPriorities7.mean())

X_train.AdsPriorities1 = X_train.AdsPriorities1.fillna(X_train.AdsPriorities1.mean())

X_train.AdsPriorities2 = X_train.AdsPriorities2.fillna(X_train.AdsPriorities2.mean())

X_train.AdsPriorities3 = X_train.AdsPriorities3.fillna(X_train.AdsPriorities3.mean())

X_train.AdsPriorities4 = X_train.AdsPriorities4.fillna(X_train.AdsPriorities4.mean())

X_train.AdsPriorities5 = X_train.AdsPriorities5.fillna(X_train.AdsPriorities5.mean())

X_train.AdsPriorities6 = X_train.AdsPriorities6.fillna(X_train.AdsPriorities6.mean())

X_train.AdsPriorities7 = X_train.AdsPriorities7.fillna(X_train.AdsPriorities7.mean())

X_train.StackOverflowRecommend = X_train.StackOverflowRecommend.fillna(99)



X_test.AssessJob1 = X_test.AssessJob1.fillna(X_test.AssessJob1.mean())

X_test.AssessJob2 = X_test.AssessJob2.fillna(X_test.AssessJob2.mean())

X_test.AssessJob3 = X_test.AssessJob3.fillna(X_test.AssessJob3.mean())

X_test.AssessJob4 = X_test.AssessJob4.fillna(X_test.AssessJob4.mean())

X_test.AssessJob5 = X_test.AssessJob5.fillna(X_test.AssessJob5.mean())

X_test.AssessJob6 = X_test.AssessJob6.fillna(X_test.AssessJob6.mean())

X_test.AssessJob7 = X_test.AssessJob7.fillna(X_test.AssessJob7.mean())

X_test.AssessJob8 = X_test.AssessJob8.fillna(X_test.AssessJob8.mean())

X_test.AssessJob9 = X_test.AssessJob9.fillna(X_test.AssessJob9.mean())

X_test.AssessJob10 = X_test.AssessJob10.fillna(X_test.AssessJob10.mean())

X_test.AssessBenefits1 = X_test.AssessBenefits1.fillna(X_test.AssessBenefits1.mean())

X_test.AssessBenefits2 = X_test.AssessBenefits2.fillna(X_test.AssessBenefits2.mean())

X_test.AssessBenefits3 = X_test.AssessBenefits3.fillna(X_test.AssessBenefits3.mean())

X_test.AssessBenefits4 = X_test.AssessBenefits4.fillna(X_test.AssessBenefits4.mean())

X_test.AssessBenefits5 = X_test.AssessBenefits5.fillna(X_test.AssessBenefits5.mean())

X_test.AssessBenefits6 = X_test.AssessBenefits6.fillna(X_test.AssessBenefits6.mean())

X_test.AssessBenefits7 = X_test.AssessBenefits7.fillna(X_test.AssessBenefits7.mean())

X_test.AssessBenefits8 = X_test.AssessBenefits8.fillna(X_test.AssessBenefits8.mean())

X_test.AssessBenefits9 = X_test.AssessBenefits9.fillna(X_test.AssessBenefits9.mean())

X_test.AssessBenefits10 = X_test.AssessBenefits10.fillna(X_test.AssessBenefits10.mean())

X_test.AssessBenefits11 = X_test.AssessBenefits11.fillna(X_test.AssessBenefits11.mean())

X_test.JobContactPriorities1 = X_test.JobContactPriorities1.fillna(X_test.JobContactPriorities1.mean())

X_test.JobContactPriorities2 = X_test.JobContactPriorities2.fillna(X_test.JobContactPriorities2.mean())

X_test.JobContactPriorities3 = X_test.JobContactPriorities3.fillna(X_test.JobContactPriorities3.mean())

X_test.JobContactPriorities4 = X_test.JobContactPriorities4.fillna(X_test.JobContactPriorities4.mean())

X_test.JobContactPriorities5 = X_test.JobContactPriorities5.fillna(X_test.JobContactPriorities5.mean())

X_test.JobEmailPriorities1 = X_test.JobEmailPriorities1.fillna(X_test.JobEmailPriorities1.mean())

X_test.JobEmailPriorities2 = X_test.JobEmailPriorities2.fillna(X_test.JobEmailPriorities2.mean())

X_test.JobEmailPriorities3 = X_test.JobEmailPriorities3.fillna(X_test.JobEmailPriorities3.mean())

X_test.JobEmailPriorities4 = X_test.JobEmailPriorities4.fillna(X_test.JobEmailPriorities4.mean())

X_test.JobEmailPriorities5 = X_test.JobEmailPriorities5.fillna(X_test.JobEmailPriorities5.mean())

X_test.JobEmailPriorities6 = X_test.JobEmailPriorities6.fillna(X_test.JobEmailPriorities6.mean())

X_test.JobEmailPriorities7 = X_test.JobEmailPriorities7.fillna(X_test.JobEmailPriorities7.mean())

X_test.AdsPriorities1 = X_test.AdsPriorities1.fillna(X_test.AdsPriorities1.mean())

X_test.AdsPriorities2 = X_test.AdsPriorities2.fillna(X_test.AdsPriorities2.mean())

X_test.AdsPriorities3 = X_test.AdsPriorities3.fillna(X_test.AdsPriorities3.mean())

X_test.AdsPriorities4 = X_test.AdsPriorities4.fillna(X_test.AdsPriorities4.mean())

X_test.AdsPriorities5 = X_test.AdsPriorities5.fillna(X_test.AdsPriorities5.mean())

X_test.AdsPriorities6 = X_test.AdsPriorities6.fillna(X_test.AdsPriorities6.mean())

X_test.AdsPriorities7 = X_test.AdsPriorities7.fillna(X_test.AdsPriorities7.mean())

X_test.StackOverflowRecommend = X_test.StackOverflowRecommend.fillna(99)
#X_train = X_train.drop('index', axis = 1)

X_train = X_train.drop('Respondent', axis = 1)

#X_test = X_test.drop('index', axis = 1)

X_test = X_test.drop('Respondent', axis = 1)
#カラム名取得

num_cols = []

for col in X_train.columns:

    num_cols.append(col)

    

print(num_cols)
#数値型特徴量のエンジニアリング：StandardScaler

from sklearn.preprocessing import StandardScaler

#num_cols = ['loan_amnt', 'installment', 'emp_length', 'annual_inc', 'zip_code', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'date_diff']

#num_cols = ['loan_amnt', 'installment', 'emp_length', 'annual_inc', 'zip_code', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'tot_coll_amt', 'tot_cur_bal', 'date_diff', 'addr_state']

scaler = StandardScaler()

scaler.fit(X_train[num_cols])
#変換後のデータで各列を置換

X_train[num_cols] = scaler.transform(X_train[num_cols])

X_test[num_cols] = scaler.transform(X_test[num_cols])
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

#from lightgbm import LGBMClassifier

from sklearn.linear_model import LinearRegression
"""

%%time

# CVしてスコアを見てみる

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    #clf = GradientBoostingClassifier() # ここではデフォルトのパラメータになっている。各自の検討項目です

    #clf = LGBMClassifier()

    lr = LinearRegression()

    

    lr.fit(X_train_, y_train_)

    #y_pred = lr.predict_proba(X_val)[:,1]

    y_pred = lr.predict(X_val)

    #score = roc_auc_score(y_val, y_pred)

    #score = np.sqrt(np.mean(((np.log(y+1)-np.log(y_pred+1))**2)))

    #scores.append(score)

    

    #print('CV Score of Fold_%d is %f' % (i, score))

"""
#全データで再学習し、testに対して予測する

#clf = GradientBoostingClassifier() # ここではデフォルトのパラメータになっている。各自の検討項目です

#clf = LGBMClassifier()

lr = LinearRegression()

#clf.fit(X_train, y_train)

lr.fit(X_train, y_train)



#y_pred = lr.predict_proba(X_test)[:,1] # predict_probaで確率を出力する

y_pred = lr.predict(X_test)
y_pred
#sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.ConvertedSalary = y_pred

submission.to_csv('submission.csv')