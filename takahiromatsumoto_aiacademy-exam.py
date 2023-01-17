# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns



from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn import preprocessing as pp

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, PowerTransformer, quantile_transform



import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor



import eli5

from eli5.sklearn import PermutationImportance



from sklearn.svm import SVC

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor



import itertools

import math

import re
df_train_original = pd.read_csv('../input/exam-for-students20200129/train.csv', index_col=0) 

df_test_original = pd.read_csv('../input/exam-for-students20200129/test.csv',index_col=0)



target = 'ConvertedSalary'



df_country_original = pd.read_csv('../input/exam-for-students20200129/country_info.csv')



df_train_original.shape, df_test_original.shape
df_train = df_train_original.copy()

df_test = df_test_original.copy()

df_country = df_country_original.copy()



df_test_with_targetcol = df_test_original.copy()

df_test_with_targetcol[target] = -1

df_all = pd.concat([df_train, df_test_with_targetcol], axis=0)
df_train
df_test
df_country
# df_train = df_train.merge(df_country, on = ['Country'], how = 'left')

# df_test = df_test.merge(df_country, on = ['Country'], how = 'left')

# df_all = df_all.merge(df_country, on = ['Country'], how = 'left')



#GDP以外実質不要

df_GDP = df_country[['Country', 'GDP ($ per capita)']]



df_train = df_train.merge(df_GDP, on = ['Country'], how = 'left')

df_test = df_test.merge(df_GDP, on = ['Country'], how = 'left')

df_all = df_all.merge(df_GDP, on = ['Country'], how = 'left')
df_train.describe()
df_test.describe()
df_train_corr = df_train.corr()

print(df_train_corr)

sns.heatmap(df_train_corr, vmax=1, vmin=-1, center=0, cmap = 'seismic')
# f = '入力する'

f = 'GDP ($ per capita)'

dens = False



plt.figure(figsize=[15,7])

df_train[f].hist(density=dens, alpha=0.5, bins=30, color = 'r', label = 'train')

df_test[f].hist(density=dens, alpha=0.5, bins=30, color = 'b', label = 'test')

plt.xlabel(f)

plt.ylabel('density')

plt.show()
c = 'Country'



# df_train[c].value_counts() / len(df_train)

df_train[c].value_counts()
df_test[c].value_counts()

#GermanyとFranceは、training dataにない。

#Countryカラムは消す
df_train[df_train['Country'] == 'Germany']

df_train[df_train['Country'] == 'France']
for col in df_all.columns:

    print(df_all[col].value_counts())

    print('____')

    

#DevType, CommunicationTools, FrameworkWorkedWithはテキスト



#マッピング

# Age, CompanySize, Exercise, HoursComputer, HoursOutside, LastNewJob, SkipMeals, StackOverflowRecommend, YearsCoding, YearsCodingProf, 
# col_outlier = target

# #max, minを設定

# maximum = 2000000

# minimum = 0

# percentile = 0.99



# #外れ値を除外

# df_train = df_train[df_train[col_outlier] < df_train[col_outlier].quantile(percentile)]

# df_train = df_train.loc[df_train[col_outlier] <= maximum]

# df_train = df_train.loc[df_train[col_outlier] >= minimum]



# #古いデータを除外

# # df_train = df_train[df_train.issue_d.dt.year >= oldest]#issue_dをちゃんと変換。既にyearがあるなら、そっち使う。
#先にターゲットをLog変換するか否か

first_log = True
y_train = df_train[target]

X_train = df_train.drop([target], axis=1)



X_test = df_test



X_all = pd.concat([X_train, X_test], axis = 0)



#ターゲットが欠損している場合、除去

X_train = X_train[y_train.isnull()==False]

y_train = y_train[y_train.isnull()==False]



#ターゲットのlog1変換

if(first_log == True):

    y_train = y_train.apply(np.log1p)
#DevType, CommunicationTools, FrameworkWorkedWithはテキスト

TXT_train_DevType = X_train.DevType.copy()

TXT_test_DevType = X_test.DevType.copy()



TXT_train_CommunicationTools = X_train.CommunicationTools.copy()

TXT_test_CommunicationTools = X_test.CommunicationTools.copy()



TXT_train_FrameworkWorkedWith = X_train.FrameworkWorkedWith.copy()

TXT_test_FrameworkWorkedWith = X_test.FrameworkWorkedWith.copy()



TXT_train_DevType.fillna('#', inplace = True)

TXT_test_DevType.fillna('#', inplace = True)



TXT_train_CommunicationTools.fillna('#', inplace = True)

TXT_test_CommunicationTools.fillna('#', inplace = True)



TXT_train_FrameworkWorkedWith.fillna('#', inplace = True)

TXT_test_FrameworkWorkedWith.fillna('#', inplace = True)
# TXT_train_DevType
X_train['missing_sum'] = X_train.isnull().sum(axis=1)

X_test['missing_sum'] = X_test.isnull().sum(axis=1)
X_train['DevType'].fillna('#', inplace = True)

X_test['DevType'].fillna('#', inplace = True)



X_train['CommunicationTools'].fillna('#', inplace = True)

X_test['CommunicationTools'].fillna('#', inplace = True)



X_train['FrameworkWorkedWith'].fillna('#', inplace = True)

X_test['FrameworkWorkedWith'].fillna('#', inplace = True)
def mapping(map_col, mapping):

    X_train[map_col]=X_train[map_col].map(mapping)

    X_test[map_col]=X_test[map_col].map(mapping)
Age_mapping = {'Under 18 years old':18, '18 - 24 years old':24, '25 - 34 years old':34, '35 - 44 years old':44,

              '45 - 54 years old':54, '55 - 64 years old':64, '65 years or older':65}



CompanySize_mapping ={'Fewer than 10 employees':10, '10 to 19 employees':19, 

                      '20 to 99 employees':99, '100 to 499 employees':499, '500 to 999 employees':999, 

                      '1,000 to 4,999 employees':4999, '5,000 to 9,999 employees':9999,

                     '10,000 or more employees':20000}



Exercise_mapping = {'I don\'t typically exercise':0, '1 - 2 times per week':2, 

                    '3 - 4 times per week':4, 'Daily or almost every day':7}



HoursComputer_mapping = {'Less than 1 hour':1, '1 - 4 hours':4, '5 - 8 hours':8, '9 - 12 hours':12, 'Over 12 hours':15}



HoursOutside_mapping = {'Less than 30 minutes':0.5, '30 - 59 minutes':1, '1 - 2 hours':2, '3 - 4 hours':4, 'Over 4 hours':6}



LastNewJob_mapping = {'I\'ve never had a job':0, 'Less than a year ago':1, 'Between 1 and 2 years ago':2, 

                      'Between 2 and 4 years ago':4, 'More than 4 years ago': 5}



SkipMeals_mapping = {'Never':0, '1 - 2 times per week':2, '3 - 4 times per week':4, 'Daily or almost every day':7}



StackOverflowJobsRecommend_mapping = {'0 (Not Likely)':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,

                                      '10 (Very Likely)':10}



YearsCoding_mapping ={'0-2 years':2, '3-5 years':5, '6-8 years': 8, '9-11 years':11, '12-14 years':14, '15-17 years':17,

                      '18-20 years':20, '21-23 years':23, '24-26 years':26, 

                      '27-29 years':29, '30 or more years':25}



YearsCodingProf_mapping = YearsCoding_mapping
mapping('Age', Age_mapping)

mapping('CompanySize', CompanySize_mapping)

mapping('Exercise', Exercise_mapping)

mapping('HoursComputer', HoursComputer_mapping)

mapping('HoursOutside', HoursOutside_mapping)

mapping('LastNewJob', LastNewJob_mapping)

mapping('SkipMeals', SkipMeals_mapping)

mapping('StackOverflowJobsRecommend', StackOverflowJobsRecommend_mapping)

mapping('YearsCoding', YearsCoding_mapping)

mapping('YearsCodingProf', YearsCodingProf_mapping)
count_col = ['DevType', 'CommunicationTools', 'FrameworkWorkedWith'] 

for i in count_col:

    X_train[i + '_count'] = X_train[i].apply(lambda x: len(re.split('\s*;\s*', x)))

    X_test[i + '_count'] = X_test[i].apply(lambda x: len(re.split('\s*;\s*', x)))
X_train['Github'] = X_train['CommunicationTools'].str.contains('Github')

X_test['Github'] = X_test['CommunicationTools'].str.contains('Github')



X_train['Full-stack'] = X_train['DevType'].str.contains('Full-stack')

X_test['Full-stack'] = X_test['DevType'].str.contains('Full-stack')



X_train['DataScientist'] = X_train['DevType'].str.contains('Data scientist')

X_test['DataScientist'] = X_test['DevType'].str.contains('Data scientist')
del X_train['Country']

del X_test['Country']



del X_train['DevType']

del X_test['DevType']



del X_train['CommunicationTools']

del X_test['CommunicationTools']



del X_train['FrameworkWorkedWith']

del X_test['FrameworkWorkedWith']



#不要と思えるものも削除

del X_train['MilitaryUS']

del X_test['MilitaryUS']



del X_train['SurveyTooLong']

del X_test['SurveyTooLong']



del X_train['SurveyEasy']

del X_test['SurveyEasy']
X_train['AssessJob_null'] = X_train['AssessJob1'].isnull().astype(int)

X_test['AssessJob_null'] = X_test['AssessJob1'].isnull().astype(int)

X_train['AssessBenefits_null'] = X_train['AssessBenefits1'].isnull().astype(int)

X_test['AssessBenefits_null'] = X_train['AssessBenefits1'].isnull().astype(int)

X_train['JobContactPriorities_null'] = X_train['JobContactPriorities1'].isnull().astype(int)

X_test['JobContactPriorities_null'] = X_test['JobContactPriorities1'].isnull().astype(int)

X_train['JobEmailPriorities_null'] = X_train['JobEmailPriorities1'].isnull().astype(int)

X_test['JobEmailPriorities_null'] = X_train['JobEmailPriorities1'].isnull().astype(int)



X_train['questionaare_missing_count'] = X_train['AssessJob_null'] + X_train['AssessBenefits_null'] + X_train['JobContactPriorities_null'] + X_train['JobEmailPriorities_null']

X_test['questionaare_missing_count'] = X_test['AssessJob_null'] + X_test['AssessBenefits_null'] + X_test['JobContactPriorities_null'] + X_test['JobEmailPriorities_null']



del X_train['AssessJob_null']

del X_test['AssessJob_null']

del X_train['AssessBenefits_null']

del X_test['AssessBenefits_null']

del X_train['JobContactPriorities_null']

del X_test['JobContactPriorities_null']

del X_train['JobEmailPriorities_null']

del X_test['JobEmailPriorities_null']
# X_train[X_train['AssessJob_null'] >0]

# X_train.head(50)
col_yj_transform = ['GDP ($ per capita)']



for col in col_yj_transform:

    pt = PowerTransformer(method='yeo-johnson')

    reshape_train = X_train[col].values.reshape(-1,1)

    reshape_test = X_test[col].values.reshape(-1,1)     

    pt.fit(reshape_train)

    X_train[col] = pt.transform(reshape_train)

    X_test[col] = pt.transform(reshape_test)
cats = []

for col in X_train.columns:

    if (X_train[col].dtype == 'object'):

        cats.append(col)

     

        print(col, X_train[col].nunique())
# for count_col in cats:

#     summary = X_all[count_col].value_counts() #X_allにしていることに注意





ordinal_col = cats

ordinal_encoder = OrdinalEncoder(cols = ordinal_col)

X_train = ordinal_encoder.fit_transform(X_train)

X_test = ordinal_encoder.transform(X_test)
X_train.fillna(-9999, inplace = True)

X_test.fillna(-9999, inplace = True)
X_train
for col in X_train.columns:

    print(X_train[col].value_counts())

    print('____')
# #######

# tfidf = TfidfVectorizer(max_features = 50)



# TXT_train_DevType = tfidf.fit_transform(TXT_train_DevType)

# TXT_test_DevType = tfidf.transform(TXT_test_DevType)



# X_train = sp.sparse.hstack([X_train, TXT_train_DevType])

# X_test = sp.sparse.hstack([X_test, TXT_test_DevType])



# X_train = pd.DataFrame(X_train.todense())

# X_test = pd.DataFrame(X_test.todense())



# #######

# tfidf = TfidfVectorizer(max_features = 50)



# TXT_train_CommunicationTools = tfidf.fit_transform(TXT_train_CommunicationTools)

# TXT_test_CommunicationTools = tfidf.transform(TXT_test_CommunicationTools)



# X_train = sp.sparse.hstack([X_train, TXT_train_CommunicationTools])

# X_test = sp.sparse.hstack([X_test, TXT_test_CommunicationTools])



# X_train = pd.DataFrame(X_train.todense())

# X_test = pd.DataFrame(X_test.todense())



# #######

# tfidf = TfidfVectorizer(max_features = 50)



# TXT_train_FrameworkWorkedWith = tfidf.fit_transform(TXT_train_FrameworkWorkedWith)

# TXT_test_FrameworkWorkedWith = tfidf.transform(TXT_test_FrameworkWorkedWith)



# X_train = sp.sparse.hstack([X_train, TXT_train_FrameworkWorkedWith])

# X_test = sp.sparse.hstack([X_test, TXT_test_FrameworkWorkedWith])



# X_train = pd.DataFrame(X_train.todense())

# X_test = pd.DataFrame(X_test.todense())

X_train
%%time



num_split = 5

num_iter = 3

stop_round = 50

scores = []

y_pred_cva = np.zeros(len(X_test)) #cvaデータ収納用



scores = []



for h in range (num_iter):

    kf = KFold(n_splits=num_split, random_state=h, shuffle=True)



    for i, (train_ix, test_ix) in tqdm(enumerate(kf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

        clf = LGBMRegressor(n_estimators=9999, random_state=71, colsample_bytree=0.9,

                            learning_rate=0.05, min_child_samples=20, #max_depth=-1,

                            min_child_weight=0.001, min_split_gain=0.0, num_leaves=15) 

           

        clf.fit(X_train_, y_train_, early_stopping_rounds=stop_round, eval_metric='rmse', eval_set=[(X_val, y_val)])  



        y_pred = clf.predict(X_val)

        score = mean_squared_error(y_val, y_pred)**0.5 #RMSEだが、既にlog変換しているので、実質RMSLE

        scores.append(score)



        y_pred_cva += clf.predict(X_test)

        print(clf.predict(X_test))

            

        

print(np.mean(scores))

print(scores)



y_pred_cva /= (num_split * num_iter)
y_pred_exp = np.exp(y_pred_cva) - 1

y_pred_exp
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(by=['importance'], ascending=False)

imp.head(50)
fig, ax = plt.subplots(figsize=(20, 10))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
submission = pd.read_csv('../input/exam-for-students20200129/sample_submission.csv', index_col=0)



submission.ConvertedSalary = y_pred_exp
# submission[target] = submission[target].apply(round)



# max_threshold = 2000000

# min_threshold = 1000



# submission.loc[submission[target] > max_threshold, target] = max_threshold

# submission.loc[submission[target] < min_threshold, target] = 0
submission.to_csv('submission.csv')

submission