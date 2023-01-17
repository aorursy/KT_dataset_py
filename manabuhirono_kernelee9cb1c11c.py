import os, glob

import pandas as pd

import numpy as np

import scipy as sp

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import quantile_transform

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import lightgbm as lgbm

import matplotlib.pyplot as plt

import seaborn as sns

from joblib import dump, load

import datetime



import warnings

warnings.filterwarnings('ignore')
filepath2 = "../input/exam-for-students20200129/"

# テーブルのインポート

df_train = pd.read_csv(filepath2+"train.csv")

df_test = pd.read_csv(filepath2+"test.csv")
y_train = df_train.ConvertedSalary

X_train = df_train.drop(['ConvertedSalary'], axis=1)

X_test = df_test
X_all = pd.concat([X_train, X_test], axis=0)

X_all.head()
# 外部データロード

df_country = pd.read_csv(filepath2+"country_info.csv")
# Train/Testにマージ

X_all = pd.merge(X_all,df_country,on="Country",how="left")
X_all['Pop. Density (per sq. mi.)'] = X_all['Pop. Density (per sq. mi.)'].str.replace(',', '.').astype(float)

X_all['Coastline (coast/area ratio)'] = X_all['Coastline (coast/area ratio)'].str.replace(',', '.').astype(float)

X_all['Net migration'] = X_all['Net migration'].str.replace(',', '.').astype(float)

X_all['Infant mortality (per 1000 births)'] = X_all['Infant mortality (per 1000 births)'].str.replace(',', '.').astype(float)

X_all['Literacy (%)'] = X_all['Literacy (%)'].str.replace(',', '.').astype(float)

X_all['Phones (per 1000)'] = X_all['Phones (per 1000)'].str.replace(',', '.').astype(float)

X_all['Arable (%)'] = X_all['Arable (%)'].str.replace(',', '.').astype(float)

X_all['Crops (%)'] = X_all['Crops (%)'].str.replace(',', '.').astype(float)

X_all['Other (%)'] = X_all['Other (%)'].str.replace(',', '.').astype(float)

X_all['Birthrate'] = X_all['Birthrate'].str.replace(',', '.').astype(float)

X_all['Deathrate'] = X_all['Deathrate'].str.replace(',', '.').astype(float)

X_all['Agriculture'] = X_all['Agriculture'].str.replace(',', '.').astype(float)

X_all['Industry'] = X_all['Industry'].str.replace(',', '.').astype(float)

X_all['Service'] = X_all['Service'].str.replace(',', '.').astype(float)
## RankGauss変換
nums = []

for col in X_all.columns:

    if X_all[col].dtype != 'object':

        nums.append(col)

        print(col, X_all[col].nunique())
X_all[nums] = quantile_transform(X_all[nums], n_quantiles=100, random_state=0, output_distribution='normal')
d = X_all['CompanySize'].value_counts().to_dict()

print(d)
d = X_all['YearsCoding'].value_counts().to_dict()

print(d)
d = X_all['YearsCodingProf'].value_counts().to_dict()

print(d)
d = X_all['LastNewJob'].value_counts().to_dict()

print(d)
d = X_all['TimeFullyProductive'].value_counts().to_dict()

print(d)
d = X_all['StackOverflowJobsRecommend'].value_counts().to_dict()

print(d)
d = X_all['HoursComputer'].value_counts().to_dict()

print(d)
d = X_all['HoursOutside'].value_counts().to_dict()

print(d)
d = X_all['SkipMeals'].value_counts().to_dict()

print(d)
d = X_all['Age'].value_counts().to_dict()

print(d)
# CompanySizeは順序があるため、順序に応じて変換する。

comsize_map = {'Fewer than 10 employees':10,'10 to 19 employees':20,'20 to 99 employees':100,'100 to 499 employees':500,'500 to 999 employees':1000,'1,000 to 4,999 employees':5000,'5,000 to 9,999 employees':10000,'10,000 or more employees':20000}

years_map = {'0-2 years':2,'3-5 years':5,'6-8 years':8,'9-11 years':11,'12-14 years':14,'15-17 years':17,'18-20 years':20,'21-23 years':23,'24-26 years':26,'27-29 years':29,'30 or more years':35}

newjob_map = {'I\'ve never had a job':0,'Less than a year ago':1,'Between 1 and 2 years ago':2,'Between 2 and 4 years ago':4,'More than 4 years ago':5}

timefully_map = {'Less than a month':1,'One to three months':3,'Three to six months':6,'Six to nine months':8,'Nine months to a year':9,'More than a year':12}

stack_map = {'0 (Not Likely)':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10 (Very Likely)':10}

hCom_map = {'Less than 1 hour':1,'1 - 4 hours':4,'5 - 8 hours':8,'9 - 12 hours':12,'Over 12 hours':15}

hOut_map = {'Less than 30 minutes':0,'30 - 59 minutes':1,'1 - 2 hours':2,'3 - 4 hours':4,'Over 4 hours':5}

meal_map = {'Never':0,'1 - 2 times per week':2,'3 - 4 times per week':4,'Daily or almost every day':7}

age_map = {'Under 18 years old':18,'18 - 24 years old':24,'25 - 34 years old':34,'35 - 44 years old':44,'45 - 54 years old':54,'55 - 64 years old':64,'65 years or older':70}
X_all['CompanySize'] = X_all['CompanySize'].map(comsize_map)

X_all['YearsCoding'] = X_all['YearsCoding'].map(years_map)

X_all['YearsCodingProf'] = X_all['YearsCodingProf'].map(years_map)

X_all['LastNewJob'] = X_all['LastNewJob'].map(newjob_map)

X_all['TimeFullyProductive'] = X_all['TimeFullyProductive'].map(timefully_map)

X_all['StackOverflowRecommend'] = X_all['StackOverflowRecommend'].map(stack_map)

X_all['StackOverflowJobsRecommend'] = X_all['StackOverflowJobsRecommend'].map(stack_map)

X_all['HoursComputer'] = X_all['HoursComputer'].map(hCom_map)

X_all['HoursOutside'] = X_all['HoursOutside'].map(hOut_map)

X_all['SkipMeals'] = X_all['SkipMeals'].map(meal_map)

X_all['Age'] = X_all['Age'].map(age_map)
del X_all['Region']

del X_all['Climate']
# カテゴリのユニーク数

cats = []

for col in X_all.columns:

    if X_all[col].dtype == 'object':

        cats.append(col)

        print(col, X_all[col].nunique(), X_train[col].nunique(), X_test[col].nunique())
# カテゴリをエンコーディング

from category_encoders import OrdinalEncoder

encoder = OrdinalEncoder(cols=cats)

X_all[cats] = encoder.fit_transform(X_all[cats])
X_all.isnull().any()
X_all.fillna(-9999, inplace=True)
# トレーニングデータ・テストデータに分割

X_train = X_all.iloc[:X_train.shape[0], :]

X_test = X_all.iloc[X_train.shape[0]:, :]
lgb_train = lgbm.Dataset(X_train, y_train)



# LightGBM のハイパーパラメータ

lgbm_params = {'objective': 'regression', 'metric': 'rmse'}



    # 上記のパラメータでモデルを学習する

model = lgbm.train(lgbm_params, lgb_train)
# テストデータを予測する

y_pred = model.predict(X_test)
submission = pd.read_csv(filepath2+'sample_submission.csv', index_col=0)
submission.ConvertedSalary = y_pred

submission.to_csv('submission.csv')