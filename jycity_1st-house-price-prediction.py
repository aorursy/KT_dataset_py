import tensorflow as tf

tf.__version__
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

import keras

import sklearn



plt.style.use('seaborn')

sns.set(font_scale=2) # 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편함.

#import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore') # 워닝 메세지 생략



#노트북 안에 그래프를 그리기 위해

%matplotlib inline
os.listdir("../input")
# 데이터 전체

df_train = pd.read_csv('../input/train.csv')

# 테스트 해야할 파일

df_test = pd.read_csv('../input/test.csv')

# 캐글에 제출할 파일

df_submit = pd.read_csv('../input/sample_submission.csv')
df_train.shape, df_test.shape, df_submit.shape
df_test.columns
df_train.columns
df_submit.columns
df_train.head()
df_test.head()
df_submit.head()
df_train.dtypes
#pandas 의 describe()를 이용해 각 feature가 가진 통계 반환

df_train.describe()
# bathroons 데이터 분포도 확인

value_counts = df_train['bathrooms'].value_counts()

print(value_counts)

print(type(value_counts))
df_test.describe()
obj_df = df_train.select_dtypes(include=['object']).copy()

obj_df.head()
# 상관관계 확인

cor_mat = df_train.corr()

cor_mat
#price와의 상관관계 내림차순 주요 피쳐 확인

cor_mat["price"].sort_values(ascending = False)
#heatmap으로 상관관계 확인



cor_abs = abs(df_train.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='price').index # price과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(df_train[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
#설립 연도별 분포

df_train["yr_built"].hist(bins = 100, figsize = (10, 10))
# 연속형 변수의 분포

continuous = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

plt.figure(figsize=(10,20))

count = 1

for col in continuous:

    plt.subplot(len(continuous), 1, count)

    # plt.title(col)

    sns.distplot(df_train[col])

    count += 1

plt.tight_layout()
# 비연속형 변수의 빈도분포

# yr_built, yr_renovated, zipcode는 제외

non_continuous = ['bedrooms', 'bathrooms', 'floors','waterfront', 'view', 'condition', 'grade']

plt.figure(figsize=(10,20))

count = 1

for col in non_continuous:

    plt.subplot(len(non_continuous), 1, count)

    # plt.title(col)

    sns.countplot(x=col,data=df_train)

    count += 1

plt.tight_layout()
df_train.columns
# 가장 높은 grade의 

sns.boxplot(data=df_train, x='grade', y='price')
# sqft_living 과의 상관분포 확인

sns.jointplot(data=df_train, x='sqft_living', y='price', kind="reg", color="m",height=8)

plt.show()
# sqft_living15

sns.jointplot(data=df_train, x='sqft_living15', y='price',kind="reg", color="m",height=8)

plt.show()
# sqft_above

sns.jointplot(data=df_train, x='sqft_above', y='price',kind="reg", color="m",height=8)

plt.show()
plt.figure(figsize=(14,7))

plt.subplot(1,2,1)

plt.title("Location with Density")

sns.scatterplot(data=df_train, x='long', y='lat', alpha=0.3, color='k')

plt.subplot(1,2,2)

plt.title("Location with Price")

sns.scatterplot(data=df_train, x='long', y='lat', alpha=0.3, color='r', hue='price')

# hue = '칼럼'  <- 카테고리 변수를 설정하여 변수에 따른 색 분포를 지정해 줄 수 있음
plt.figure(figsize=(10,6))

sns.distplot(df_train['price'])
# log를 취하여 다시 그려보자

plt.figure(figsize=(10,6))

sns.distplot(np.log1p(df_train['price']))
train = pd.read_csv("../input/train.csv")
train.drop(columns=['id'], inplace=True)

df = train.copy()
train.shape
train.columns
train['buy_year'] = train['date'].map(lambda x : int(x.split('T')[0][:4]))

train['buy_month'] = train['date'].map(lambda x : int(x.split('T')[0][4:6]))

train['buy_day'] = train['date'].map(lambda x : int(x.split('T')[0][6:]))
train.head(10)
train.drop(columns=['date'], inplace=True)
train.columns
train[['buy_year', 'buy_month','buy_day', 'price']].corr()
train['view'].value_counts() # 1~5까지 맞춰서 condition하고 엮어보자.