#필요 패키지를 모두 업로드한다.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



#PATH를 체크한다.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#데이터를 경로로부터 로드한다.

df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')



#Introduction

#이번 컴피티션의 문제는 캘리포니아 지역 집 가치의 중위값을 예측하는 것이다. 주어진 데이터를 구성하는 피처는 아래와 같다.

df.columns
#위도경도, 하우스연식 중앙값, 방의 개수, 화장실 개수, 인구, 세대수, 소득의 중앙값, 오션뷰 여부 등으로 있고 우리가 예측해야할 Y값은 'median_house_value' 임을 확인하였다.

#피처개수가 위도경도를 제외하고 보아도 7개 정도로 많지 않기 때문에 예측하려는 'median_house_value'와의 분포를 아래와 같이 그려 보았다.



sns.lmplot(data=df, x='housing_median_age',y='median_house_value')
sns.lmplot(data=df, x='total_rooms',y='median_house_value')
sns.lmplot(data=df, x='total_bedrooms',y='median_house_value')
sns.lmplot(data=df, x='population',y='median_house_value')
sns.lmplot(data=df, x='households',y='median_house_value')
sns.lmplot(data=df, x='median_income',y='median_house_value')
sns.countplot()
#ocean_proximity의 경우 문자열로 입력되어 있기 때문에 변환이 필요하다.

df = pd.get_dummies(df)

sns.lmplot(data=df, x='ocean_proximity_<1H OCEAN',y='median_house_value')
sns.lmplot(data=df, x='ocean_proximity_INLAND',y='median_house_value')
df.columns
#dataframe shape을 알아본다.

df.shape
#dataframe 상위 다섯줄만 불러온다.

df.head()
#dataframe의 기초통계를 파악한다.

df.describe()
#테이블로 보기 어렵기 때문에 분포를 그려서 시각화로 알아본다.

df.columns



sns.countplot(data=df, x='total_rooms')
#dataframe 내 결측치 확인

df.isnull().sum()



#아래 결과에 따르면 total_bedrooms 피처에 207개의 결측치가 존재한다.
#total_bedrooms의 결측치를 대체한다.
#seaborn을 활용한 pairplot 시각화를 수행한다.

sns.pairplot(df)
df[df['total_bedrooms'].isnull()] = df['total_bedrooms'].median()
df.isnull().count()