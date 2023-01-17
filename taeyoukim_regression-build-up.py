import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc

## 초심으로.. 데이터 불러오기.

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv') 

submission = pd.read_csv('../input/sample_submission.csv')
##컬럼(Feature) 확인하기

df_train.columns
#기술통계(Descriptive Statistics) 확인 

df_train['price'].describe()
#히스토그램으로 그려보기

sns.distplot(df_train['price']);
#skewness and kurtosis 점검합니다.

print("Skewness: %f" % df_train['price'].skew())

print("Kurtosis: %f" % df_train['price'].kurt())
#scatter plot with 'price'

var = 'sqft_living'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,5000000));
#scatter plot with 'price'

var = 'sqft_lot'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,5000000));
#scatter plot with 'price'

var = 'sqft_basement'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,5000000));
#scatter plot with 'price'

var = 'sqft_living15'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,5000000));
#scatter plot with 'price'

var = 'sqft_above'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='price', ylim=(0,5000000));
#box plot cat per price

var = 'waterfront'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="price", data=data)

fig.axis(ymin=0, ymax=5000000);
#box plot cat per price

var = 'view'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="price", data=data)

fig.axis(ymin=0, ymax=5000000);
#box plot cat per price

var = 'condition'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="price", data=data)

fig.axis(ymin=0, ymax=5000000);
#box plot cat per price

var = 'grade'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="price", data=data)

fig.axis(ymin=0, ymax=9000000);
## Correlation Matrix 생성 ( 히트맵 )



corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#price correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'price')['price'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
## sqft_living 은 price와 강한 상관관계에 있음을 확인.

## sqft_living은  grade,sqft_above, sqft_living15, bathrooms, bedrooms 등의 변수와 상관성이 높기에.. 차원축소 작업등이 필요해보임 
#scatterplot: price와 코릴레이션이 높은 변수들의 pairplot 그리기

sns.set()

cols = ['price', 'sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
## 결측치 찾기 - 결과::없음.

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
##Univariate 분석

#standardizing data

price_scaled = StandardScaler().fit_transform(df_train['price'][:,np.newaxis]);

low_range = price_scaled[price_scaled[:,0].argsort()][:10]

high_range= price_scaled[price_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#히스토그램과 정규 확률 분포 인지 확인 -> 결과 정규화가 필요한것으로 판단

sns.distplot(df_train['price'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['price'], plot=plt)
## 예측해야할 y (price) 값의 정규화 (Log Transformation)



df_train['price'] = np.log(df_train['price'])



#정규화 저용후 히스토그램 및 정규확률플롯 그리기

sns.distplot(df_train['price'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['price'], plot=plt)

#sqft_living 피쳐의 정규화

sns.distplot(df_train['sqft_living'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['sqft_living'], plot=plt)
#data transformation

df_train['sqft_living'] = np.log(df_train['sqft_living'])
#histogram and normal probability plot

sns.distplot(df_train['sqft_living'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['sqft_living'], plot=plt)
skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)

## 기존에 올라온 Feature Engineering 참조.
for df in [df_train,df_test]:

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    df['grade_condition'] = df['grade'] * df['condition']

    df['sqft_total'] = df['sqft_living'] + df['sqft_lot']

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

    
df_train['per_price'] = df_train['price']/df_train['sqft_total_size']

zipcode_price = df_train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

df_train = pd.merge(df_train,zipcode_price,how='left',on='zipcode')

df_test = pd.merge(df_test,zipcode_price,how='left',on='zipcode')

del df_train['per_price']