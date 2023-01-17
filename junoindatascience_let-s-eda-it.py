import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy as sp

import warnings

import seaborn as sns

import random

warnings.filterwarnings('ignore')
train = pd.read_csv("../dataset/train.csv")

train.tail()
test = pd.read_csv("../dataset/test.csv")

test.tail()
train.info()

cols = []

for col in train.columns[1:]:

    if train[col].dtype == 'object':

        cols.append(col)

print("object data type을 가진 columns : ",cols)
df_train = train[[column for column in train.columns if column != cols[0] and column != 'id']]

df_train.describe()
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
df_train.describe()
df_train[df_train.sqft_living == 13540.000000]
df_train.tail()
for col in df_train.columns:

    print("column : ", col)

    print("{}의 unique value 갯수 : {}\n".format(col, df_train[col].nunique()))
# 결측치 시각화

# 본 코드는 현우님 kernel을 참고하였습니다. (https://www.kaggle.com/chocozzz/house-price-prediction-eda-updated-2019-03-12)

# yr_built, yr_renovated, zipcode는 제외하였다.

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
uniq = dict()

cols = ['bedrooms', 'bathrooms', 'floors','waterfront', 'view', 'condition', 'grade']



for col in cols:

    uniq[col] = len(df_train[col].unique())



def generate_color():

    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))

    return color

    

data = [

    go.Bar(

        x = list(uniq.keys()), # ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']

        y = list(uniq.values()), # [11, 29, 6, 2, 5, 5, 12]

        name = 'Unique value in features',

        textfont=dict(size=20),

        marker=dict(

        line=dict(

            color= generate_color(),

            #width= 2,

        ), opacity = 0.45

    )

    ),

    ]

layout= go.Layout(

        title= "Unique Value By Column",

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='skin')
# correlation이 높은 상위 10개의 heatmap

# continuous + sequential variables --> spearman

# abs는 반비례관계도 고려하기 위함

cor_abs = abs(df_train.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='price').index # price과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(df_train[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
# 가장 높은 grade를 따로 뽑아보자

# grade는 순서의 의미를 가진 변수

# 확률변수값 x price --> boxplot

sns.boxplot(data=df_train, x='grade', y='price')
# 다음으로 높은 sqft_living

sns.jointplot(data=df_train, x='sqft_living', y='price',kind="reg", color="m",height=8)

plt.show()
# sqft_living15

sns.jointplot(data=df_train, x='sqft_living15', y='price',kind="reg", color="m",height=8)

plt.show()
# sqft_above

sns.jointplot(data=df_train, x='sqft_above', y='price',kind="reg", color="m",height=8)

plt.show()
# bathrooms

plt.figure(figsize=(18,5))

sns.boxplot(data=df_train, x='bathrooms', y='price')

plt.show()
# bedrooms

plt.figure(figsize=(10,5))

sns.boxplot(data=df_train, x='bedrooms', y='price')

plt.show()
# floors

plt.figure(figsize=(8,5))

sns.boxplot(data=df_train, x='floors', y='price')

plt.show()
# view

plt.figure(figsize=(7,5))

sns.boxplot(data=df_train, x='view', y='price')

plt.show()
plt.figure(figsize=(14,7))

plt.subplot(1,2,1)

plt.title("Location with Density")

sns.scatterplot(data=df_train, x='long', y='lat', alpha=0.3, color='k')

plt.subplot(1,2,2)

plt.title("Location with Price")

sns.scatterplot(data=df_train, x='long', y='lat', alpha=0.3, color='r', hue='price')
plt.figure(figsize=(10,6))

sns.distplot(df_train['price'])
# log를 취하여 다시 그려보자

plt.figure(figsize=(10,6))

sns.distplot(np.log1p(df_train['price']))
# latitude와 price가 어느정도 상관관계가 있는것으로 나왔다.

sns.jointplot(data=df_train, x='lat', y='price', kind='reg', height=7, color='r')
train.drop(columns=['id'], inplace=True)

df = train.copy()
df['buy_year'] = df['date'].map(lambda x : int(x.split('T')[0][:4]))

df['buy_month'] = df['date'].map(lambda x : int(x.split('T')[0][4:6]))

df['buy_day'] = df['date'].map(lambda x : int(x.split('T')[0][6:]))
df.drop(columns=['date'], inplace=True)
df[['buy_year', 'buy_month','buy_day', 'price']].corr()
df.columns
df['view'].value_counts() # 1~5까지 맞춰서 condition하고 엮어보자.
df[['view', 'price']].corr()
df['condition'].value_counts()
df[['condition', 'price']].corr()
# condition view
df['zipcode'].value_counts()
# 변수 다시 Load

train.drop(columns=['id'], inplace=True)

df = train.copy()



df['buy_year'] = df['date'].map(lambda x : int(x.split('T')[0][:4]))

df['buy_month'] = df['date'].map(lambda x : int(x.split('T')[0][4:6]))

df['buy_day'] = df['date'].map(lambda x : int(x.split('T')[0][6:]))



df.drop(columns=['date'], inplace=True)



df.tail()
df.corr(method='spearman')
df.tail()
df.columns
# sqft 단위를 평수로 scaling

cols = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15','sqft_lot15']

for col in cols:

    feature_name = "{}_kor".format(col)

    df[feature_name] = df[col] * 0.0281

df.tail()    
sns.distplot(np.log1p(df['sqft_living_kor']))
sum(df['sqft_living'] != (df['sqft_above'] + df['sqft_basement'])) # 0

# sum(df['sqft_living_kor'] != (df['sqft_above_kor'] + df['sqft_basement_kor'])) # not 0, scaling하면서 오차가 생긴듯하다.
# 주거면적 > 부지면적 ; aka 1.5층, 2층, 3층, 3.5층 집~

sample_df = df[(df['sqft_living'] / df['sqft_lot']) > 1] 

sample_df[sample_df.floors < 2]
np.corrcoef(df['price'], (df['sqft_living'] / df['sqft_lot']))
np.corrcoef(df['price'], (df['sqft_above'] / df['sqft_living']))
len(df[df.sqft_living != df.sqft_living15][df.yr_renovated == 0])
print("내 집이 주변 이웃보다 더 큰 경우 : ", len(df[df.sqft_living > df.sqft_living15]))

print("내 집이 주변 이웃과 같은 경우 : ", len(df[df.sqft_living == df.sqft_living15]))

print("내 집이 주변 이웃보다 더 작은 경우 : ", len(df[df.sqft_living < df.sqft_living15]))
sns.distplot(df['sqft_living'] - df['sqft_living15'])
df['zipcode'].nunique()
df2 = df.copy()
# 평당가격 (실제론 피트당 가격)

# sqft_living

df2['per_price'] = df2['price'] / df2['sqft_living']

price_per_zipcode = df2.groupby(['zipcode'])['per_price'].agg({'zipprice_mean' : 'mean', 'zipprice_std' : np.std}).reset_index()

price_per_zipcode.tail()
price_per_zipcode['zipprice_mean'].describe()
# merge df2 and price_per_zipcode

df2 = df2.merge(price_per_zipcode, how='left', on='zipcode')

df2.tail()
df2[df2.zipprice_mean > 317.0].zipcode.unique(), df2[df2.zipprice_mean > 317.0].zipcode.nunique()
df2.tail()
df2[df2.zipprice_mean > 317.0].zipcode.unique(), df2[df2.zipprice_mean > 317.0].zipcode.nunique()
sample = df2[df2.zipprice_mean > 317.0]

sample.shape[0]
a = sample.loc[(sample.lat < 47.8) & (sample.lat > 47.5), :]

plt.figure(figsize=(13,13))

sns.scatterplot(data=a, x='long', y='lat', hue='zipprice_mean', legend='full', palette='Set1', alpha=0.5)
plt.figure(figsize=(10,10))

sns.scatterplot(data=a, x='long', y='lat', hue='zipcode', legend='full', palette='Set1')

plt.tight_layout()
a.corr(method='spearman')
np.corrcoef(a['price'], a['sqft_living'])
np.corrcoef(a['price'], a['sqft_living15'])
print(np.corrcoef(a['price'], a['sqft_lot']))

print()

print(np.corrcoef(a['price'], a['sqft_lot15']))
np.corrcoef(df['price'], (df['bedrooms'] / (df['bedrooms'] + df['bathrooms'])))
np.corrcoef(df['price'],  (df['bedrooms'] + df['bathrooms']))
df[df.bedrooms == 0]
df[df.bathrooms == 0.]
df[df.bathrooms == 0.][['sqft_living','sqft_lot','sqft_living15','sqft_lot15']]
renovated_y = df[df.yr_renovated != 0]

renovated_n = df[df.yr_renovated == 0]



print(np.corrcoef(df.loc[df.yr_renovated != 0, 'price'], renovated_y['sqft_living']))

print()

print(np.corrcoef(df.loc[df.yr_renovated == 0, 'price'], renovated_n['sqft_living']))
# 1900~2015년까지

df_train['yr_built'].describe()
train.drop(columns=['id'], inplace=True)

df = train.copy()



df['buy_year'] = df['date'].map(lambda x : int(x.split('T')[0][:4]))

df['buy_month'] = df['date'].map(lambda x : int(x.split('T')[0][4:6]))

df['buy_day'] = df['date'].map(lambda x : int(x.split('T')[0][6:]))



df.drop(columns=['date'], inplace=True)
# 먼저 yr_built 지어진지 몇년되었는 확인 (2015년 현재)

df['Years_of_construction'] = 2015 - df['yr_built']

df.tail()
np.corrcoef(np.log1p(df['price']), np.log1p(df['Years_of_construction']))
sns.jointplot(data=df, x='Years_of_construction', y='price')
# 지어진지 얼마나 되었는지에 따라 가격의 변동이 있는가? 패턴이 있는가?

# 5년,10년 단위로 묶어보자

plt.figure(figsize=(14,5))



plt.subplot(1,3,1)

plt.title("Years of construction ~4")

sns.boxplot(data=df[df['Years_of_construction'] < 5], x='Years_of_construction', y='price')



plt.subplot(1,3,2)

plt.title("Years of construction 5~9")

sns.boxplot(data=df[(df['Years_of_construction'] >= 5) & (df['Years_of_construction'] < 10)], x='Years_of_construction', y='price')



plt.subplot(1,3,3)

plt.title("Years of construction 10~19")

sns.boxplot(data=df[(df['Years_of_construction'] >= 10) & (df['Years_of_construction'] < 20)], x='Years_of_construction', y='price')



plt.tight_layout()

plt.show()
# 미국 대선이 있는 11월기준으로 앞뒤 10월, 12월의 가격을 보자

# 2012년 10월 11월 12월 (가장 가까운 대선기준)

# buy_year말고 yr_built로 봐보자 (buy_year는 없으므로)

df['buy_year'].value_counts()
# 2014년/2015년 집값의 분포 각가봐보자

plt.figure(figsize=(10,5))



plt.subplot(1,2,1)

plt.title("2014 price distribution")

sns.distplot(df[df['buy_year'] == 2014]['price'])



plt.subplot(1,2,2)

plt.title("2015 price distribution")

sns.distplot(df[df['buy_year'] == 2015]['price'])



plt.tight_layout()
# 대선이있는 연도에 지어진집

df2 = df.copy()

df2['election_year'] = df['yr_built'].map(lambda x: 1 if x % 4 == 0 else 0)

df2[['price','election_year']].corr(method='spearman')
election_y = df2[df2.election_year != 0]

election_n = df2[df2.election_year == 0]



print(np.corrcoef(df2.loc[df2.election_year != 0, 'price'], election_y['sqft_living']))

print()

print(np.corrcoef(df2.loc[df2.election_year == 0, 'price'], election_n['sqft_living']))
df.buy_year.value_counts()
df_train = pd.read_csv("../dataset/train.csv")

df_train.tail()
df_train['buy_year'] = df_train['date'].map(lambda x : int(x.split('T')[0][:4]))

df_train['buy_month'] = df_train['date'].map(lambda x : int(x.split('T')[0][4:6]))

df_train.drop(columns=['id','date'], inplace=True)
df_train['bathrooms'].unique()
def is_center(train):

    # 평당가격 (실제론 피트당 가격)

    # sqft_living

    train['per_price'] = train['price'] / train['sqft_living']

    price_per_zipcode = train.groupby(['zipcode'])['per_price'].agg({'zipprice_mean' : 'mean', 'zipprice_std' : np.std}).reset_index()



    # 평당가격이 317 이상되면 중심부가 아닐까?



    # merge df2 and price_per_zipcode

    train = train.merge(price_per_zipcode, how='left', on='zipcode')



    train_idx = train[(train.zipprice_mean > 317.) & (train.lat >= 47.5) & (train.lat < 47.8)].index

    train['is_center'] = 0

    train.loc[train_idx, 'is_center'] = 1

    

    train.drop(columns=['zipprice_mean','per_price','zipprice_std'], inplace=True)

    return train
df_train = is_center(df_train)
df_train.describe()
# 지역성에서 고려할만한 변수1

# 거리의 개념

# 가장 비싼 집으로부터 떨어진 거리?

df_train[df_train['price'] == df_train['price'].max()]
sp.linalg.norm(np.array(df_train[['lat','long']])[0] - np.array(df_train[['lat','long']]), axis=1)
sp.linalg.norm([1,1])
# 거리측정하는 함수 만들자

# 위도 경도로 정확하게 구할 것인가

# 대략적인 norm으로 구할 것인가 --> 이걸로 구해보자

def get_distance_from_max(train):

    max_idx = train[train['price'] == train['price'].max()].index[0]

    max_location = np.array(df_train[['lat','long']].loc[max_idx,:]) # 가장 비싼 집의 좌표

    location = np.array(df_train[['lat','long']])

    

    # compute distance : ||location1 - location2||

    distance_from_max = sp.linalg.norm(max_location - location, axis=1)

    return distance_from_max
dist = get_distance_from_max(df_train)

df_train['distance_from_max'] = dist

df_train[['distance_from_max']].describe()
sns.jointplot(x='distance_from_max', y='price', data=df_train, kind='reg')

plt.show()
# 위에서 나눈 중심지와 비중심지로 나눠서 distance의 영향을 보자

plt.figure(figsize=(10,8))



#plt.subplot(1,2,1)

#plt.title("in Center region")

sns.jointplot(x='distance_from_max', y='price', data=df_train[df_train['is_center']==1], kind='reg')



#plt.subplot(1,2,2)

#plt.title("in Non-Center region")

sns.jointplot(x='distance_from_max', y='price', data=df_train[df_train['is_center']==0], kind='reg')



plt.tight_layout()

plt.show()
plt.scatter(1/(df_train['distance_from_max']), df_train['price'])
# 지역성에서 고려할만한 변수2 : 면적과 접목

# 가장 비싼 지역여부

# 비싸다는 기준은 무엇인가?

    # price / sqft_living이 zipcode별로 높은 곳인가?

    # price / sqft_lot이 zipcode별로 높은 곳인가?

    # price / sqft_living15가 zipcode별로 높은 곳인가?

    # price / sqft_lot15이 zipcode별로 높은 곳인가?

    # price / sqft_above가 zipcode별로 높은 곳인가?

    # price / sqft_basement가 zipcode별로 높은 곳인가? --> 이건....음

def most_expensive_region(train):

    """

    expensive region의 기준은 주거면적 피트당 가격 ()

    """

    
# 지역성에서 고려할만한 변수2

# zipcode 일부지역 dummy 처리?
