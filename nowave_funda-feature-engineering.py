%matplotlib inline



from matplotlib.pylab import rcParams, style    

rcParams['figure.figsize'] = 12, 8

rcParams['font.family'] = 'AppleGothic'

rcParams['font.size'] = 10

style.use('ggplot')
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/funda_train.csv')

df.head()
df.loc[(df.store_id == 538) & (df.transacted_date > '2018-12-01') ].sum()['amount']
df.loc[(df.store_id == 1408) & (df.transacted_date > '2018-12-01') ].sum()['amount']
df.loc[(df.store_id == 1601) & (df.transacted_date > '2018-12-01') ].sum()['amount']
df.shape
df.info()
df.isnull().sum()
# 수량이 0인 data 수

df.loc[df['amount'] < 0].shape
df.shape
# 수량이 0 이상인 data 수

#df = df.loc[df['amount'] > 0]
df.shape
pd.isnull(df['store_id']).sum()
df['transacted_date'] = df['transacted_date'] + ' ' + df['transacted_time']
df['transacted_date'].unique()
df['transacted_date'] = pd.to_datetime(df['transacted_date'])
# 최초일부터 현재까지 표시

print('Date Range: %s ~ %s' % (df['transacted_date'].min(), df['transacted_date'].max()))
df.loc[df['transacted_date'] < '2016-08-01'].shape
df.shape
df = df.loc[df['transacted_date'] >= '2016-08-01']
print('Date Range: %s ~ %s' % (df['transacted_date'].min(), df['transacted_date'].max()))
df.shape
temp = df.groupby('store_id').sum()['amount']

temp.shape
# store_id 별 card_id별 Sales 합과 인보이스날짜 최대값

orders_df = df.groupby(['store_id', 'card_id']).agg({'amount': sum,'transacted_date': max})
orders_df
def groupby_sum(x):

    return x.sum()



def groupby_mean(x):

    return x.mean()



def groupby_count(x):

    return x.count()



def purchase_duration(x):

    return (x.max() - x.min()).days



def avg_frequency(x):

    return (x.max() - x.min()).days / x.count()



groupby_sum.__name__ = 'sum'

groupby_mean.__name__ = 'avg'

groupby_count.__name__ = 'count'

purchase_duration.__name__ = 'purchase_duration'

avg_frequency.__name__ = 'purchase_frequency'
summary_df = orders_df.reset_index().groupby('store_id').agg({'amount': [min, max, sum, 

                                                                          groupby_mean, groupby_count],'transacted_date': [min, max, purchase_duration, avg_frequency]})

summary_df.round()
# Index를 붙여넣고 변수명을 소문자로 변환

summary_df.columns = ['_'.join(col).lower() for col in summary_df.columns]
summary_df.head()
summary_df.loc[summary_df.index == 538]
summary_df.shape
# duration 구매기간이 0인 데이터 확인

summary_df.loc[summary_df['transacted_date_purchase_duration'] < 1]
# Sales_count를 기준으로 Sales avg값을 시각화

ax = summary_df.groupby('amount_count').count()['amount_avg'][:200].plot(

    kind='bar', 

    color='skyblue',

    figsize=(12,7), 

    grid=True

)



ax.set_ylabel('count')



plt.show()
summary_df['amount_count'].describe()
summary_df['amount_avg'].describe()
# 구매날짜 인보이스 빈도 히스토그램 시각화 

ax = summary_df['transacted_date_purchase_frequency'].hist(

    bins=20,

    color='skyblue',

    rwidth=0.7,

    figsize=(12,7)

)



ax.set_xlabel('avg. number of days between purchases')

ax.set_ylabel('count')



plt.show()
summary_df['transacted_date_purchase_frequency'].describe()
summary_df['transacted_date_purchase_duration'].describe()
df.info()
# 3개월 인자 생성 Customer Value 측정

clv_freq = '3M'
data_df = orders_df.reset_index().groupby(['store_id', 

                                           pd.Grouper(key='transacted_date', freq=clv_freq)]).agg({'amount': [groupby_sum, groupby_mean, groupby_count],})

data_df.head(10)
# 컬럼명에 인덱스를 붙이고 소문자로 컬럼명 변환

data_df.columns = ['_'.join(col).lower() for col in data_df.columns]
data_df.head()
data_df = data_df.reset_index()
data_df.head(10)
# 3개월 단위로 맵핑

date_month_map = {

    str(x)[:10]: 'M_%s' % (i+1) for i, x in enumerate(sorted(data_df.reset_index()['transacted_date'].unique(), reverse=True))

}

date_month_map
data_df['M'] = data_df['transacted_date'].apply(lambda x: date_month_map[str(x)[:10]])
date_month_map
data_df.head(12)
features_df = pd.pivot_table(data_df.loc[data_df['M'] != 'M_1'], 

                             values=['amount_sum', 'amount_avg', 'amount_count'], columns='M', index='store_id')

features_df
# 인덱스를 컬럼명에 붙인다.

features_df.columns = ['_'.join(col) for col in features_df.columns]
features_df.head()
features_df.shape
# Null에 0 채우기

features_df = features_df.fillna(0)
features_df
# response 반응에 M1 값만 CustomerID와 sales_sum을 데이터프레임으로 불러오기

response_df = data_df.loc[data_df['M'] == 'M_1', ['store_id', 'amount_sum']]

response_df.head()
# response_df의 변수명을 변경

response_df.columns = ['store_id', 'CLV_' + clv_freq]
response_df.shape
response_df.head()
response_df.loc[response_df.store_id == 538]['CLV_3M']
sample_set_df = features_df.merge(response_df,  left_index=True, right_on='store_id', how='left')
sample_set_df.shape
sample_set_df
# 결측치 처리

sample_set_df = sample_set_df.fillna(0)
sample_set_df['CLV_' + clv_freq].describe().round()
sample_set_df.to_csv('sample_set.csv')
fig, ax = plt.subplots()

ax.scatter(x = sample_set_df['store_id'], y = sample_set_df['CLV_3M'])

plt.ylabel('3M acmount', fontsize=12)

plt.xlabel('storeid', fontsize=12)

plt.show()
from sklearn.model_selection import train_test_split
target_var = 'CLV_' + clv_freq

all_features = [x for x in sample_set_df.columns if x not in ['store_id', target_var]]
target_var
all_features
from sklearn.linear_model import LinearRegression



# Try these models as well

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor
X = sample_set_df[all_features]

y = sample_set_df[target_var]
reg_fit = LinearRegression()
rfg_fit = RandomForestRegressor()
reg_fit.fit(X, y)
rfg_fit.fit(X, y)
reg_fit.intercept_
coef = pd.DataFrame(list(zip(all_features, reg_fit.coef_)))

coef.columns = ['feature', 'coef']



coef
from sklearn.metrics import r2_score, median_absolute_error
reg_preds =  reg_fit.predict(X)
rfg_preds = rfg_fit.predict(X)
# 데이콘이 제공하는 제출 포멧(sampleSubmission.csv)을 읽어옵니다.

submission = pd.read_csv("submission.csv")



print(submission.shape)

submission.head()
submission['amount'] = reg_preds

print(submission.shape)



# submission 데이터의 상위 5개를 띄웁니다.

submission
submission.to_csv('../input/submission#3.csv', index=False)