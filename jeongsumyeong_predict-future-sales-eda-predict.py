import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Matplolib for visualization

from matplotlib import pyplot as plt

# display plots in the notebook

%matplotlib inline



# Datetime deal with dates formats

import datetime as dt

import plotly.express as px

import plotly.graph_objects as go
train = pd.read_csv('../input/please-come/sales_train.csv')

test = pd.read_csv('../input/please-come/test.csv')

items = pd.read_csv('../input/please-come/items.csv')

items_categories = pd.read_csv('../input/please-come/item_categories.csv')

shops = pd.read_csv('../input/please-come/shops.csv')
print('test shop_id 갯수: ',len(test['shop_id'].unique()))

print('test item_id의 갯수: ',len(test['item_id'].unique()))
print('공통 shop_id 갯수: ', len(set(test['shop_id'].unique()) & set(train['shop_id'].unique())))

print('공통 item_id 갯수: ', len(set(test['item_id'].unique()) & set(train['item_id'].unique())))
train['item_cnt_day'].describe()
print(train['item_cnt_day'].min())

print(train['item_cnt_day'].quantile(0.01))

print(train['item_cnt_day'].quantile(0.99))

print(train['item_cnt_day'].max())



#마지막 값과 마지막 직전 값의 차이가 매우 크다는 것을 알 수 있음
plt.scatter(train['date'], train['item_cnt_day'])
train.isnull().sum() * 100 / len(train)
train.head(5)
train = train.merge(items, how = 'left', on='item_id').merge(shops,  how = 'left', on='shop_id').merge(items_categories, how='left', on='item_category_id')

train.head(5)
import datetime as dt

train['date'] = train.date.apply(lambda x:dt.datetime.strptime(x, '%d.%m.%Y'))
train['year'] = train['date'].dt.year

train['month'] = train['date'].dt.month

train['dayofweek'] = train['date'].dt.dayofweek

train['month_year'] = train['date'].dt.to_period('M')
train.head()
test_item_id = test['item_id'].unique()

test_shop_id = test['shop_id'].unique()



lk_train = train[train['item_id'].isin(test_item_id)]

print(train.shape, lk_train.shape) # 확 줄었다
lk_train.groupby(['month', 'year']).sum()['item_cnt_day'].unstack().plot(figsize=(13,5))

plt.xlabel('month')

plt.ylabel('Total item_cnt_day')

plt.show()
date_sum = lk_train.groupby(['year','date'])['item_cnt_day'].sum().reset_index()

fig = px.line(date_sum, x="date", y="item_cnt_day", title='Sales by Date', width=900, height=500, color='year')

fig.show()
lk_train.groupby(['month','year']).sum()['item_cnt_day'].unstack().plot(figsize=(13,3))

plt.xlabel('Month')

plt.ylabel('Total item_cnt_day')

plt.show()
# # Since there is no data for November or December 2015, just select data for 2013 and 2014 and confirm it
train_2year = lk_train[(lk_train['date'] >= '2013-01-01') & (lk_train['date'] <= '2014-12-31')]

monthly_sum = train_2year.groupby('month')['item_cnt_day'].sum().reset_index()

fig = px.bar(monthly_sum, x='month',y='item_cnt_day',title='Sales by Month', width=900, height=500, color_continuous_midpoint= 'item_cnt_day')

fig.show()
day_sum = lk_train.groupby('dayofweek')['item_cnt_day'].sum().reset_index()

fig = px.bar(day_sum, x='dayofweek',y='item_cnt_day',title='Sales by Day', width=700, height=500 )

fig.show()
category_sum = lk_train.groupby('item_category_id')['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day', ascending=False)

fig = px.bar(category_sum, x='item_category_id',y='item_cnt_day',title='Sales by Category', width=900, height=500)

fig.show()
category_sum = lk_train.groupby(['item_category_id', 'item_category_name'])['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day', ascending=False)[:10]

labels = list(category_sum['item_category_name'])

values = list(category_sum['item_cnt_day'])



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
#shop별 주로 파는 item category 경향성



shop_top_cat = lk_train.groupby(['shop_id', 'item_category_name'])['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day', ascending=False)

shop_top_cat = shop_top_cat.drop_duplicates('shop_id', keep='first')

labels = list(shop_top_cat['item_category_name'])

values = list(shop_top_cat['item_cnt_day'])



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()  
item_sum = lk_train.groupby('item_id')['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day',ascending=False)[:5]

fig = px.bar(item_sum, x='item_id',y='item_cnt_day',title='Sales by Item', width=900, height=500)

fig.show()
item_sum = lk_train.groupby(['item_id', 'item_name'])['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day', ascending=False)[:5]

labels = list(item_sum['item_name'])

values = list(item_sum['item_cnt_day'])



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
print('item_id 30: ', items[items['item_id'] == 20949].values, "Corporate package shirt 1C Interest white")

print('item_id 55: ', items[items['item_id'] == 2808].values, "Diablo III [PC, Jewell, Russian Version]")

print('item_id 10: ', items[items['item_id'] == 3732].values, "Grand Theft Auto V[PS3] Russian Subtitles")
category_sum = lk_train.groupby('shop_id')['item_cnt_day'].sum().reset_index().sort_values('item_cnt_day',ascending=False)

fig = px.bar(category_sum, x='shop_id',y='item_cnt_day',title='Sales by shop', width=900, height=500)

fig.show()
print('shop_id 31: ', shops[shops['shop_id'] == 31].values, "Moscow TRC Semenovsky")

print('shop_id 25: ', shops[shops['shop_id'] == 25].values, "Moscow TRK Atrium")

print('shop_id 28: ', shops[shops['shop_id'] == 28].values, "Moscow Warm Stan II")



# shop name에 city가 포함되어 있음
lk_train.groupby(['month', 'year']).sum()['item_price'].unstack().plot(figsize=(13,5))

plt.xlabel('month')

plt.ylabel('Total item_price')

plt.show()
date_sum = lk_train.groupby(['year','date'])['item_price'].sum().reset_index()

fig = px.line(date_sum, x="date", y="item_price", title='Sales by item_price', width=900, height=500, color='year')

fig.show()
category_sum = lk_train.groupby('item_category_id')['item_price'].sum().reset_index()

fig = px.bar(category_sum, x='item_category_id',y='item_price',title='Sales by Category', width=900, height=500)

fig.show()
category_sum = lk_train.groupby(['item_category_id', 'item_category_name'])['item_price'].sum().reset_index().sort_values('item_price', ascending=False)[:10]

labels = list(category_sum['item_category_name'])

values = list(category_sum['item_price'])



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()  
price = lk_train[['item_category_id','item_price']]

price = price.drop_duplicates(['item_category_id','item_price'])

fig = px.box(price, x="item_category_id", y="item_price", title = 'Price variation by Item', width=2000, height=500)

fig.show()
print(lk_train[lk_train['item_category_id'] == 9]['item_category_name'].unique(), ': delivery the product')

print(lk_train[lk_train['item_category_id'] == 16]['item_category_name'].unique() , ': Game Console-XBXOnE')
price = lk_train[['item_id','item_price']]

price = price.drop_duplicates(['item_id','item_price'])

fig = px.box(price, x="item_id", y="item_price", title = 'Price variation by Item', width=2000, height=500)

fig.show()
print(lk_train[lk_train['item_id'] == 13401]['item_name'].unique(), ': Call of Duty: Advanced Warfare (xbox)')

print(lk_train[lk_train['item_id'] == 11365]['item_name'].unique() , ': delivery EMS')
good_shop= lk_train.groupby(['shop_id', 'shop_name'])['item_price'].median().reset_index().sort_values('item_price')[:10]

fig = px.bar(good_shop, x='shop_id',y='item_price',title='Sales by item_price', width=900, height=500,color_continuous_midpoint= 'item_price')

fig.show()
# 수익 칼럼 만들어주기

lk_train['revenue'] = lk_train['item_price'] * lk_train['item_cnt_day']



best_shop= lk_train.groupby(['shop_id', 'shop_name'])['revenue'].sum().reset_index().sort_values('revenue')[:10]

fig = px.bar(best_shop, x='shop_id',y='revenue',title='Sales by revenue', width=900, height=500)

fig.show()
# 1-1)Add item_cnt_day in November 2013 and 2014 to make average bets 

# 2013년 2014년의 11월 'item_cnt_day' 평균 + 중복값을 first로 남기기

# score: 1.25



only11_2013 = train[(train['year']==2013) & (train['month']==11)][['shop_id', 'item_id', 'item_cnt_day']]

only11_2014 = train[(train['year']==2014) & (train['month']==11)][['shop_id', 'item_id', 'item_cnt_day']] 



only11 = only11_2013.merge(only11_2014, on=['shop_id', 'item_id'], how='left').fillna(0)



only11 = test.merge(only11, on=['shop_id', 'item_id'], how='left').fillna(0) 

only11['item_cnt_month'] = (only11['item_cnt_day_x'] + only11['item_cnt_day_y']) / 2



final_only11 = only11[['ID', 'item_cnt_month']]



subset = ['ID']

final_only11.drop_duplicates(subset=subset, inplace=True, keep='first')



final_only11.sort_values('ID') # 확인

final_only11.to_csv("only11_submission_first.csv", index=False)
# 1-2) Leave duplicate value to last

# 2013년 2014년의 11월 'item_cnt_day' 평균 + 중복값을 last로 남기기

# score: 1.23

subset = ['ID']

final_only11.drop_duplicates(subset=subset, inplace=True, keep='last')



final_only11.sort_values('ID') # 확인

final_only11.to_csv("only11_submission_last.csv", index=False)
# 2) 모든 기간의 총 'item_cnt_day' 평균

# score: 2.0



sales = test.merge(train, how='left', on = ['shop_id', 'item_id'])



subset =['ID', 'shop_id', 'item_id', 'date', 'date_block_num', 'item_price','item_cnt_day']

sales.drop_duplicates(subset=subset, inplace=True, keep='first')

sales = sales.dropna(how='any', thresh=None, subset=None)



samp = sales[['ID', 'date_block_num', 'item_cnt_day']]

samp = samp.sort_values(['ID', 'date_block_num'])

samp = (samp.groupby('ID')['item_cnt_day'].sum() / 34).reset_index()



total_avg = test.merge(samp, how='left', on='ID').fillna(0)

total_avg = total_avg.rename(columns = {'item_cnt_day' : 'item_cnt_month'})

total_avg.to_csv('total_avg_submission.csv', index_label='ID')
#  3) Weighted in 2014 to reflect growth rate

# score: 1.23



a = train[(train['date'] >= '2013-01-01') & (train['date'] <= '2013-10-31')]

b = train[(train['date'] >= '2014-01-01') & (train['date'] <= '2014-10-31')]

c = train[(train['date'] >= '2015-01-01') & (train['date'] <= '2015-10-31')]



sum13 = a.groupby('year')['item_cnt_day'].sum().values

sum14 = b.groupby('year')['item_cnt_day'].sum().values

sum15 = c.groupby('year')['item_cnt_day'].sum().values



grow_2013 = sum15 / sum13 * 100

grow_2014 = sum15 / sum14 * 100



only11_2013 = train[(train['year']==2013) & (train['month']==11)][['shop_id', 'item_id', 'item_cnt_day']]

only11_2014 = train[(train['year']==2014) & (train['month']==11)][['shop_id', 'item_id', 'item_cnt_day']] 



only11 = only11_2013.merge(only11_2014, on=['shop_id', 'item_id'], how='left').fillna(0)



only11 = test.merge(only11, on=['shop_id', 'item_id'], how='left').fillna(0) 

only11['item_cnt_month'] = ((only11['item_cnt_day_x'] * grow_2013) + (only11['item_cnt_day_y'] * grow_2014)) / 2



final_only11 = only11[['ID', 'item_cnt_month']]



subset = ['ID']

final_only11.drop_duplicates(subset=subset, inplace=True, keep='last')



final_only11.sort_values('ID') # 확인

final_only11.to_csv("g_submission.csv", index=False)
def get_outlier(df=None, column=None, weight=1.5):

    fraud=df[column]

    print(fraud.shape)

    q_25 = np.percentile(fraud.values, 25)

    q_75 = np.percentile(fraud.values, 75)

    

    iqr = q_75 - q_25

    iqr_weight = iqr * weight

    lowest_val = q_25 - iqr_weight

    highest_val = q_75 + iqr_weight

    outlier_index = fraud[(fraud<lowest_val) | (fraud<highest_val)].index

    return outlier_index
outlier_index = get_outlier(df=lk_train, column='item_cnt_day')

len(outlier_index)
# 월별 판매량을 만들기 위해 date 칼럼 변경

lk_train['date'] = lk_train['date'].apply(lambda x : x.strftime('%Y-%m'))
outlier_v2 = lk_train.loc[outlier_index,:]

lk_train_v2 = lk_train[lk_train.isin(outlier_v2) == False]
pd.set_option('float_format', '{:f}'.format)

lk_train['item_cnt_day'].describe()
lk_train_v2['item_cnt_day'].describe()
df2 = lk_train_v2.groupby(['date','item_id','shop_id'])['item_cnt_day'].sum().reset_index()

df2 = df2.pivot_table(index=['shop_id','item_id'], columns='date',values='item_cnt_day').reset_index()

df2.head()
df2_test = pd.merge(test, df2, on=['item_id','shop_id'], how='left').fillna(0)

df2_test = df2_test.drop(columns=['ID', 'shop_id', 'item_id'], axis=1)

df2_test.head()
TARGET = '2015-10'



y_train = df2_test[TARGET]

X_train = df2_test.drop(columns = [TARGET], axis=1)



X_train.head()
y_train.head()
X_test = df2_test.drop(labels=['2013-01'], axis=1)

X_test.head()
!pip install lightgbm



from lightgbm import LGBMRegressor
model=LGBMRegressor()
model.fit(X_train, y_train)
X_train.head()
X_test.describe()
X_test.head()
y_pred = model.predict(X_test).clip(0., 20.)

preds = pd.DataFrame(y_pred, columns=['item_cnt_month'])

preds.to_csv('submission_modeling_v4.csv',index_label='ID')
y_pred
X_test['2015-11'] = y_pred
X_test
model.feature_importances_
from sklearn.ensemble import RandomForestRegressor

raf = RandomForestRegressor(n_estimators=1000,

                              n_jobs=2,

                              random_state=42)
raf.fit(X_train, y_train)
y_pred = raf.predict(X_test).clip(0., 20.)
preds = pd.DataFrame(y_pred, columns=['item_cnt_month'])

preds.head()
preds.to_csv('submission_light.csv',index_label='ID')
pd.set_option('float_format', '{:f}'.format)

raf.feature_importances_