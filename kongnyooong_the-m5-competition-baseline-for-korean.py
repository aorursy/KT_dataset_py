import numpy as np 

import pandas as pd 

import plotnine 

import matplotlib.pyplot as plt

import seaborn as sns

import os



from itertools import cycle

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])



%matplotlib inline
df_train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

df_sell = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

df_calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

sub = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
print("Unit sales of all products, aggregated for each state", df_train['state_id'].nunique())

print("Unit sales of all products, aggregated for each store", df_train['store_id'].nunique())

print("Unit sales of all products, aggregated for each category", df_train['cat_id'].nunique())

print("Unit sales of all products, aggregated for each department", df_train['dept_id'].nunique())

print("Unit sales of all products, aggregated for each State and category", df_train['state_id'].nunique() * df_train['cat_id'].nunique())

print("Unit sales of all products, aggregated for each State and department", df_train['state_id'].nunique() * df_train['dept_id'].nunique())

print("Unit sales of all products, aggregated for each store and category", df_train['store_id'].nunique() * df_train['cat_id'].nunique())

print("Unit sales of all products, aggregated for each store and department", df_train['store_id'].nunique() * df_train['dept_id'].nunique())

print("Unit sales of all products, aggregated for each  and category", df_train['dept_id'].nunique() * df_train['cat_id'].nunique())

print("Unit sales of product x, aggregated for all stores/states", df_train['item_id'].nunique())

print("Unit sales of product x, aggregated for all states", df_train['item_id'].nunique() * df_train['state_id'].nunique())

print("Unit sales of product x, aggregated for all stores", df_train['item_id'].nunique() * df_train['store_id'].nunique())
df_calendar.head(8)
df_calendar[df_calendar['event_name_1'].notnull()].head()
f, ax = plt.subplots(figsize = (16, 12))

ax.grid(axis='x', linestyle='--')



sns.countplot(y = "event_name_1", data = df_calendar, ax = ax, palette = "Greens_d",edgecolor='black', linewidth=0.8)

plt.title("Count of Event Name 1", size = 20)

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.xlabel("Count", size = 18)

plt.ylabel("Event Name", size = 18)
f, ax = plt.subplots(figsize = (14, 8))

ax.grid(axis='x', linestyle='--')



sns.countplot(y = "event_type_1", data = df_calendar, ax = ax, palette = "Greens_d",edgecolor='black', linewidth=0.8)

plt.title("Count of Event Type 1", size = 20)

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.xlabel("Count", size = 18)

plt.ylabel("Event Type", size = 18)
df_calendar[df_calendar['event_name_2'].notnull()].head()
print("event_name_2 notnull shape : ", df_calendar[df_calendar['event_name_2'].notnull()].shape)

print("event_name_1 and 2 notnull shape : ", df_calendar[(df_calendar['event_name_2'].notnull()) & (df_calendar['event_name_1'].notnull())].shape)
df_calendar.loc[(df_calendar['event_name_2'].notnull()) & (df_calendar['event_name_1'].notnull())]
df_calendar.loc[df_calendar['event_name_2'].notnull()]
f, ax = plt.subplots(figsize = (16, 12))

ax.grid(axis='x', linestyle='--')



sns.countplot(y = "event_name_2", data = df_calendar, ax = ax, palette = "Greens_d",edgecolor='black', linewidth=0.8)

plt.title("Count of Event Name 2", size = 20)

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.xlabel("Count", size = 18)

plt.ylabel("Event Name", size = 18)
f, ax = plt.subplots(figsize = (14, 8))

ax.grid(axis='x', linestyle='--')



sns.countplot(y = "event_type_2", data = df_calendar, ax = ax, palette = "Greens_d",edgecolor='black', linewidth=0.8)

plt.title("Count of Event Type 2", size = 20)

plt.xticks(size = 15)

plt.yticks(size = 15)

plt.xlabel("Count", size = 18)

plt.ylabel("Event Type", size = 18)
print(df_sell.shape)

df_sell.head()
df_sell["sell_price"].isnull().sum()
df_train["cat_id"].unique()
df_train["dept_id"].unique()
print(df_train.shape)

df_train.head()
sub["id"].unique
sub.head()
d_cols = [c for c in df_train.columns if 'd_' in c] 



# d_로 시작하는 columns
len(df_train["id"].unique())
df_item = df_train.loc[df_train['id'] == 'FOODS_3_090_CA_3_validation'][d_cols].T # 시계열 시각화를 위해 "d_" 변수들과 하나의 제품 변수를 골라서 transpose해준다. 

df_item = df_item.rename(columns={8412:'FOODS_3_090_CA_3'}) # 인덱싱 했던 행의 번호로 되어있는 column name을 제품의 id로 바꿔준다.

df_item = df_item.reset_index().rename(columns={'index': 'd'}) # 인덱스 이름을 "d"로 바꿔준다.

df_item = df_item.merge(df_calendar, how='left', validate='1:1') # 위에서 만들어준 데이터프레임과 calendar 데이터프레임을 병합한다. (d로 만들어준 컬럼과 date 컬럼을 사용하기 위함)

df_item.set_index('date')['FOODS_3_090_CA_3'].plot(figsize=(15, 5),

                                                   color=next(color_cycle))



plt.title('FOODS_3_090_CA_3 sales by actual sale dates', size = 20)

plt.xticks(size = 13)

plt.yticks(size = 13)

plt.xlabel("Date", size = 15)

plt.ylabel("Sales", size = 15)

plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))



df_item.groupby('wday').mean()['FOODS_3_090_CA_3'].plot(kind='line', title='average sale: day of week', color=next(color_cycle), ax=ax1)

# 요일에 따른 판매량 확인



df_item.groupby('month').mean()['FOODS_3_090_CA_3'].plot(kind='line', title='average sale: month', color=next(color_cycle), ax=ax2)

# 월별 판매량 확인



df_item.groupby('year').mean()['FOODS_3_090_CA_3'].plot(kind='line', title='average sale: year', color=next(color_cycle), ax=ax3)

# 연도별 판매량 확인



fig.suptitle('Trends for item: FOODS_3_090_CA_3', size=20, y=1.1)

plt.tight_layout()

plt.show()
item_id_split = df_sell['item_id'].str.split('_', expand=True)

item_id_split[0].unique()
past_sales = df_train.set_index('id')[d_cols].T.merge(df_calendar.set_index('d')['date'], left_index=True, right_index=True, validate='1:1').set_index('date')



item_type_list = item_id_split[0].unique()



for i in item_type_list:

    items_col = [c for c in past_sales.columns if i in c]

    past_sales[items_col].sum(axis=1).plot(figsize=(15, 5), alpha=0.8)



plt.title('Total Sales by Item Type', size = 20)

plt.xticks(size = 13)

plt.yticks(size = 13)

plt.xlabel("Date", size = 15)

plt.ylabel("Sales", size = 15)

plt.legend(item_type_list)

plt.show()



# 식료품의 판매량이 압도적으로 높고, 그 다음이 가정용품 그리고 취미용품 순서인 것을 알 수 있다.
store_list = df_sell['store_id'].unique()



for s in store_list:

    store_items = [c for c in past_sales.columns if s in c]

    past_sales[store_items].sum(axis=1).rolling(90).mean().plot(figsize=(18, 6), alpha=0.8) # .rolling은 이동평균을 위한 함수



    

plt.title('Rolling 90 Day Average Total Sales (10 stores)', size = 20)

plt.xticks(size = 13)

plt.yticks(size = 13)

plt.xlabel("Date", size = 15)

plt.ylabel("Sales", size = 15)    

plt.legend(store_list)

plt.show()
days = range(1, 1913 + 1)

time_series_columns = [f'd_{i}' for i in days]

time_series_data = df_train[time_series_columns]
MA_x = 34  #play here



forecast = time_series_data.iloc[:, -MA_x:].copy()

for i in range(28):

    forecast['F'+str(i+1)] = forecast.iloc[:, -MA_x:].mean(axis=1)    

    

forecast = forecast[['F'+str(i+1) for i in range(28)]]

forecast.head(20)
validation_ids = df_train['id'].values

evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]

ids = np.concatenate([validation_ids, evaluation_ids])

predictions = pd.DataFrame(ids, columns=['id'])

forecast = pd.concat([forecast] * 2).reset_index(drop=True)

predictions = pd.concat([predictions, forecast], axis=1)

predictions.to_csv('submission.csv', index=False)
predictions.head()