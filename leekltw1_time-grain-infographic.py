import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
%matplotlib inline
train = pd.read_csv('../input/train.csv')
def generate_date_feature(data):
    data['day_of_month'] = pd.to_datetime(data['date']).dt.day
    data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
    data['day_of_year'] = pd.to_datetime(data['date']).dt.dayofyear
    data['week_of_year'] = pd.to_datetime(data['date']).dt.weekofyear
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['year'] = pd.to_datetime(data['date']).dt.year
    data.drop('date',axis=1,inplace=True)
    return data
    
date_features = ['day_of_month','day_of_week','day_of_year','week_of_year','month','year']

train = generate_date_feature(train)

def different_store_item_sales_relation_with_date_feature(store_or_item,date_feature,legend_fontsize=10):
    for i in train[store_or_item].unique():
        if store_or_item == 'store':
            plt.plot(train[train[store_or_item]==i].groupby(by=date_feature).mean()['sales'],'-',label=f'store{i}')
        elif store_or_item == 'item':
            plt.plot(train[train[store_or_item]==i].groupby(by=date_feature).mean()['sales'],'-',label=f'item{i}')
    plt.plot(train.groupby(by=date_feature).mean()['sales'],'k-',linewidth=3,label='total mean')
#     plt.title(f'{store_or_item} sales relationship with {date_feature}')
    plt.xlabel(f'{date_feature}')
    plt.ylabel('sales')
    plt.legend(loc='upper left',fontsize=legend_fontsize)
plt.figure(figsize=(20,10))
for number, date_feature in enumerate(date_features):
    plt.subplot(2,3,number+1)
    different_store_item_sales_relation_with_date_feature('store',date_feature)
plt.figure(figsize=(20,20))
for number, date_feature in enumerate(date_features):
    plt.subplot(2,3,number+1)
    different_store_item_sales_relation_with_date_feature('item',date_feature,legend_fontsize=6)
for date_feature in date_features:
    std_sales = train.groupby([date_feature]).std()['sales']
    print('The maximum and minimum sales of standard deviation of ',date_feature,' are ',str(max(std_sales)),', ',min(std_sales))
linear_regression = LinearRegression(n_jobs=-1)
linear_regression.fit(train['year'].values.reshape(-1, 1),train['sales'].values)
linear_regression_pred = linear_regression.predict(train['year'].values.reshape(-1, 1))
plt.figure(figsize=(20,5))
plt.title('Linear regression prediction')
plt.plot(train.groupby(by='year').mean()['sales'],'g',label='training data')
plt.plot(train['year'].values,linear_regression_pred,'b',label='prediction')
plt.xlabel('year')
plt.ylabel('sales')
plt.xlim(min(train['year'].values),max(train['year'].values))
plt.legend(loc='upper left')