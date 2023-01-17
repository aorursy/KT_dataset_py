# import required libraries



import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
train_df = pd.read_csv('../input/train_data.csv',index_col=None)

test_df = pd.read_csv('../input/test_data.csv',index_col=None)



# print some information

print(train_df.info())

print(test_df.info())
# preview the data

train_df.head()
# Count number of NaN values in each column

train_df.isna().sum()
# Average unit price grouped by storey_range

train_df[['storey_range', 'resale_price']].groupby(['storey_range'], as_index=False).mean().sort_values(by='resale_price', ascending=False)
# Average unit price grouped by town

train_df[['town', 'resale_price']].groupby(['town'], as_index=False).mean().sort_values(by='resale_price', ascending=False)
# correlation between remaining_lease, nearest_mrt_distance and floor_area_sqm and retail_price

train_df[['remaining_lease', 'nearest_mrt_distance', 'floor_area_sqm', 'resale_price']].corr()['resale_price']
# remove target column from train data



train_label = train_df['resale_price']

train_df = train_df.drop(columns=['resale_price'])
# combine train and test

train_length = len(train_df)

all_data = pd.concat(objs=[train_df, test_df], axis=0)
def process_month(df):

    df['month_year'] = df.apply(lambda row: row['month'].split('-')[0], axis=1)

    df['month_month'] = df.apply(lambda row: row['month'].split('-')[1], axis=1)



process_month(all_data)

    

# 'month' column now can be removed

all_data = all_data.drop(columns=['month'])



all_id = all_data['id']

all_data = all_data.drop(columns=['id', 'lease_commence_date'])
all_data = pd.get_dummies(all_data)

all_data.head()
min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(all_data)

df = pd.DataFrame(x_scaled)



df.head()
train_data = x_scaled[:train_length]

test_data = x_scaled[train_length:]

train_label = train_label.values
# create model

model = RandomForestRegressor()



# fit model with train data

model.fit(train_data, train_label)
prediction = model.predict(test_data)



test_id = all_id[train_length:].values



result_df = pd.DataFrame({'id': test_id,'resale_price': prediction})

result_df.to_csv('submission.csv',index=False)
