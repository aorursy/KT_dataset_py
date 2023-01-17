import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error
train_data = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

revenue_data = pd.read_csv('../input/imdb-revenue-dataset/train_revenue.csv')

train_data = pd.merge(train_data, revenue_data, how='left', on=['imdb_id'])
zero_usa_revenue_percent = len(train_data[train_data.usa_revenue==0])/len(train_data)*100

zero_world_revenue_percent = len(train_data[train_data.world_revenue==0])/len(train_data)*100

print("{0:.2f} percent films hasn't USA revenue information".format(zero_usa_revenue_percent))

print("{0:.2f} percent films hasn't World revenue information".format(zero_world_revenue_percent))

train_data = train_data[(train_data.usa_revenue > 0) & (train_data.world_revenue > 0)]

print("full revenue information films count: {}".format(len(train_data)))
train_data['is_usa_revenue'] = abs(train_data.revenue - train_data.usa_revenue) < train_data[['revenue', 'usa_revenue']].max(axis=1)*0.05

train_data['is_ww_revenue'] = abs(train_data.revenue - train_data.world_revenue) < train_data[['revenue', 'world_revenue']].max(axis=1)*0.05
usa_vc = train_data['is_usa_revenue'].value_counts()

ww_vc = train_data['is_ww_revenue'].value_counts()

pd.DataFrame({'true':[usa_vc.get(True), ww_vc.get(True)],

             'false':[ww_vc.get(False), ww_vc.get(False)]},

             index = ['USA revenue','World Revenue']).plot.bar()

train_data['is_only_usa_revenue'] = train_data.is_usa_revenue & ~train_data.is_ww_revenue

train_data['is_only_ww_revenue'] = train_data.is_ww_revenue & ~train_data.is_usa_revenue

train_data['both_revenues'] = train_data.is_usa_revenue & train_data.is_ww_revenue

train_data['strange_revenue'] = ~train_data.is_usa_revenue & ~train_data.is_ww_revenue
train_data[['is_only_usa_revenue','is_only_ww_revenue','both_revenues','strange_revenue']].sum().plot.bar()
train_data['usa_revenue_diff'] = abs(train_data.revenue - train_data.world_revenue)/train_data.revenue

top_error_movies = train_data[train_data.is_only_usa_revenue & (train_data.usa_revenue_diff>0.1)].sort_values(by='usa_revenue_diff',ascending=False)

print(top_error_movies.usa_revenue_diff.describe())
top_error_movies.to_csv('top_error_films.csv',index=False)

top_error_movies[['original_title','imdb_id','revenue','usa_revenue','world_revenue','usa_revenue_diff']].head(20)