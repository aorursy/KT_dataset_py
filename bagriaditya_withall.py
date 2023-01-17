import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time
train1 = pd.read_csv("../input/train.csv", usecols=['date_time','srch_ci','srch_co','srch_destination_id','is_booking','srch_children_cnt','srch_adults_cnt','srch_destination_type_id','hotel_cluster','user_location_city','orig_destination_distance', 'hotel_country'], nrows=10000000)
test = pd.read_csv('../input/test.csv', dtype={'srch_destination_id':np.int32}, usecols=['srch_destination_id'])
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["srch_co"] = pd.to_datetime(train1["srch_co"], format='%Y-%m-%d', errors="coerce")
train1["stay_span"] = (train1["srch_co"] - train1["srch_ci"]).astype('timedelta64[D]')
train1 = train1.drop('srch_co', axis=1)
train1["date_time"] = pd.to_datetime(train1["date_time"], format='%Y-%m-%d', errors="coerce")
train1["search_span"] = (train1["srch_ci"] - train1["date_time"]).astype('timedelta64[D]')
train1 = train1.drop('srch_ci', axis=1)
train1['year'] = train1['date_time'].dt.year
train1['month'] = train1['date_time'].dt.month
train1['day_of_week'] = train1['date_time'].dt.dayofweek
train1['hour'] = train1['date_time'].dt.hour
train1 = train1.drop('date_time', axis=1)
train1.ix[(train1['hour'] >= 10) & (train1['hour'] < 18), 'hour'] = 1
train1.ix[(train1['hour'] >= 18) & (train1['hour'] < 22), 'hour'] = 2
train1.ix[(train1['hour'] >= 22) & (train1['hour'] == 24), 'hour'] = 3
train1.ix[(train1['hour'] >= 1) & (train1['hour'] < 10), 'hour'] = 3
train1['Individuals'] = train1['srch_adults_cnt']+train1['srch_children_cnt']
train1 = train1.drop('srch_adults_cnt', axis=1)
train1 = train1.drop('srch_children_cnt', axis=1)
train1.info()
train1 = train1.drop('search_span', axis=1)
train1 = train1.drop('user_location_city', axis=1)
train1 = train1.drop('hotel_country', axis=1)
train1.info()
train1 = train1[['orig_destination_distance','srch_destination_id','srch_destination_type_id','is_booking','hotel_cluster','stay_span','year','month','day_of_week','hour', 'Individuals']]
train1.info()
train1 = train1.groupby(['srch_destination_id','srch_destination_type_id','hotel_cluster','day_of_week','hour'])['is_booking'].agg(['sum','count'])
train1.reset_index(inplace=True)
CLICK_WEIGHT = 0.05
train1 = train1.groupby(['srch_destination_id','srch_destination_type_id','hotel_cluster','day_of_week','hour']).sum().reset_index()
train1['count'] -= train1['sum']
train1 = train1.rename(columns={'sum':'bookings','count':'clicks'})
train1['relevance'] = train1['bookings'] + CLICK_WEIGHT * train1['clicks']
train1.head()
def most_popular(group, n_max=5):
    relevance = group['relevance'].values
    hotel_cluster = group['hotel_cluster'].values
    most_popular = hotel_cluster[np.argsort(relevance)[::-1]][:n_max]
    return np.array_str(most_popular)[1:-1]
most_pop = train1.groupby(['srch_destination_id']).apply(most_popular)
most_pop = pd.DataFrame(most_pop).rename(columns={0:'hotel_cluster'})
most_pop.head()
test = test.merge(most_pop, how='left',left_on='srch_destination_id',right_index=True)
test.head()
test.hotel_cluster.isnull().sum()
most_pop_all = train1.groupby('hotel_cluster')['relevance'].sum().nlargest(5).index
most_pop_all = np.array_str(most_pop_all)[1:-1]
most_pop_all
test.hotel_cluster.fillna(most_pop_all,inplace=True)
test.hotel_cluster.to_csv('predicted_with_pandas.csv',header=True, index_label='id')