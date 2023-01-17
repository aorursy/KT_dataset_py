# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

# Number of missing values in each column of training data

missing_val_count_by_column = (df.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0], len(df))
cols_with_missing = ['agent', 'company', 'country']



# Drop columns in training and validation data

reduced_df = df.drop(cols_with_missing, axis=1)

reduced_df.describe()
d = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

reduced_df.arrival_date_month = reduced_df.arrival_date_month.map(d)
assigned_df = reduced_df.assign(arrival_date = reduced_df['arrival_date_year'].map(str) + '-' + reduced_df['arrival_date_month'].map(str) + '-' + reduced_df['arrival_date_day_of_month'].map(str)

                 ,length_of_stay = reduced_df.stays_in_weekend_nights+reduced_df.stays_in_week_nights )
assigned_df['reservation_status_date'] = pd.to_datetime(assigned_df['reservation_status_date'])

assigned_df['arrival_date'] = pd.to_datetime(assigned_df['arrival_date'])

temp = assigned_df['length_of_stay'].apply(np.ceil).apply(lambda x: pd.Timedelta(x, unit='D'))

assigned_df['leaving_date'] = assigned_df['arrival_date'] + temp

# dropping ALL duplicte values 

assigned_df.drop_duplicates(subset =['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',

       'arrival_date_month', 'arrival_date_week_number',

       'arrival_date_day_of_month', 'stays_in_weekend_nights',

       'stays_in_week_nights', 'babies', 'meal', 'market_segment',

       'distribution_channel', 'is_repeated_guest', 'previous_cancellations',

       'previous_bookings_not_canceled', 'reserved_room_type',

       'assigned_room_type', 'booking_changes', 'deposit_type',

       'days_in_waiting_list', 'customer_type', 'adr',

       'required_car_parking_spaces', 'total_of_special_requests',

       'reservation_status', 'reservation_status_date', 'arrival_date', 'length_of_stay', 'leaving_date'], 

                     keep = False, inplace = True) 

assigned_df['cus_id'] = pd.factorize(assigned_df.apply(tuple, axis=1))[0] + 1

assigned_df.shape
stayed_dt = pd.concat([pd.Series(r.cus_id, pd.date_range(r.arrival_date, r.leaving_date, freq='D'))

                      for r in assigned_df.itertuples()]).reset_index()

stayed_dt.columns = ['stay_date', 'cus_id']

print(stayed_dt)
stayed_df = pd.merge(stayed_dt, assigned_df[['hotel', 'cus_id', 'assigned_room_type', 'is_canceled']], left_on=['cus_id'], right_on=['cus_id'], how='left')

stayed_df.tail(10)
city_max = stayed_df[stayed_df['is_canceled']==0][stayed_df['hotel']=='City Hotel'].groupby(['stay_date','hotel','assigned_room_type']).count().max(level=2).sum()['cus_id']

resort_max = stayed_df[stayed_df['is_canceled']==0][stayed_df['hotel']=='Resort Hotel'].groupby(['stay_date','hotel','assigned_room_type']).count().max(level=2).sum()['cus_id']

city_max_by_day = stayed_df[stayed_df['is_canceled']==0][stayed_df['hotel']=='City Hotel'].groupby(['stay_date']).count().max()['cus_id']

resort_max_by_day = stayed_df[stayed_df['is_canceled']==0][stayed_df['hotel']=='Resort Hotel'].groupby(['stay_date']).count().max()['cus_id']

print("City Hotel's number of rooms by room type", stayed_df[stayed_df['is_canceled']==0][stayed_df['hotel']=='City Hotel'].groupby(['stay_date','hotel','assigned_room_type']).count().max(level=2)['cus_id'])

print("City Hotel's total number of rooms", city_max)

print("Resort Hotel's number of rooms by room type",stayed_df[stayed_df['is_canceled']==0][stayed_df['hotel']=='Resort Hotel'].groupby(['stay_date','hotel','assigned_room_type']).count().max(level=2)['cus_id'])

print("Resort Hotel's total number of rooms", resort_max)

print("City Hotel's maximum number of rooms without consideration of room type : ", city_max_by_day)

print("Resort Hotel's maximum number of rooms without consideration of room type : ",

      resort_max_by_day)
import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()

fig, ax = plt.subplots(figsize=(15,7))

stayed_df.groupby(['stay_date','hotel']).count()['cus_id'].unstack().plot(ax=ax)
fig, ax = plt.subplots(2, figsize=(15,7))

stayed_df[stayed_df['hotel']=='City Hotel'].groupby(['stay_date','is_canceled']).count()['cus_id'].unstack().plot(ax=ax[0])

ax[0].set_title('City Hotel Cancellation Y/N')

stayed_df[stayed_df['hotel']=='Resort Hotel'].groupby(['stay_date','is_canceled']).count()['cus_id'].unstack().plot(ax=ax[1])

ax[1].set_title('Resort Hotel Cancellation Y/N')
from fbprophet import Prophet



city_booking = pd.DataFrame(stayed_df[stayed_df['hotel']=='City Hotel'].groupby(['stay_date']).count()['is_canceled'].reset_index())

city_booking.columns = ['ds', 'y']



resort_booking = pd.DataFrame(stayed_df[stayed_df['hotel']=='Resort Hotel'].groupby(['stay_date']).count()['is_canceled'].reset_index())

resort_booking.columns = ['ds', 'y']
c = Prophet()

c.fit(city_booking)

future_c = c.make_future_dataframe(periods=365, freq='D')

forecast_c = c.predict(future_c)

forecast_c
r = Prophet()

r.fit(resort_booking)

future_r = r.make_future_dataframe(periods=365, freq='D')

forecast_r = r.predict(future_r)

forecast_r
import plotly.offline as py

import plotly.graph_objs as go

py.iplot([

    go.Scatter(x=city_booking['ds'], y=city_booking['y'], name='y'),

    go.Scatter(x=forecast_c['ds'], y=forecast_c['yhat'], name='yhat'),

    go.Scatter(x=forecast_c['ds'], y=forecast_c['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast_c['ds'], y=forecast_c['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=forecast_c['ds'], y=forecast_c['trend'], name='Trend')

])



py.iplot([

    go.Scatter(x=resort_booking['ds'], y=resort_booking['y'], name='y'),

    go.Scatter(x=forecast_r['ds'], y=forecast_r['yhat'], name='yhat'),

    go.Scatter(x=forecast_r['ds'], y=forecast_r['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast_r['ds'], y=forecast_r['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=forecast_r['ds'], y=forecast_r['trend'], name='Trend')

])

print(len(city_booking), len(resort_booking))
# Calculate root mean squared error.

print('RMSE for City: %f' % np.sqrt(np.mean((forecast_c.loc[:804, 'yhat']-city_booking['y'])**2)) )

print('RMSE for Resort: %f' % np.sqrt(np.mean((forecast_r.loc[:807, 'yhat']-city_booking['y'])**2)) )
city_booking_c = pd.DataFrame(stayed_df[stayed_df['hotel']=='City Hotel'][stayed_df['is_canceled']==0].groupby(['stay_date']).count()['cus_id'].reset_index())

city_booking_c.columns = ['ds', 'y']



resort_booking_c = pd.DataFrame(stayed_df[stayed_df['hotel']=='Resort Hotel'][stayed_df['is_canceled']==0].groupby(['stay_date']).count()['cus_id'].reset_index())

resort_booking_c.columns = ['ds', 'y']
c_c = Prophet()

c_c.fit(city_booking_c)

future_city_cc = c_c.make_future_dataframe(periods=365, freq='D')

forecast_city_cc = c_c.predict(future_city_cc)

forecast_city_cc
r_c = Prophet()

r_c.fit(resort_booking_c)

future_resort_cc = r_c.make_future_dataframe(periods=365, freq='D')

forecast_resort_cc = r_c.predict(future_resort_cc)

forecast_resort_cc
py.iplot([

    go.Scatter(x=city_booking_c['ds'], y=city_booking_c['y'], name='y'),

    go.Scatter(x=forecast_city_cc['ds'], y=forecast_city_cc['yhat'], name='yhat'),

    go.Scatter(x=forecast_city_cc['ds'], y=forecast_city_cc['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast_city_cc['ds'], y=forecast_city_cc['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=forecast_city_cc['ds'], y=forecast_city_cc['trend'], name='Trend')

])



py.iplot([

    go.Scatter(x=resort_booking_c['ds'], y=resort_booking_c['y'], name='y'),

    go.Scatter(x=forecast_resort_cc['ds'], y=forecast_resort_cc['yhat'], name='yhat'),

    go.Scatter(x=forecast_resort_cc['ds'], y=forecast_resort_cc['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast_resort_cc['ds'], y=forecast_resort_cc['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=forecast_resort_cc['ds'], y=forecast_resort_cc['trend'], name='Trend')

])

print(len(city_booking_c), len(resort_booking_c))
print('RMSE for City: %f' % np.sqrt(np.mean((forecast_city_cc.loc[:800, 'yhat']-city_booking_c['y'])**2)) )

print('RMSE for Resort: %f' % np.sqrt(np.mean((forecast_resort_cc.loc[:807, 'yhat']-city_booking_c['y'])**2)) )
difference_city = pd.merge(forecast_c, forecast_city_cc, left_on=['ds'], right_on=['ds'], how='left')

difference_resort = pd.merge(forecast_r, forecast_resort_cc, left_on=['ds'], right_on=['ds'], how='left')

city_exp = difference_city[(difference_city['ds']>'2017-09-12') & (difference_city['ds']<='2018-09-11')]

resort_exp = difference_resort[(difference_resort['ds']>'2017-09-12') & (difference_city['ds']<='2018-09-11')]

city_exp = city_exp.assign(exp = city_exp.yhat_x - city_exp.yhat_y

                 ,min_exp = city_exp.yhat_lower_x - city_exp.yhat_upper_y

                 ,max_exp = city_exp.yhat_upper_x - city_exp.yhat_lower_y)

city_exp['last_min_booking_yn1']=city_exp['max_exp'].apply(lambda x: 0 if x < city_max else 1)

city_exp['last_min_booking_yn2']=city_exp['max_exp'].apply(lambda x: 0 if x < city_max_by_day else 1)

resort_exp = resort_exp.assign(exp = resort_exp.yhat_x - resort_exp.yhat_y

                 ,min_exp = resort_exp.yhat_lower_x - resort_exp.yhat_upper_y

                 ,max_exp = resort_exp.yhat_upper_x - resort_exp.yhat_lower_y)

resort_exp['last_min_booking_yn1']=city_exp['max_exp'].apply(lambda x: 0 if x < resort_max else 1)

resort_exp['last_min_booking_yn2']=city_exp['max_exp'].apply(lambda x: 0 if x < resort_max_by_day else 1)
city_exp.head(100)
py.iplot([

    go.Scatter(x=city_exp['ds'], y=city_exp['exp'], name='yhat'),

    go.Scatter(x=city_exp['ds'], y=city_exp['min_exp'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=city_exp['ds'], y=city_exp['max_exp'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=city_exp['ds'],   y=[city_max]*len(city_exp), mode='lines', name = 'total no. of rooms (room type-considered)'),

    go.Scatter(x=city_exp['ds'],   y=[city_max_by_day]*len(city_exp), mode='lines', name = 'total no. of rooms (room type-not considered)')

])



py.iplot([

    go.Scatter(x=resort_exp['ds'], y=resort_exp['exp'], name='yhat'),

    go.Scatter(x=resort_exp['ds'], y=resort_exp['min_exp'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=resort_exp['ds'], y=resort_exp['max_exp'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=resort_exp['ds'],   y=[resort_max]*len(resort_exp), mode='lines', name = 'total no. of rooms (room type-considered)'),

    go.Scatter(x=resort_exp['ds'],   y=[resort_max_by_day]*len(resort_exp), mode='lines', name = 'total no. of rooms (room type-not considered)')

])

final_result = pd.merge(city_exp[['ds', 'last_min_booking_yn2']], resort_exp[['ds', 'last_min_booking_yn2']], left_on=['ds'], right_on=['ds'], how='outer')

final_result.columns =['date','City Hotel', 'Resort Hotel']

# 0: Available for booking at the last minute 1: Not available for booking at the last minute

final_result[final_result['Resort Hotel']==1]