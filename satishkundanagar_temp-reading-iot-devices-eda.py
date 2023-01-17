import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go
path = '/kaggle/input/temperature-readings-iot-devices/'

temp_iot_data = pd.read_csv(path + 'IOT-temp.csv')
temp_iot_data.shape
temp_iot_data.info()
temp_iot_data.head()
temp_iot_data.rename(columns = {'room_id/id':'room_id', 'out/in':'out_in'}, inplace = True)
temp_iot_data.head()
def get_df_summary(df):

    

    '''This function is used to summarise especially unique value count and data type for variable'''

    

    unq_val_cnt_df = pd.DataFrame(df.nunique(), columns = ['unq_val_cnt'])

    unq_val_cnt_df.reset_index(inplace = True)

    unq_val_cnt_df.rename(columns = {'index':'variable'}, inplace = True)

    unq_val_cnt_df = unq_val_cnt_df.merge(df.dtypes.reset_index().rename(columns = {'index':'variable', 0:'dtype'}),

                                          on = 'variable')

    unq_val_cnt_df = unq_val_cnt_df.sort_values(by = 'unq_val_cnt', ascending = False)

    

    return unq_val_cnt_df
unq_val_cnt_df = get_df_summary(temp_iot_data)
unq_val_cnt_df
temp_iot_data.drop(columns = 'room_id', inplace = True)
print('No. of duplicate records in the data set : {}'.format(temp_iot_data.duplicated().sum()))
# Check for duplicate records.



temp_iot_data[temp_iot_data.duplicated()]
temp_iot_data.loc[temp_iot_data['id'] == '__export__.temp_log_196108_4a983c7e']
# Drop duplicate records.



temp_iot_data = temp_iot_data.drop_duplicates()
# Convert noted_date into date-time.



temp_iot_data['noted_date'] = pd.to_datetime(temp_iot_data['noted_date'], format = '%d-%m-%Y %H:%M')
# Check data duplicacy based on noted_date variable.



temp_iot_data.groupby(['noted_date'])['noted_date'].count().sort_values(ascending = False).head()
temp_iot_data.loc[temp_iot_data['noted_date'] == pd.to_datetime('2018-09-12 03:09:00', format = '%Y-%m-%d %H:%M:%S'), ].sort_values(by = 'id').head(10)
# Check if last but one bit of "id" can be used as primary key.



temp_iot_data['id'].apply(lambda x : x.split('_')[6]).nunique() == temp_iot_data.shape[0]
# Create a new column to store last but one bit of id value.



temp_iot_data['id_num'] = temp_iot_data['id'].apply(lambda x : int(x.split('_')[6]))
temp_iot_data.loc[temp_iot_data['noted_date'] == pd.to_datetime('2018-09-12 03:09:00', format = '%Y-%m-%d %H:%M:%S'), ].sort_values(by = 'id_num').head(10)
temp_iot_data.loc[temp_iot_data['id_num'].isin(range(17003, 17007))].sort_values(by = 'id_num')
temp_iot_data.loc[temp_iot_data['id_num'].isin(range(17006, 17010))].sort_values(by = 'id_num')
temp_iot_data.loc[temp_iot_data['noted_date'] == pd.to_datetime('2018-09-09 16:24:00', format = '%Y-%m-%d %H:%M:%S'), ].sort_values(by = 'id_num').head(10)
temp_iot_data.loc[temp_iot_data['id_num'].isin(range(4000, 4003))].sort_values(by = 'id_num')
temp_iot_data.loc[temp_iot_data['id_num'].isin(range(4002, 4007))].sort_values(by = 'id_num')
temp_iot_data.loc[:, 'id'] = temp_iot_data.loc[:, 'id_num']
# Drop id_num column from the data set.



temp_iot_data.drop(columns = 'id_num', inplace = True)
print('No. of years data : {}'.format(temp_iot_data['noted_date'].dt.year.nunique()))
print('No. of months data : {}'.format(temp_iot_data['noted_date'].dt.month.nunique()))
sorted(temp_iot_data['noted_date'].dt.month.unique())
print('No. of days data : {}'.format(temp_iot_data['noted_date'].dt.day.nunique()))
temp_iot_data['month'] = temp_iot_data['noted_date'].apply(lambda x : int(x.month))
# temp_iot_data['month'].unique()
temp_iot_data['day'] = temp_iot_data['noted_date'].apply(lambda x : int(x.day))
# print(sorted(temp_iot_data['day'].unique()))
temp_iot_data['day_name'] = temp_iot_data['noted_date'].apply(lambda x : x.day_name())
# print(temp_iot_data['day_name'].unique())
temp_iot_data['hour'] = temp_iot_data['noted_date'].apply(lambda x : int(x.hour))
print(sorted(temp_iot_data['hour'].unique()))
temp_iot_data.head()
def map_month_to_seasons(month_val):

    if month_val in [12, 1, 2]:

        season_val = 'Winter'

    elif month_val in [3, 4, 5]:

        season_val = 'Summer'

    elif month_val in [6, 7, 8, 9]:

        season_val = 'Monsoon'

    elif month_val in [10, 11]:

        season_val = 'Post_Monsoon'

    

    return season_val
temp_iot_data['season'] = temp_iot_data['month'].apply(lambda x : map_month_to_seasons(x))
temp_iot_data['season'].value_counts(dropna = False)
temp_iot_data.head()
temp_iot_data['month_name'] = temp_iot_data['noted_date'].apply(lambda x : x.month_name())
# temp_iot_data['month_name'].value_counts(dropna = False)
def bin_hours_into_timing(hour_val):

    

    if hour_val in [22,23,0,1,2,3]:

        timing_val = 'Night (2200-0359 Hours)'

    elif hour_val in range(4, 12):

        timing_val = 'Morning (0400-1159 Hours)'

    elif hour_val in range(12, 17):

        timing_val = 'Afternoon (1200-1659 Hours)'

    elif hour_val in range(17, 22):

        timing_val = 'Evening (1700-2159 Hours)'

    else:

        timing_val = 'X'

        

    return timing_val
temp_iot_data['timing'] = temp_iot_data['hour'].apply(lambda x : bin_hours_into_timing(x))
temp_iot_data['timing'].value_counts(dropna = False)
del unq_val_cnt_df
fig = px.box(temp_iot_data, x = 'out_in', y = 'temp', labels = {'out_in':'Outside/Inside', 'temp':'Temperature'})

fig.update_xaxes(title_text = 'In or Out')

fig.update_yaxes(title_text = 'Temperature (in degree celsius)')

fig.update_layout(title = 'Overall Temp. Variation Inside-Outside Room')

fig.show()
fig = px.box(temp_iot_data, 

             x = 'season', 

             y = 'temp', 

             color = 'out_in', 

             labels = {'out_in':'Outside/Inside', 'temp':'Temperature', 'season':'Season'})

fig.update_xaxes(title_text = 'Inside/Outside - Season')

fig.update_yaxes(title_text = 'Temperature (in degree celsius)')

fig.update_layout(title = 'Season-wise Temp. Variation')

fig.show()
fig = px.box(temp_iot_data, x = 'month_name', y = 'temp', 

             category_orders = {'month_name':['July', 'August', 'September', 'October', 'November', 'December']},

             color = 'out_in')

fig.update_xaxes(title_text = 'Inside/Outside Month')

fig.update_yaxes(title_text = 'Temperature (in degree celsius)')

fig.update_layout(title = 'Monthly Temp. Variation')

fig.show()
round(temp_iot_data['month_name'].value_counts(dropna = False) * 100 / temp_iot_data.shape[0],1)
temp_iot_data.head()
for in_out_val in ['In', 'Out']:



    fig = px.box(temp_iot_data.loc[temp_iot_data['out_in'] == in_out_val], x = 'month_name', y = 'temp', 

                 category_orders = {'month_name':['July', 'August', 'September', 'October', 'November', 'December'], 

                                    'timing':['Morning (0400-1159 Hours)', 'Afternoon (1200-1659 Hours)', 'Evening (1700-2159 Hours)', 'Night (2200-0359 Hours)']},

                 hover_data = ['hour'],

                 labels = {'timing':'Timing', 'hour':'Hour', 'month_name':'Month', 'temp':'Temperature'},

                 color = 'timing')

    fig.update_xaxes(title_text = 'Month-Day Timings')

    fig.update_yaxes(title_text = 'Temperature (in degree celsius)')

    fig.update_layout(title = 'Temperature Variation in a Day (' + in_out_val + ')')

    fig.show()
tmp_df = round(temp_iot_data.groupby(['out_in', 'month', 'month_name', 'hour'])['temp'].mean(), 1).reset_index()

tmp_df.head()
for out_in_val in ['In', 'Out']:



    fig = go.Figure()

    

    for mth in range(9, 13):

    

        mth_name = pd.to_datetime('01' + str(mth) + '2019', format = '%d%m%Y').month_name()

        filter_cond = ((tmp_df['month'] == mth) & (tmp_df['out_in'] == out_in_val))



        fig.add_trace(go.Scatter(x = tmp_df.loc[filter_cond, 'hour'],

                                 y = tmp_df.loc[filter_cond, 'temp'],

                                 mode = 'lines+markers',

                                 name = mth_name))

    

    fig.update_xaxes(tickvals = list(range(0, 24)), ticktext = list(range(0, 24)), title = '24 Hours')

    fig.update_yaxes(title = 'Temperature (in degree Celsius)')

    fig.update_layout(title = 'Hourly Avg. Temperature for each month (' + out_in_val + ')')

    fig.show()
tmp_df = round(temp_iot_data.groupby(['out_in', 'month', 'month_name', 'day_name'])['temp'].mean(), 1).reset_index()

tmp_df.head()
for out_in_val in ['In', 'Out']:



    fig = go.Figure()

    

    for mth in range(9, 13):

    

        mth_name = pd.to_datetime('01' + str(mth) + '2019', format = '%d%m%Y').month_name()

        filter_cond = ((tmp_df['month'] == mth) & (tmp_df['out_in'] == out_in_val))



        fig.add_trace(go.Scatter(x = tmp_df.loc[filter_cond, 'day_name'],

                                 y = tmp_df.loc[filter_cond, 'temp'],

                                 mode = 'markers',

                                 name = mth_name,

                                 marker = dict(size = tmp_df.loc[filter_cond, 'temp'].tolist())                                 

                                ))

    

    fig.update_xaxes(title = 'Day', categoryarray = np.array(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))

    fig.update_yaxes(title = 'Temperature (in degree Celsius)')

    fig.update_layout(title = 'Day-wise Avg. Temperature for each month (' + out_in_val + ')')

    fig.show()
tmp_df = temp_iot_data.groupby(['noted_date', 'out_in'])['temp'].mean().round(1).reset_index()
fig = go.Figure()



for out_in_val in ['In', 'Out']:



    filter_cond = (tmp_df['out_in'] == out_in_val)



    fig.add_trace(go.Scatter(x = tmp_df.loc[filter_cond, 'noted_date'],

                             y = tmp_df.loc[filter_cond, 'temp'],

                             mode = 'lines',

                             name = out_in_val))

    

fig.update_xaxes(title = 'Noted Date')

fig.update_yaxes(title = 'Temperature (in degree Celsius)')

fig.update_layout(title = 'Day-wise Temperature')

fig.show()