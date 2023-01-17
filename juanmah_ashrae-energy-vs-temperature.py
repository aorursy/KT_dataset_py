import numpy as np

import pandas as pd

import plotly.express as px

from IPython.core.display import display, HTML

import pickle
data_path = '../input/ashrae-data-wrangling-csv-to-pickle/'

with open(data_path + 'X_train.pickle', 'rb') as f:

    X_train = pickle.load(f)

with open(data_path + 'nan.pickle', 'rb') as f:

    nan = pickle.load(f)    

with open(data_path + 'building_metadata.pickle', 'rb') as f:

    building_metadata = pickle.load(f)

with open(data_path + 'weather_train.pickle', 'rb') as f:

    weather_train = pickle.load(f)

    

world_cities = pd.read_csv('../input/world-cities/worldcities.csv')
weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'], format='%Y-%m-%d %H:%M:%S')
weather_train['hour'] = weather_train['timestamp'].dt.hour

weather_train['day'] = weather_train['timestamp'].dt.day

weather_train['week'] = weather_train['timestamp'].dt.week

weather_train['month'] = weather_train['timestamp'].dt.month
weather_train.head()
weather_hour = weather_train.groupby(by=['site_id', 'hour']).agg([np.mean, np.std])

weather_hour.reset_index(inplace=True)

weather_hour
weather_hour['air_temperature']['mean']
level_0 = weather_hour.columns.droplevel(0)

level_1 = weather_hour.columns.droplevel(1)

level_0 = ['' if x == '' else '-' + x for x in level_0]

weather_hour.columns = level_1 + level_0
import seaborn as sns

palette = sns.color_palette("Oranges", 16)

palette_hex = [f'#{int(256*x[0]):02x}{int(256*x[1]):02x}{int(256*x[2]):02x}'for x in palette]

fig = px.line(weather_hour,

             x='hour',

             y='air_temperature-mean',

             color='site_id',

             color_discrete_sequence=palette_hex)

fig.update_layout()

fig.show()
weather_hour
time_offset = weather_hour[['site_id', 'air_temperature-mean']].groupby(by=['site_id']).agg(['idxmin'])

time_offset[('air_temperature-mean', 'idxmin')]
weather_time_offset = weather_hour.iloc[time_offset[('air_temperature-mean', 'idxmin')]][['site_id', 'hour']]

weather_time_offset.reset_index(drop=True, inplace=True)

weather_time_offset['offset'] = weather_time_offset['hour'] - weather_time_offset['hour'].min()

weather_time_offset.drop(['hour'], axis='columns', inplace=True)

weather_time_offset
weather_time_offset['city'] = ['Jacksonville',

                               'London',

                               'Phoenix',

                               'Philadelphia',

                               'San Francisco',

                               'Leicester',

                               'Philadelphia',

                               'Montréal',

                               'Jacksonville',

                               'San Antonio',

                               'Las Vegas',

                               'Montréal',

                               'Dublin',

                               'Minneapolis',

                               'Philadelphia',

                               'Pittsburgh']

weather_time_offset
def get_city_info(city):

    results = world_cities[world_cities['city'] == city].sort_values(by='population', ascending=False)[['lat', 'lng', 'country', 'admin_name', 'population']]

    if results.empty:

        return None

    else:

        return results.iloc[0]



city_info = weather_time_offset['city'].apply(get_city_info)

city_info
weather_time_offset = pd.concat([weather_time_offset, city_info], axis='columns')

weather_time_offset
import plotly.express as px

fig = px.scatter_geo(weather_time_offset,

                     lat='lat',

                     lon='lng',

                     color='offset',

                     size='population',

                     hover_name='city',

                     hover_data=['admin_name', 'country'],

                     projection='kavrayskiy7',

                     color_continuous_scale=px.colors.sequential.Plasma)

fig.show()