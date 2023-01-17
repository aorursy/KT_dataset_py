import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
weather_data = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

weather_data.set_index('timestamp', inplace=True)

weather_data.index = pd.to_datetime(weather_data.index)

weather_data['hourofday'] = weather_data.index.hour

weather_data['dayofyear'] = weather_data.index.dayofyear

weather_data.tail()
weather_data[weather_data['site_id']== 0]['air_temperature'][:24].plot()
weather_data[weather_data['site_id']== 0]['air_temperature'][5496:5520].plot()
weather_data[weather_data['site_id']== 0]['air_temperature'][24:48].plot()
weather_data[weather_data['site_id']== 10]['air_temperature'][:24].plot()
weather_data[weather_data['site_id']== 10]['air_temperature'][5496:5520].plot()
site_ids = list(set(weather_data['site_id']))

result = pd.DataFrame()

for site in site_ids:

    weather_data_site = weather_data[weather_data['site_id']== site]

    t_air_max_every_day = weather_data_site.groupby('dayofyear')['air_temperature'].max()

    weather_data_site['hit_t_air_max'] = weather_data_site.apply(lambda x: 1 if x['air_temperature'] == t_air_max_every_day[x['dayofyear']] else 0,axis=1)

    result['site'+str(site)] = weather_data_site.groupby('hourofday')['hit_t_air_max'].sum().values
plt.figure(figsize=(20, 10))

sns.heatmap(result, annot=True)