from IPython.display import HTML

from IPython.display import IFrame



HTML('<iframe width="560" height="315" src="//www.youtube.com/embed/nBnCsMYm2yQ" frameborder="0" allowfullscreen></iframe>')
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
station_df = pd.read_csv("/kaggle/input/aws-open-source-weather-transaction-and-metadata/2019.csv", header = 1, \

                 names = ['station_code', 'w_date', 'element_type', 'element_value', 'measurement_flag', 'quality_flag', \

                          'source_flag', 'obs_time'], \

                delimiter = ',')

station_df.head()
unique_statuion = station_df['element_type'].unique()

unique_statuion
unique_statuion.size
station_df['w_date'].max()
df_grp_dt = station_df.groupby('w_date').count().sort_values(by = 'station_code', ascending = False)

df_grp_dt['station_code'].head()
station_df[station_df.measurement_flag == 'T'].head()
station_df[station_df.quality_flag == 'W'].head(5)
station_df[station_df.source_flag == 'E'].head(5)
import pandas as pd

md_station_df = pd.read_csv("/kaggle/input/aws-open-source-weather-transaction-and-metadata/ghcnd-stations.txt", header = None, sep = '\s+', \

                         names = ['station_id', 'latitude', 'longitude', 'elevation', 'state', 'name', 'gsn_flag', \

                                  'hcn_flag', 'wmo_id'])

md_station_df.head()

import folium

from folium import features





md_city_cordinates = md_station_df[md_station_df.state.str.contains("BERLIN", na=False)][['state','latitude', 'longitude']]

md_city_cordinates



berlin_location = [md_city_cordinates.iloc[0].latitude, md_city_cordinates.iloc[0].longitude]



#tiles="https://1.base.maps.api.here.com/maptile/2.1/maptile/newest/normal.day/{z}/{x}/{y}/256/png8?lg=eng&app_id=%s&app_code=%s"



m = folium.Map(location=berlin_location, zoom_start=11, tiles="openstreetmap", attr="HERE.com")



# mark each station as a point

for index, row in md_city_cordinates.iterrows():

    folium.CircleMarker([row['latitude'], row['longitude']],

                        radius=15,

                        popup=row['state'],

                        fill_color="blue", # divvy color

                        con_color='white',

                       ).add_to(m)



    



m
md_station_df[(md_station_df.station_id.str.startswith('GM')) & (md_station_df.state.str.startswith('MUNCHEN'))]
import pandas as pd

from io import StringIO



file = "/kaggle/input/aws-open-source-weather-transaction-and-metadata/ghcnd-countries.txt"



def parse_country_file(filename):

    with open(filename) as f:

        for line in f:

            yield line.strip().split(' ', 1)



country_df = pd.DataFrame(parse_country_file(file))

country_df.columns=['country_code', 'country_name']

country_df[country_df['country_name'].isin(['Germany', 'Spain', 'Italy', 'France', 'United Kingdom'])]
import pandas as pd

from io import StringIO



file = "/kaggle/input/aws-open-source-weather-transaction-and-metadata/ghcnd-states.txt"



def parse_country_file(filename):

    with open(filename) as f:

        for line in f:

            yield line.strip().split(' ', 1)



state_df = pd.DataFrame(parse_country_file(file))

state_df.head()
import pandas as pd

file = "/kaggle/input/aws-open-source-weather-transaction-and-metadata/ghcnd-inventory.txt"

ghcnd_inventory_df = pd.read_csv(file, sep = '\s+', header=None, names = ['staion_id','latitude', 'longitude', 'element', 'firstyear', 'lastyear'])

ghcnd_inventory_df.head()
import matplotlib.pyplot as plt



sydney_station_id = md_station_df[md_station_df.state.str.contains('SYDNEY', na=False)].station_id.unique().tolist()

sydney_station_data = station_df[station_df.station_code.isin(sydney_station_id)]



sydney_data_plot = sydney_station_data[sydney_station_data.element_type.str.contains("TAVG")][['station_code', 'w_date', 'element_type']]

sydney_data_plot['element_type_celcius'] = sydney_station_data.element_value/10



sydney_data_pyplot = sydney_data_plot

sydney_data_pyplot.w_date = pd.to_datetime(sydney_data_pyplot['w_date'], format='%Y%m%d')

sydney_data_pyplot.set_index(['w_date'],inplace=True)



plt.figure(figsize = (20,4))

sydney_data_pyplot.plot()

hamburg_station_id = md_station_df[md_station_df.state.str.contains('ERLANGEN', na=False)].station_id.unique().tolist()

hamburg_station_data =  station_df[station_df.station_code.isin(hamburg_station_id)]

hamburg_data_plot = hamburg_station_data[hamburg_station_data.element_type.str.contains("TAVG")][['station_code', 'w_date', 'element_type']]

hamburg_data_plot['element_type_celcius'] = hamburg_station_data.element_value/10



tokyo_station_id = md_station_df[md_station_df.state.str.contains('TOKYO', na=False)].station_id.unique().tolist()

tokyo_station_data =  station_df[station_df.station_code.isin(tokyo_station_id)]

tokyo_data_plot = tokyo_station_data[tokyo_station_data.element_type.str.contains("TAVG")][['station_code', 'w_date', 'element_type']]

tokyo_data_plot['element_type_celcius'] = tokyo_station_data.element_value/10



sydney_station_id = md_station_df[md_station_df.state.str.contains('SYDNEY', na=False)].station_id.unique().tolist()

sydney_station_data =  station_df[station_df.station_code.isin(sydney_station_id)]

sydney_data_plot = sydney_station_data[sydney_station_data.element_type.str.contains("TAVG")][['station_code', 'w_date', 'element_type']]

sydney_data_plot['element_type_celcius'] = sydney_station_data.element_value/10
import seaborn as sns



#bts_data_seaplot = bts_data_plot

hamburg_data_plot.w_date = pd.to_datetime(hamburg_data_plot['w_date'], format='%Y%m%d')

tokyo_data_plot.w_date = pd.to_datetime(tokyo_data_plot['w_date'], format='%Y%m%d')

sydney_data_plot.w_date = pd.to_datetime(sydney_data_plot['w_date'], format='%Y%m%d')



plt.figure(figsize = (20,4))

sns.set(rc={'axes.facecolor':'cyan', 'figure.facecolor':'cornflowerblue'})

#sns.set_style("darkgrid")

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.lineplot(hamburg_data_plot["w_date"],hamburg_data_plot["element_type_celcius"], color = "red")

sns.lineplot(tokyo_data_plot["w_date"],tokyo_data_plot["element_type_celcius"], color = "darkslategrey")

sns.lineplot(sydney_data_plot["w_date"],sydney_data_plot["element_type_celcius"], color = "steelblue")

plt.show()
station_tran_ms_df = pd.merge(station_df, md_station_df, left_on = ['station_code'], right_on = ['station_id'], \

                                how='left')

station_tran_ms_df.head()
station_tran_ms_tmax_df = station_tran_ms_df[(station_tran_ms_df.element_type.str.contains("TMAX" , na=False)) & 

                                            (station_tran_ms_df.station_code.str.startswith('GM', na=False))]

max_tmp = station_tran_ms_tmax_df['element_value'].max()

max_tmp
max_tmp_loc = station_tran_ms_tmax_df[station_tran_ms_tmax_df.element_value == max_tmp][['state','latitude', 'longitude']]

max_tmp_loc



plt_location = [max_tmp_loc.iloc[0].latitude, max_tmp_loc.iloc[0].longitude]



mpl = folium.Map(location=plt_location, zoom_start=11)

folium.TileLayer('stamenterrain').add_to(mpl)





# mark each station as a point

for index, row in max_tmp_loc.iterrows():

    folium.CircleMarker([row['latitude'], row['longitude']],

                        radius=15,

                        popup=row['state'],

                        fill_color="blue", # divvy color

                        con_color='white',

                       ).add_to(mpl)



    



mpl
station_tran_snowfall_max_df = station_tran_ms_df[station_tran_ms_df.element_type.str.contains("SNOW" , na=False)]

max_snowfall = station_tran_snowfall_max_df['element_value'].max()

max_snowfall

max_snowfall_loc = station_tran_snowfall_max_df[station_tran_snowfall_max_df.element_value == max_snowfall][['state','latitude', 'longitude']]

plt_location = [max_snowfall_loc.iloc[0].latitude, max_snowfall_loc.iloc[0].longitude]



mpl = folium.Map(location=plt_location, zoom_start=11, tiles="openstreetmap", attr="HERE.com")



# mark each station as a point

for index, row in max_snowfall_loc.iterrows():

    folium.CircleMarker([row['latitude'], row['longitude']],

                        radius=15,

                        popup=row['state'],

                        fill_color="blue", # divvy color

                        con_color='white',

                       ).add_to(mpl)



    



mpl