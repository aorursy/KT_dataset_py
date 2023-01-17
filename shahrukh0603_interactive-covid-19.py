# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go
import plotly.express as px
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
confirmed_cases_path = base_url + 'time_series_covid19_confirmed_global.csv'
death_global_path = base_url + 'time_series_covid19_deaths_global.csv'
recovered_global_path = base_url + 'time_series_covid19_recovered_global.csv'
def clean_mydata(url_path, name_of_values):
    dtf = pd.read_csv(url_path,header = 0,skip_blank_lines = True)
    dtf.drop(['Province/State'],axis = 1,inplace = True)
    dtf.rename({'Country/Region': 'Country'},axis=1,inplace= True)
    dtf = pd.melt(dtf, id_vars = ['Country', 'Lat', 'Long'],var_name='Date',value_name=name_of_values)
    dtf['Date'] = pd.to_datetime(dtf['Date'])
    return dtf
def sum_cases(df):
    return df.groupby(['Country', 'Date'], as_index=False).sum(axis=0)
confirmed_cases = clean_mydata(confirmed_cases_path, 'confirmed')
death_cases = clean_mydata(death_global_path, 'deaths')
recovered_cases = clean_mydata(recovered_global_path, 'recovered')
confirmed_cases = sum_cases(confirmed_cases)
death_cases = sum_cases(death_cases)
recovered_cases = sum_cases(recovered_cases)
confirmed_cases
confirmed_cases_for_map = confirmed_cases.drop(['Date', 'Lat', 'Long'], axis = 1)
co = confirmed_cases_for_map.groupby(['Country']).max().reset_index()
text_country = input("Country you entered ")
if(text_country != ''):
    fig = px.line(confirmed_cases[(confirmed_cases.Country == text_country)], 
              x = 'Date', 
              y = 'confirmed')
    fig.show()
import datetime

t_day = datetime.datetime.today()
y_day = datetime.timedelta(days=1)
diff = (t_day-y_day).strftime('%Y-%m-%d')
joined_confirmed_death = pd.merge(confirmed_cases, death_cases, on=['Country', 'Date', 'Lat', 'Long'], how='outer')
joined_confirm_death_recover = pd.merge(joined_confirmed_death, recovered_cases, on=['Country', 'Date', 'Lat', 'Long'], how='outer')
joined_confirm_death_recover.describe()
df = confirmed_cases_for_map.groupby(['Country']).max().reset_index()
df = confirmed_cases
if(not(df.dtypes['Date']) is np.dtype('O')):
    df.Date = df.Date.apply(lambda x: x.strftime('%Y-%m-%d'))
df_static_world_map_cases = df[df.Country != 'US']
fig = px.choropleth(df_static_world_map_cases, locations="Country",
                    locationmode="country names",
                    color="confirmed",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Cividis_r)
fig.update_geos(projection_type="natural earth", 
                showcountries=True)
fig.update_layout(height=600, 
                  margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
fig = px.scatter_geo(df, locations="Country",
                    locationmode="country names",
                    color="confirmed",
                    hover_name="Country",
                    size = "confirmed",
                    animation_frame = "Date",
                    animation_group = "Country",
                    color_continuous_scale=px.colors.sequential.Cividis_r)
fig.update_geos(projection_type="natural earth", 
                showcountries=True)
fig.update_layout(height=600, 
                  margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
temp = joined_confirm_death_recover.sort_values(by='confirmed', ascending = False)
top_10_confirm = temp[(temp.Date == diff)].nlargest(10, 'confirmed')
fig = px.bar(top_10_confirm,
             category_orders = {'Country':top_10_confirm.confirmed.sort_values(ascending=True)},
             x = 'Country', 
             y = 'confirmed')
fig.show()
# Top 10 Countries with Deaths
top_10_confirm = temp[(temp.Date == diff)].nlargest(10, 'deaths')
fig = px.bar(top_10_confirm,
             category_orders = {'Country':top_10_confirm.deaths.sort_values(ascending=True)},
             x = 'Country', 
             y = 'deaths')
fig.show()