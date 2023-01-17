# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import requests
mydateparser = lambda x: pd.datetime.strptime(x, "%d/%m/%y")
df_covid19_india = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv', parse_dates = ['Date'] ,date_parser = mydateparser)
df_patients = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')
df_patients.head()
perc_age_null = df_patients[df_patients.age.isnull()].shape[0] / df_patients.shape[0]
perc_gender_null = df_patients[df_patients.gender.isnull()].shape[0] / df_patients.shape[0]
print('Percentage of records without age {:.2f} %'.format(perc_age_null*100))
print('Percentage of records without gender {:.2f} %'.format(perc_gender_null*100))
df_patients_sanity = df_patients.groupby('detected_state').agg({'age':['count'], 'gender':['count'], 'id':['count']})
df_patients_sanity['%age data given'] = df_patients_sanity['age'] / df_patients_sanity['id']
df_patients_sanity['%gender data given'] = df_patients_sanity['gender'] / df_patients_sanity['id']
df_patients_sanity.sort_values(['%age data given','%gender data given'], ascending=False)
df_statewise_timeseries = df_covid19_india.groupby(['Date','State/UnionTerritory']).agg({'Confirmed':['sum'], 'Cured':['sum'], 'Deaths':['sum']})
# df_statewise_timeseries = df_statewise_timeseries.sort_values(['Date', 'State/UnionTerritory'])
df_statewise_timeseries = df_statewise_timeseries.reset_index()
df_statewise_timeseries.columns = df_statewise_timeseries.columns.get_level_values(0)
fig = px.line(df_statewise_timeseries, x='Date', y='Confirmed', color='State/UnionTerritory', line_group='State/UnionTerritory', labels = 'Tag')
fig.update_layout(template='plotly_dark')
fig.show()
df_statewise_timeseries
# fig = px.scatter( confirmed_by_country_df, x='Confirmed', y='Confirmed', log_y=True,  size = 'Deaths', color = 'Deaths',
#                  log_x=True,  hover_name='Country', animation_frame="Date", animation_group="Country", text='Country',
#                 )
max_xvalue = df_statewise_timeseries.Confirmed.max()
max_xvalue = np.log10(max_xvalue) + 1

max_yvalue = df_statewise_timeseries.Deaths.max() 
# max_yvalue = np.log10(max_yvalue) + 1

df_statewise_timeseries['Date_str'] = df_statewise_timeseries.Date.apply(lambda x: x.date()).apply(str) 
fig = px.scatter( df_statewise_timeseries, x='Confirmed', y='Deaths',  color = 'Deaths',
                 log_x=True,  hover_name='State/UnionTerritory', animation_frame='Date_str', animation_group='State/UnionTerritory', text='State/UnionTerritory'
                )
fig.update_layout(template='plotly_dark', title='State-wise Confirmed Rate Analysis', )
fig.update_xaxes(range=[0, max_xvalue])
fig.update_yaxes(range=[0, max_yvalue])
fig.show()
r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson')
# r = requests.get(url='https://raw.githubusercontent.com/karthikcs/india-states-geojson/master/india-states.geojson')
# r = requests.get(url='https://github.com/karthikcs/india-states-geojson/raw/master/map.geojson')

geojson = r.json()

def change_state_name(state):
    if state == 'Odisha':
        return 'Orissa'
    elif state == 'Telengana':
        return 'Telangana'
    elif state == 'Jharkhand#':
        return 'Jharkhand'
    return state
df_ind_cases = df_statewise_timeseries
df_ind_cases['State/UnionTerritory'] = df_ind_cases.apply(lambda x: change_state_name(x['State/UnionTerritory']), axis=1)
last_date = df_ind_cases.Date.max()
df_ind_states = df_ind_cases.copy()
df_ind_cases = df_ind_cases[df_ind_cases['Date']==last_date]
columns = ['State/UnionTerritory', 'Cured', 'Deaths','Confirmed']
df_ind_cases = df_ind_cases[columns]
df_ind_cases.sort_values('Confirmed',inplace=True, ascending=False)
df_ind_cases.reset_index(drop=True,inplace=True)
df_ind_cases.style.background_gradient(cmap='Reds')
# max_y = df_statewise_timeseries['Confirmed'].max()
df_statewise_timeseries.sort_values(['Date','Confirmed'], ascending=[True,False], inplace=True)
fig = px.bar(df_statewise_timeseries, log_y = True, x='State/UnionTerritory', y='Confirmed', animation_frame='Date_str', animation_group='State/UnionTerritory', )
fig.update_layout(template='plotly_dark', title='State-wise confirmed cases over period of time', )
fig.show()
fig = px.choropleth(df_ind_cases, geojson=geojson, 
                    color="Confirmed",
                    locations="State/UnionTerritory", featureidkey="properties.NAME_1",
                    hover_data=['Cured','Deaths'],
                    color_continuous_scale=px.colors.sequential.YlOrRd,
                    title='India: Total Current cases per state'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=600,margin={"r":0,"t":30,"l":0,"b":30})
fig.show()
