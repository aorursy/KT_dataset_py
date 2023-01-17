import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns 

import datetime as dt

import folium

from folium.plugins import HeatMap, HeatMapWithTime

%matplotlib inline
df=pd.read_csv(r"/kaggle/input/corona-virus-report/covid_19_clean_complete.csv",parse_dates=True)

df.head()
df.info()
#Convert date from string to date format for easier analysis

df['Date']=pd.to_datetime(df['Date'])

#Renaming the column 

df.rename(columns={"Country/Region":"Country"},inplace=True)

#Dropping province column

df.drop(columns=['Province/State'],inplace=True)
print("The data has COVID presence statistics from {} countries, starting from {} till {}".format(df['Country'].nunique(), df['Date'].min().strftime("%d/%m/%Y"), df['Date'].max().strftime("%d/%m/%Y")))
df.describe()
fig, axes = plt.subplots(nrows=4,figsize=(24,12))

df.plot(x="Date",y="Confirmed",ax=axes[0],color='coral')

df.plot(x="Date",y="Deaths",ax=axes[1],color='goldenrod')

df.plot(x="Date",y="Recovered",ax=axes[2],color='lightsteelblue')

df.plot(x="Date",y="Active",ax=axes[3],color='yellowgreen')
df_recent = df[df['Date']>'2020-06-15']

df_recent.head()
data_wld = df_recent[['Country','Lat','Long']]

data_wld.shape
data_wld=data_wld.drop_duplicates(subset = ['Country'],keep='first').set_index('Country')

data_wld.head()
df_recent_sum = df_recent.groupby('Country').sum()

df_recent_sum.head()
for i in data_wld.index:

    data_wld.loc[i,'Confirmed'] = df_recent_sum.loc[i,'Confirmed']

    data_wld.loc[i,'Deaths'] = df_recent_sum.loc[i,'Deaths']

    data_wld.loc[i,'Recovered'] = df_recent_sum.loc[i,'Recovered']

    data_wld.loc[i,'Active'] = df_recent_sum.loc[i,'Active']

    

data_wld = data_wld.astype({"Confirmed":int, "Deaths":int, "Recovered":int, "Active":int})
data_wld = data_wld.reset_index()

data_wld.head()
#This json file has the boundary data for countries

world_geo = '/kaggle/input/python-folio-country-boundaries/world-countries.json'



m = folium.Map(location=[0, 0], zoom_start=2)



folium.Choropleth(

    geo_data=world_geo,

    name='Confirmed cases - regions',

    key_on='feature.properties.name',

    fill_color='YlGn',

    fill_opacity=0.05,

    line_opacity=0.3,

).add_to(m)



radius_min = 0.1

radius_max = 30

weight = 4

fill_opacity = 0.2



_color_act = 'red'

group0 = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Confirmed cases</span>')

for i in range(len(data_wld)):

    lat = data_wld.loc[i, 'Lat']

    lon = data_wld.loc[i, 'Long']

    country = data_wld.loc[i, 'Country']

    active = data_wld.loc[i, 'Active']

    recovered = data_wld.loc[i, 'Recovered']

    death = data_wld.loc[i, 'Deaths']



    _radius_act = np.sqrt(np.cbrt(data_wld.loc[i, 'Active']))

    if _radius_act < radius_min:

        _radius_act = radius_min



    if _radius_act > radius_max:

        _radius_act = radius_max



    _popup_act = str(country) + '\n(Active='+str(active) + '\nDeaths=' + str(death) + '\nRecovered=' + str(recovered) + ')'

    folium.CircleMarker(location = [lat,lon], 

                        radius = _radius_act, 

                        popup = _popup_act, 

                        color = _color_act, 

                        fill_opacity = fill_opacity,

                        weight = weight, 

                        fill = True, 

                        fillColor = _color_act).add_to(group0)



group0.add_to(m)

folium.LayerControl().add_to(m)

m
df_region = df[['Date','WHO Region','Confirmed']]

df_region['Month'] = pd.DatetimeIndex(df_region['Date']).month

df_region.drop(columns=['Date'],inplace=True)

df_region.sort_values(by='Month',ascending=True,inplace=True)

#df_region['Month'].replace({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun'},inplace=True)

df_region.groupby(['Month','WHO Region']).mean()

df_region.tail()
csmp = pd.pivot_table(df_region, index=df_region['WHO Region'], columns=df_region['Month'], values='Confirmed')

plt.figure(figsize=(11,9))

plt.title("Average confirmed cases each day by month for each WHO region")

sns.heatmap(data=csmp,fmt=".1f",annot=True,cmap = sns.cm.rocket_r)
fig, axes = plt.subplots(figsize=(16,25),nrows=4)

sns.lineplot(x="Date", y="Confirmed", hue='WHO Region', data=df,palette=['green','orange','brown','dodgerblue','red',"yellowgreen"],ci=None,ax=axes[0])

sns.lineplot(x="Date", y="Deaths", hue='WHO Region', data=df,palette=['green','orange','brown','dodgerblue','red',"yellowgreen"],ci=None,ax=axes[1])

sns.lineplot(x="Date", y="Recovered", hue='WHO Region', data=df,palette=['green','orange','brown','dodgerblue','red',"yellowgreen"],ci=None,ax=axes[2])

sns.lineplot(x="Date", y="Active", hue='WHO Region', data=df,palette=['green','orange','brown','dodgerblue','red',"yellowgreen"],ci=None,ax=axes[3])
df_focus = df[df['WHO Region'].isin(['Eastern Mediterranean', 'South-East Asia', 'Americas'])]

df_focus.drop(columns=['Lat','Long','WHO Region'],inplace=True)

df_focus = df_focus.groupby('Country').sum().reset_index()

df_focus.head()
fig = px.scatter(df_focus.sort_values('Deaths', ascending=False).iloc[:20, :], 

                 x='Confirmed', y='Deaths', color='Country', size='Confirmed', 

                 height=700, text='Country', log_x=True, log_y=True, 

                 title='Deaths vs Confirmed (log10)')

fig.update_traces(textposition='top center')

fig.update_layout(showlegend=False)

fig.update_layout(xaxis_rangeslider_visible=False)

fig.show()