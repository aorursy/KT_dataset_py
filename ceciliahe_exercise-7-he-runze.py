import pandas as pd

import plotly.express as px



df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()



fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color="Confirmed",hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Mint',range_color=(0.1,300000))

fig.update_layout(title_text='Confirmed Cumulative Cases per Country',title_x=0.5)

#choose color: https://plotly.com/python/builtin-colorscales/#



fig.show()
fig.write_html('Confirmed Cumulative Cases per Country.html')
df['daily_exsiting'] = df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()



fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color="daily_exsiting",hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Burg',range_color=(0.1,300000))

fig.update_layout(title_text='Remaining Confirmed Cases per Country of each day',title_x=0.5)



fig.show()
fig.write_html('Remaining Confirmed Cases per Country of each day.html')
df = pd.read_csv('../input/nepal-earthquake-april-25-2015/query.csv',header=0)

#data from https://earthquake.usgs.gov/earthquakes/search/



df.index = pd.to_datetime(df['time'])

#change minutes to 0 to showing the hour.

#reduce the resolution to an hourly data, by using function strftime('%Y-%m-%d %H:00:00')

df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')



fig = px.scatter_geo(df,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='Agsunset',range_color=(5.,7.))

#'px.choropleth' shows the country maps or some singular values for the same countries.

#'px.scatter_geo' will be using the latitude and longtitude, and will be putting the points.



fig.show()