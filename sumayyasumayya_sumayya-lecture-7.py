import pandas as pd

import plotly.express as px



df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()



fig = px.choropleth(df,locations='Country/Region',locationmode='country names',

                    color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',

                    color_continuous_scale='Viridis_r',range_color=(0.1,200000))

fig.update_layout(title_text='Confirmed Cumulative Cases per Country',title_x=0.5)

fig.show()
df = pd.read_csv('../input/earthquake-on-april-25-2015/20150425_earthquake.csv', header=0) #from USGS

df = df.groupby(['time','latitude','longitude']).sum().reset_index()

df.index = pd.to_datetime(df['time'])

df['time'] = df.index.strftime('%H:00')

fig = px.scatter_geo(df,lat='latitude',lon='longitude',size='mag',color='mag',animation_frame='time',

                     color_continuous_scale='Viridis_r',size_max= 20,range_color=(4.,8.))

fig.update_layout(title_text='Nepal Earthquake April 25, 2015',title_x=0.5)

fig.show()