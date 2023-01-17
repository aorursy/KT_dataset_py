import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import plotly.express as px

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header= 0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

fig = px.choropleth(df, locations = 'Country/Region', locationmode = 'country names', color = 'Confirmed', hover_name = 'Country/Region',
                     animation_frame = 'ObservationDate', color_continuous_scale = 'jet', range_color = (0.1,200000.))
fig.update_layout(title_text='LIU Murong', title_x=0.5)
fig.show()



df = pd.read_csv('../input/apr-2015-nepal-earthquake/query.csv',header = 0)

df.index = pd.to_datetime(df['time'])
df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')
fig = px.scatter_geo(df, lat='latitude', lon='longitude', color='mag', animation_frame='time', color_continuous_scale='rainbow', 
                     range_color=(4.5,6.5))
fig.update_layout(title_text='LIU Murong', title_x=0.5)
fig.show()
#print(df)