import pandas as pd

import plotly.express as px



df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()



df['daily_existing']=df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()



fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='thermal',range_color=(0.1,50000.))

fig.update_layout(title_text='Confirmed Cumulative Cases per Country_Shiho Tomita',title_x=0.5)

fig.write_html('Confirmed Cumulative Cases per Country_Shiho Tomita.html')

fig.show()
df = pd.read_csv('../input/earthquake-may252015/query (2).csv',header=0)



df.index = pd.to_datetime(df['time'])

df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')

fig = px.scatter_geo(df,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='YlOrRd',range_color=(3.,6.5))

fig.write_html('Nepal earthquake May-25-2015_Shiho Tomita.html')

fig.show()