import pandas as pd

import plotly.express as px



df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()



fig = px.choropleth(df,locations='Country/Region',locationmode='country names', color='Confirmed',hover_name='Country/Region', animation_frame='ObservationDate',color_continuous_scale='Viridis',range_color=(0.1,100000.))

fig.update_layout(title_text='Confirmed Cumulative COVID-19 Cases per Country',title_x=0.5)

fig.show()

fig.write_html('Animation_COVID-19_Cummulative_Cases_Global_20M51889.html')

df['daily_existing'] = df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()



fig2 = px.choropleth(df,locations='Country/Region',locationmode='country names', color='daily_existing',hover_name='Country/Region', animation_frame='ObservationDate',color_continuous_scale='Viridis',range_color=(0.1,100000.))

fig2.update_layout(title_text='Remaining Confirmed COVID-19 Cases per Country',title_x=0.5)

fig2.show()

fig2.write_html('Animation_COVID-19_Remaining_Cases_Global_20M51889.html')
dfe = pd.read_csv('../input/april-25-2015-earthquake/query_nepal.csv', header=0)



#first convert the time into date time object 

dfe.index = pd.to_datetime(dfe['time'])



#return time object to string and group by hour

dfe['time']=dfe.index.strftime('%Y-%m-%d %H:00:00')



fig3 = px.scatter_geo(dfe,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='Viridis',range_color=(5.,7.))

fig3.update_layout(title_text='Hourly Global Map of Earthquake, April 25th, 2015',title_x=0.5)

fig3.show()



fig3.write_html('Animation_Nepal_Earthqukae_April252015_20M51889.html')


