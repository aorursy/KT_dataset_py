import pandas as pd
import plotly.express as px
import numpy as np

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

# Confirmed Cumulative Cases
fig = px.choropleth(df, locations='Country/Region', locationmode='country names', color='Confirmed', hover_name='Country/Region', animation_frame='ObservationDate', color_continuous_scale='amp',range_color=(0.1,300000))
fig.update_layout(title_text='Confirmed Cumulative Cases per Country (Eishi Kitano)', title_x=0.5)
fig.show()
fig.write_html('Confirmed Cumulative Cases per Country (Eishi Kitano).html')
# Remaining Confirmed Cases
df['daily_existing'] = df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()
fig2 = px.choropleth(df, locations='Country/Region', locationmode='country names', color='daily_existing', hover_name='Country/Region', animation_frame='ObservationDate', color_continuous_scale='Blues',range_color=(0.1,100000))
fig2.update_layout(title_text='Remaining Confirmed Cases per Country of every day (Eishi Kitano)', title_x=0.5)
fig2.show()
fig2.write_html('Remaining Confirmed Cases per Country of every day (Eishi Kitano).html')
# Daily confirmed
df2= df.sort_values(by=['Country/Region','ObservationDate'],ascending=[True,True])
df3 = pd.pivot_table(df, index='ObservationDate', columns='Country/Region', values='Confirmed')
df3 = df3.fillna(0)
df3 = df3.diff()
df3.to_csv('daily.csv')
df4 = pd.read_csv('../input/daily3/daily3.csv',header=0)
fig3 = px.choropleth(df4, locations='Country/Region', locationmode='country names', color='Confirmed', hover_name='Country/Region', animation_frame='ObservationDate', color_continuous_scale='Reds',range_color=(0.1,7000))
fig3.update_layout(title_text='Daily_confirmed per Country (Eishi Kitano)', title_x=0.5)
fig3.show()
fig3.write_html('Daily_confirmed per Country (Eishi Kitano).html')
# April 2015 Nepal Earthquake
df5 = pd.read_csv('../input/april-25-2015-earthquake/query (1).csv')

df5.index = pd.to_datetime(df5['time'])
df5['time'] = df5.index.strftime('%Y-%m-%d %H:00:00')
fig4 = px.scatter_geo(df5, lat='latitude', lon='longitude', color='mag', animation_frame='time', color_continuous_scale='Viridis',range_color=(1.,7.))
fig4.update_layout(title_text='April 2015 Nepal Earthquake (Eishi Kitano)', title_x=0.5)
fig4.show()
fig4.write_html('April 2015 Nepal Earthquake(Eishi Kitano).html')
# Recent situation
df6 = pd.read_csv('../input/recent-situation/query_recent.csv')

df6.index = pd.to_datetime(df6['time'])
df6['time'] = df6.index.strftime('%Y-%m-%d %H:00:00')
fig5 = px.scatter_geo(df6, lat='latitude', lon='longitude', color='mag', animation_frame='time', color_continuous_scale='Viridis',range_color=(4.,7.))
fig5.update_layout(title_text='Earthquake Recent-situation (Eishi Kitano)', title_x=0.5)
fig5.show()
fig5.write_html('Earthquake Recent-situation(Eishi Kitano).html')
