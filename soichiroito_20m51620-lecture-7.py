import pandas as pd
import plotly.express as px

df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df=df.groupby(['ObservationDate','Country/Region']).sum().reset_index()
#print(df)
fig=px.choropleth(df,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='OrRd',range_color=(0.1,200000.))
fig.update_layout(title_text='Soichiro Ito',title_x=0.5)
fig.show()
df=pd.read_csv('../input/april-25-2015-earthquake/query.csv',header=0)

df.index=pd.to_datetime(df['time'])
df['time']=df.index.strftime('%Y-%m-%d %H:00:00')
fig=px.scatter_geo(df,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='Rainbow',range_color=(5.,7.))
fig.update_layout(title_text='Aplil 25,2015 Nepal earthquake',title_x=0.5)
fig.show()