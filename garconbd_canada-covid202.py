import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
%matplotlib inline

import plotly
import plotly.express as px
import plotly.graph_objects as go
#plt.rcParams['figure.figsize']=17,8
import cufflinks as cf
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot

import folium

df=pd.read_csv("../input/covidcanada/Canada_covid19.csv")
df.drop(['id'],axis=1,inplace=True)
total_cases=df['case'].sum()
total_death=df['death'].sum()
print('The total number of cases as of in Canada is ',total_cases)
print('The total number of death as of in Canada is ',total_death)
df_province=df.groupby('province')
for province, df_provincee in df_province:
    print(df_provincee)

df_sum=df_province.sum()
df_sum.sort_values(by=['case'],ascending=False)
pyo.init_notebook_mode(connected=True)
cf.go_offline()
df.iplot(kind='bar',x='province',y='case', color=['red'])
#Matplotlib
fig=plt.figure(figsize=(20,10),dpi=200)
axes=fig.add_axes([0,0,1,1])
axes.bar(df['province'],df['case'],color='green')
axes.set_title('Total Cases in Canada')
axes.set_xlabel('province')
axes.set_ylabel('Total Cases')
plt.show()
fig=plt.figure(figsize=(20,10),dpi=200)
axes=fig.add_axes([0,0,1,1])
axes.bar(df['province'],df['death'],color='red')
axes.set_title('Total Death in Canada')
axes.set_xlabel('Name of Province')
axes.set_ylabel('Total Death')
df_sum.sort_values('province',inplace=True)
plt.show()

Canada_Cord=pd.read_csv("../input/canadamap/Canada_map.csv")
df_full=pd.merge(Canada_Cord,df_sum,on='province')
df_full
map=folium.Map(location=[56,-90],zoom_start=3.5,tiles='Stamenterrain')

for lat,long,value, name in zip(df_full['Latitude'],df_full['Longitude'],df_full['case'],df_full['province']):
    folium.CircleMarker([lat,long],radius=value*0.001,popup=('<strong>State</strong>: '+str(name).capitalize()+'<br>''<strong>Total Cases</strong>: ' + str(value)+ '<br>'),color='red',fill_color='red',fill_opacity=0.3).add_to(map)
    
map
dbd_canada=pd.read_csv("../input/canada-datewise/Canada_datewise.csv")
#matplotlib
fig=plt.figure(figsize=(10,5),dpi=200)
axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.bar(dbd_canada['date'],dbd_canada['case'],color='blue')
axes.set_xlabel('date')
axes.set_ylabel('case')
axes.set_title('Total Case in Canada')
#plotly Express

fig=px.bar(dbd_canada,x='date',y='cum_case',color='cum_case',title='Confirmed cases in Canada')
fig.show()
fig=go.Figure()
fig.add_trace(go.Scatter(x=dbd_canada['date'],y=dbd_canada['cum_case'],mode='markers',marker=dict(color='rgba(0, 256, 0, 0.9)',size=3),name='Case'))
fig.add_trace(go.Scatter(x=dbd_canada['date'],y=dbd_canada['cum_death'],mode='markers',marker=dict(color='rgba(256, 0, 0, 0.5)',size=3),name='Deaths',line=dict(color='red',width=2)))
fig.add_trace(go.Scatter(x=dbd_canada['date'],y=dbd_canada['cum_recovered'],mode='markers',marker=dict(color='rgba(0, 0, 256, 0.9)',size=3),name='Recovered',line=dict(color='green',width=2)))
df_world=pd.read_csv('../input/world-timeseries/world_timeseries.csv')
df_world
fig=px.density_mapbox(df_world,lat='lat',lon='long',hover_name='location',hover_data=['total_cases','total_deaths','total_tests'],animation_frame='date',color_continuous_scale="Portland",radius=7,zoom=1,height=700)
fig.update_layout(title='Worldwide Corona Virus Cases')
fig.update_layout(mapbox_style="open-street-map",mapbox_center_lon=0)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})