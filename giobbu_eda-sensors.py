from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from datetime import datetime

import datetime

import plotly.graph_objects as go

import plotly.express as px

import folium

from folium import plugins
# marker_sal = folium.CircleMarker(location=[45.431078, 12.336378], popup='Punta della Salute s Station',color = "blue")

# marker_gior= folium.CircleMarker(location=[45.428408, 12.346264], popup='S. Giorgio s Station',color = "green" )

# marker_sal.add_to(folium_map)

# marker_gior.add_to(folium_map)
folium_map = folium.Map(location=[45.438759, 12.327145],

                        zoom_start=10,

                        tiles="CartoDB dark_matter")



popups = ['Punta della Salute','San Giorgio', 'Burano','Malamocco','Chioggia']

data = np.array([np.array([45.431078,45.428408,45.487503,45.339800,45.232539]),np.array([12.336378, 12.346264, 12.415486, 12.291967, 12.280597 ])]).T





plugins.MarkerCluster(data,popups).add_to(folium_map)



folium_map
df1 = pd.read_csv('/kaggle/input/venice-high-water-acqua-alta/VeneziaPuntaSalute.csv',delimiter=',')

df1.columns = ['date','time','liv','temp']



df1.time = df1.time.map(lambda x: x.rstrip('AMP').replace(';',''))

df1.temp = df1.temp.map(lambda x: x.replace(';','')).replace(r'^\s*$', np.nan, regex=True).astype(float)



# df1.isnull().sum()



df1.liv = df1.liv.fillna(method='pad')

df1.temp = df1.temp.fillna(method='pad')



#df1.isnull().sum()



dt = datetime.datetime(2019, 10, 10, 0, 10, 0)

end = datetime.datetime(2020, 1, 11, 0, 10, 0)

step = datetime.timedelta(minutes=10)



result = []



while dt < end:

    result.append(dt.strftime('%Y-%m-%d %H:%M:%S'))

    dt += step



df1.insert(0,'datetime',result)
df1.to_csv('df1.csv')

df1.head()
df1['datetime'] = pd.to_datetime(df1['datetime'])

df1['month'] = df1['datetime'].dt.month

df1['date'] = df1['datetime'].dt.date

df1['hour'] = df1['datetime'].dt.hour
import plotly.graph_objects as go





# Create traces

fig = go.Figure()



fig.add_trace(go.Scatter(x = df1['datetime'], y=df1['liv'],

                    mode='lines',

                    name='water level (m)'))

fig.add_trace(go.Scatter(x = df1['datetime'], y=df1['temp'],

                    mode='lines',

                    name='temperature (Celsius)'))



fig.update_layout(title='Water Level and Temperature',

                   xaxis_title='Datetime')





fig.show()
# Create figure

fig = go.Figure()



fig.add_trace(

    go.Scatter(x=list(df1.datetime), y=list(df1.liv)))



# Set title

fig.update_layout(

    title_text="Time series with range slider and selectors"

)



# Add range slider

fig.update_layout(

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label="day",

                     step="day",

                     stepmode='backward'),

                dict(count=2,

                     label="2days",

                     step="day",

                     stepmode='backward'),

                dict(count=3,

                     label="3days",

                     step="day",

                     stepmode='backward'),

                dict(count=144,

                     label="week",

                     step="hour",

                     stepmode='backward'),

                dict(count=1,

                     label="month",

                     step="month",

                     stepmode='backward'),

                dict(count=2,

                     label="2months",

                     step="month",

                     stepmode='backward'),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            visible=True

        ),

        type="date"

    )

)



fig.show()
fig = px.scatter(df1, x="hour", y="liv",

                 animation_frame = df1.datetime.dt.dayofyear,

                 range_x=[-1,24], range_y=[-1, 2.2])

fig.show()
df1 = df1.assign(latitude=45.431078)

df1 = df1.assign(longitude=12.336378)

df1
def color(elev): 

    if 0 <= elev <= 0.5 : 

        col = 'green'

    elif 0.5 <= elev <= 0.7 :

        col = 'blue'

    elif 0.7 <= elev <= 1.2 :

        col = 'orange'

    else: 

        col='red'

    return col 





def create_geojson_features(df):

    print('> Creating GeoJSON features...')

    features = []

    for _, row in df.iterrows():

        feature = {

            'type': 'Feature',

            'geometry': {

                'type':'Point', 

                'coordinates':[row['longitude'],row['latitude']]

            },

            'properties': {

                'time': row['date'].__str__(),

#                 'style': {'color' : color(row['liv'])},

                'icon': 'circle',

                'iconstyle':{

                    'fillColor': color(row['liv']),

                    'fillOpacity': 1,

                    'stroke': 'false',

                    'radius': 20,

                    'opacity':1

                }

            }

        }

        features.append(feature)

    return features





features = create_geojson_features(df1)
from folium.plugins import TimestampedGeoJson



mappo = folium.Map(location=[45.438759, 12.327145],

                        zoom_start=14,

                        tiles="CartoDB dark_matter")



TimestampedGeoJson(

        {'type': 'FeatureCollection',

        'features': features}

        , period='PT3H'

        , add_last_point=True

        , auto_play=False

        , loop=False

        , max_speed=30

        , loop_button=True

        , date_options='YYYY/MM/DD'

        , time_slider_drag_update=True

    ).add_to(mappo)

mappo
fig = go.Figure()



fig.update_layout(

    title='Water Level Outliers',

    yaxis=dict(

        autorange=True,

        showgrid=True,

        zeroline=True,

        dtick=5,

        gridcolor='rgb(255, 255, 255)',

        gridwidth=1,

        zerolinecolor='rgb(255, 255, 255)',

        zerolinewidth=2,

    ),

    margin=dict(

        l=40,

        r=30,

        b=80,

        t=100,

    ),

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

    showlegend=False

)



fig.add_trace(go.Box(x=df1['liv'],

    name="In Red only Suspected Outliers ",

    boxpoints='suspectedoutliers', # only suspected outliers

    marker=dict(

        color='rgb(8,81,156)',

        outliercolor='rgba(219, 64, 82, 0.6)',

        line=dict(

            outliercolor='rgba(219, 64, 82, 0.6)',

            outlierwidth=2)),

    line_color='rgb(8,81,156)'

))



fig.show()
df_month = df1.set_index('datetime').resample('1H').max().reset_index()

df_month['month'] = df_month['datetime'].dt.month





fig = go.Figure()



fig.update_layout(

    title='Water Level Outliers - Monthly',

    yaxis=dict(

        autorange=True,

        showgrid=True,

        zeroline=True,

        dtick=5,

        gridcolor='rgb(255, 255, 255)',

        gridwidth=1,

        zerolinecolor='rgb(255, 255, 255)',

        zerolinewidth=2,

    ),

    margin=dict(

        l=40,

        r=30,

        b=80,

        t=100,

    ),

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

    showlegend=False

)



fig.add_trace(go.Box(x = df_month['month'],y =df_month['liv'],

    name="In Red only Suspected Outliers ",

    boxpoints='suspectedoutliers', # only suspected outliers

    marker=dict(

        color='rgb(8,81,156)',

        outliercolor='rgba(219, 64, 82, 0.6)',

        line=dict(

            outliercolor='rgba(219, 64, 82, 0.6)',

            outlierwidth=2)),

    line_color='rgb(8,81,156)'

))



fig.show()
# fig = px.box(df1, x = 'date', y="liv")

# fig.show()

df_day = df1.set_index('datetime').resample('1H').max().reset_index()

df_day['dayofmonth'] = df_day['datetime'].dt.day



fig = go.Figure()



fig.update_layout(

    title='Water Level Outliers - Daily',

    yaxis=dict(

        autorange=True,

        showgrid=True,

        zeroline=True,

        dtick=5,

        gridcolor='rgb(255, 255, 255)',

        gridwidth=1,

        zerolinecolor='rgb(255, 255, 255)',

        zerolinewidth=2,

    ),

    margin=dict(

        l=40,

        r=30,

        b=80,

        t=100,

    ),

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

    showlegend=False

)



fig.add_trace(go.Box(x = df_day['dayofmonth'],y=df_day['liv'],

    name="In Red only Suspected Outliers ",

    boxpoints='suspectedoutliers', # only suspected outliers

    marker=dict(

        color='rgb(8,81,156)',

        outliercolor='rgba(219, 64, 82, 0.6)',

        line=dict(

            outliercolor='rgba(219, 64, 82, 0.6)',

            outlierwidth=2)),

    line_color='rgb(8,81,156)'

))



fig.show()
df_hour = df1.set_index('datetime').resample('1H').max().reset_index()

df_hour['hour'] = df_hour['datetime'].dt.hour



fig = go.Figure()



fig.update_layout(

    title='Water Level Outliers - Hourly ',

    yaxis=dict(

        autorange=True,

        showgrid=True,

        zeroline=True,

        dtick=5,

        gridcolor='rgb(255, 255, 255)',

        gridwidth=1,

        zerolinecolor='rgb(255, 255, 255)',

        zerolinewidth=2,

    ),

    margin=dict(

        l=40,

        r=30,

        b=80,

        t=100,

    ),

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

    showlegend=False

)



fig.add_trace(go.Box(x = df_hour['hour'],y=df_hour['liv'],

    name="In Red only Suspected Outliers ",

    boxpoints='suspectedoutliers', # only suspected outliers

    marker=dict(

        color='rgb(8,81,156)',

        outliercolor='rgba(219, 64, 82, 0.6)',

        line=dict(

            outliercolor='rgba(219, 64, 82, 0.6)',

            outlierwidth=2)),

    line_color='rgb(8,81,156)'

))



fig.show()
df2 = pd.read_csv('/kaggle/input/venice-high-water-acqua-alta/VeneziaSanGiorgio.csv',delimiter=',')

df2.columns = ['date','time','vv','vmax','dv','tair','um','rs']

# df2.time = df2.time.map(lambda x: x.rstrip('AMP').replace(';',''))



dt = datetime.datetime(2019, 10, 11, 1, 0, 0)

end = datetime.datetime(2020, 1, 12, 1, 0, 0)

step = datetime.timedelta(hours=1)



result = []



while dt < end:

    result.append(dt.strftime('%Y-%m-%d %H:%M:%S'))

    dt += step



df2.insert(0,'datetime',result)

df2.drop(['date','time'],axis=1, inplace=True)
df2.to_csv('df2.csv')

df2.head()
import plotly.graph_objects as go





# Create traces

fig = go.Figure()



fig.add_trace(go.Scatter(x = df2['datetime'], y=df2['vv'],

                    mode='lines',

                    name='vv'))

fig.add_trace(go.Scatter(x = df2['datetime'], y=df2['vmax'],

                         line = dict(color='firebrick', width=2, dash='dot'),

                    name='vmax'))



fig.update_layout(title='Average and Max Wind Speed ',

                   xaxis_title='Datetime',

                   yaxis_title='Wind Velocity')





fig.show()
df2['datetime'] = pd.to_datetime(df2['datetime'])

df_month = df2.set_index('datetime').resample('1H').max().reset_index()

df_month['month'] = df_month['datetime'].dt.month



fig = go.Figure()



fig.update_layout(

    title='Wind Velocity Outliers - Monthly',

#     subtitle='Wind Velocity Outliers',

    boxmode='group',

    yaxis=dict(

        autorange=True,

        showgrid=True,

        zeroline=True,

        dtick=5,

        gridcolor='rgb(255, 255, 255)',

        gridwidth=1,

        zerolinecolor='rgb(255, 255, 255)',

        zerolinewidth=2,

    ),

    margin=dict(

        l=40,

        r=30,

        b=80,

        t=100,

    ),

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

    showlegend=True

)



fig.add_trace(go.Box(x = df_month['month'],y =df_month['vv'],

    name="Wind Mean Velocity Outliers",

    boxpoints='suspectedoutliers', # only suspected outliers

    marker=dict(

        color='rgb(25, 156, 8)',

        outliercolor='rgba(219, 64, 82, 0.6)',

        line=dict(

            outliercolor='rgba(219, 64, 82, 0.6)',

            outlierwidth=2)),

    line_color='rgb(25, 156, 8)'

))



fig.add_trace(go.Box(x = df_month['month'],y =df_month['vmax'],

    name="Wind Max Outliers ",

    boxpoints='suspectedoutliers', # only suspected outliers

    marker=dict(

        color='rgb(156, 151, 8)',

        outliercolor='rgba(219, 64, 82, 0.6)',

        line=dict(

            outliercolor='rgba(219, 64, 82, 0.6)',

            outlierwidth=2)),

    line_color='rgb(156, 151, 8)'

))





fig.show()
df2['datetime'] = pd.to_datetime(df2['datetime'])

df_date = df2.set_index('datetime').resample('1H').max().reset_index()

df_date['dayofmonth'] = df_date['datetime'].dt.day



fig = go.Figure()



fig.update_layout(

    title='Max Wind Velocity Outliers - Daily',

    yaxis=dict(

        autorange=True,

        showgrid=True,

        zeroline=True,

        dtick=5,

        gridcolor='rgb(255, 255, 255)',

        gridwidth=1,

        zerolinecolor='rgb(255, 255, 255)',

        zerolinewidth=2,

    ),

    margin=dict(

        l=40,

        r=30,

        b=80,

        t=100,

    ),

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

    showlegend=False

)



fig.add_trace(go.Box(x = df_date['dayofmonth'],y =df_date['vmax'],

    name="In Red only Suspected Outliers ",

    boxpoints='suspectedoutliers', # only suspected outliers

    marker=dict(

        color='rgb(25, 156, 8)',

        outliercolor='rgba(219, 64, 82, 0.6)',

        line=dict(

            outliercolor='rgba(219, 64, 82, 0.6)',

            outlierwidth=2)),

    line_color='rgb(25, 156, 8)'

))



fig.show()
df3 = pd.read_csv('/kaggle/input/venice-high-water-acqua-alta/burano.csv',delimiter=',')

df4 = pd.read_csv('/kaggle/input/venice-high-water-acqua-alta/malamocco.csv',delimiter=',')

df5 = pd.read_csv('/kaggle/input/venice-high-water-acqua-alta/chioggia.csv',delimiter=',')



df3 = df3.iloc[1:,:]

df4 = df4.iloc[1:,:]

df5 = df5.iloc[1:,:]



df3.columns = ['date','time','liv']

df4.columns = ['date','time','liv']

df5.columns = ['date','time','liv']



df3.time = df3.time.map(lambda x: x.rstrip('AMP').replace(';',''))

df4.time = df4.time.map(lambda x: x.rstrip('AMP').replace(';',''))

df5.time = df5.time.map(lambda x: x.rstrip('AMP').replace(';',''))



df3.liv = df3.liv.fillna(method='pad')

df4.liv = df4.liv.fillna(method='pad')

df5.liv = df5.liv.fillna(method='pad')



dt = datetime.datetime(2019, 10, 11, 0, 10, 0)

end = datetime.datetime(2020, 1, 12, 0, 10, 0)

step = datetime.timedelta(minutes=10)



result = []



while dt < end:

    result.append(dt.strftime('%Y-%m-%d %H:%M:%S'))

    dt += step



df3.insert(0,'datetime',result)

df4.insert(0,'datetime',result)

df5.insert(0,'datetime',result)



df3.to_csv('df3.csv')

df4.to_csv('df4.csv')

df5.to_csv('df5.csv')

df3.head()
import plotly.graph_objects as go





# Create traces

fig = go.Figure()



fig.add_trace(go.Scatter(x = df3['datetime'], y=df3['liv'],

                    mode='lines',

                    name='Burano Station'))



fig.add_trace(go.Scatter(x = df4['datetime'], y=df4['liv'],

                    mode='lines',

                    name='Malamocco Station'))



fig.add_trace(go.Scatter(x = df5['datetime'], y=df5['liv'],

                    mode='lines',

                    name='Chioggia Station'))



fig.update_layout(title='Water Level',

                   xaxis_title='Datetime')





fig.show()
df1 = df1[df1['datetime']>='2019-10-11 00:10:00']

df3 = df3[df3['datetime']<='2020-01-11 00:00:00']

df4 = df4[df4['datetime']<='2020-01-11 00:00:00']

df5 = df5[df5['datetime']<='2020-01-11 00:00:00']

df1 = df1.reset_index().drop('index',axis=1)

df3 = df3.reset_index().drop('index',axis=1)

df4 = df4.reset_index().drop('index',axis=1)

df5 = df5.reset_index().drop('index',axis=1)



df_all_sensors = pd.concat([df1.liv,df3.liv,df4.liv,df5.liv],axis=1)

df_all_sensors = df_all_sensors.astype(float)

df_all_sensors.columns = ['Punta della Salute',

                   'Burano','Malamocco','Chioggia']



df_all_sensors
import plotly.graph_objects as go





fig = go.Figure(data=go.Heatmap(

                z=df_all_sensors.T,

                x=df1['datetime'],

                y= ['Punta della Salute',

                   'Burano','Malamocco','Chioggia'],

                colorscale='Viridis'))



fig.update_layout(title='Water Level between Oct-Jan',xaxis_nticks=36)



fig.show()