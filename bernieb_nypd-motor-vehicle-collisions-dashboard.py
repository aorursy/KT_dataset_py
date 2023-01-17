import pandas as pd

import matplotlib.pyplot as plt

import folium

import datetime

#pk.eyJ1IjoiYmVybmllMzE4IiwiYSI6ImNqcjAyaGtnOTA1dnU0NmxqY3ltMTZvZjEifQ.2CfkEzCdGvgewTPsIDJaWg

#for second map in plotly



import plotly.plotly as py

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
df = pd.read_csv("../input/nypd-motor-vehicle-collisions.csv", dtype=str)
#df.info()
idx = df['LATITUDE'].isna() | (df['LATITUDE'] == '0.0000000')
df['time'] = pd.to_datetime(df['DATE'] + " " + df['TIME'])

df.drop(['DATE', 'TIME'], axis=1, inplace=True)
df.columns
numeric_columns = ['LATITUDE',

                   'LONGITUDE',

                   'NUMBER OF PERSONS INJURED',

                   'NUMBER OF PERSONS KILLED',

                   'NUMBER OF PEDESTRIANS INJURED',

                   'NUMBER OF PEDESTRIANS KILLED',

                   'NUMBER OF CYCLIST INJURED',

                   'NUMBER OF CYCLIST KILLED',

                   'NUMBER OF MOTORIST INJURED',

                   'NUMBER OF MOTORIST KILLED']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce') 

#coerce sets it at NaN 
df.drop(['LOCATION', 'UNIQUE KEY'], axis=1, inplace=True)
df.info()
df1 = df.loc[~idx] # where LAT/LON are known 

df1.head()

df1.info()
#min_lat, max_lat = df1['LATITUDE'].min(), df1['LATITUDE'].max()

#min_lon, max_lon = df1['LONGITUDE'].min(), df1['LONGITUDE'].max()
#print(min_lat, max_lat, min_lon, max_lon)
#now = datetime.datetime.now()

now = datetime.datetime(2018, 12, 15, 18, 00)

#this is fussy! do not know why but check and make sure stage_df shape is not 0 and not too large (down below)

earliest_date = now - datetime.timedelta(hours=24*38)

#print(year_ago)

print(now)

#print(datetime.timedelta((hours=24*38)))

print(earliest_date)
stage_df = df1[df1['time'] > earliest_date]
stage_df.shape[0]

#stage_df.head()
#import gc

#gc.collect()
# Create a Map instance

m = folium.Map(location=[40.6971494,-74.2598745], tiles='Stamen Toner',

                   zoom_start=10, control_scale=True)



from folium.plugins import MarkerCluster



mc = MarkerCluster()



for each in stage_df.iterrows():

    mc.add_child(folium.Marker(

        location = [each[1]['LATITUDE'],each[1]['LONGITUDE']])) #, 

        #clustered_marker = True)



m.add_child(mc)



display(m)
my_style = 'mapbox://styles/bernie318/cjr02ieyu42bo2rjycac488jd'

mapbox_access_token = 'pk.eyJ1IjoiYmVybmllMzE4IiwiYSI6ImNqcjAyaGtnOTA1dnU0NmxqY3ltMTZvZjEifQ.2CfkEzCdGvgewTPsIDJaWg'




import warnings

#warnings.filterwarnings('ignore')



collisions = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')



# Convert to datetime format

collisions['date_parsed'] = pd.to_datetime(collisions['DATE'], format="%Y-%m-%d")

collisions_by_month = collisions['date_parsed'].value_counts().resample('m').sum()



locations = go.Scattermapbox(

    lat=collisions['LATITUDE'][:10001],

    lon=collisions['LONGITUDE'][:10001],

    mode='markers',

    marker=dict(

        size=4,

        color='gold',

        opacity=0.8

    ),

    text=('Date: '+collisions['date_parsed'][:10001].astype(str)+

          '</br>Injured: '+collisions['NUMBER OF PERSONS INJURED'][:10001].astype(str)+

          '</br>Killed: '+collisions['NUMBER OF PERSONS KILLED'][:10001].astype(str)

    ),

    name='locations'

)



dates = go.Scatter(

    x=collisions_by_month.index,

    y=collisions_by_month.values,

    line=dict(color='gold'),

    name='dates'

)



data = [locations, dates]



layout = dict(

    title='NYPD Motor Vehicle Collisions',

    titlefont=dict(

        size=20,

        family="Raleway, Roman, Arial"

    ),

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        domain=dict(x = [0, 0.55],

#                     y= [0, 0.9]

        ),

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=40.7,

            lon=-73.9

        ),

        pitch=0,

        zoom=8.5,

        style=my_style,

    ),

    xaxis = dict(

        domain = [0.6, 1]

    ),

#     yaxis = dict(

#         domain = [0, 0.9]

#     )

)



annotations =  [

    {

      "x": 0.3, 

      "y": 1.0, 

      "font": {"size": 12, "family":"Raleway, Roman, Arial"}, 

      "showarrow": False, 

      "text": "10K Most Recent", 

      "xanchor": "center", 

      "xref": "paper", 

      "yanchor": "bottom", 

      "yref": "paper"

    }, 

    {

      "x": 0.8, 

      "y": 1.0, 

      "font": {"size": 12, "family":"Raleway, Roman, Arial"}, 

      "showarrow": False, 

      "text": "By Month", 

      "xanchor": "center", 

      "xref": "paper", 

      "yanchor": "bottom", 

      "yref": "paper"

    }

]



layout['annotations'] = annotations



fig = dict(data=data, layout=layout)

fig['layout'].update(showlegend=False, plot_bgcolor='black', paper_bgcolor='black', font=dict(color= 'white'))



iplot(fig)
