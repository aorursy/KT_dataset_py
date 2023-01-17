import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go
!ls ../input/data.csv
df = pd.read_csv('../input/data.csv', parse_dates=["from_date"])



# list first few rows

df.head()
# check datatypes

df.dtypes
# check statistics of the features

df.describe()
print(df.isnull().sum())
# minimum and maximum longitude

min_long = min(df.from_long.min(), df.to_long.min())

max_long = max(df.from_long.max(), df.to_long.max())
# minimum and maximum lattitude

min_lat = min(df.from_lat.min(), df.to_lat.min())

max_lat = max(df.from_lat.max(), df.to_lat.max())
df.head()
df['count'] = 1
accesstoken = "pk.eyJ1IjoidGFubWF5cGFuZGV5NyIsImEiOiJjazE4NDAyM3cwNTVtM210Y3FibnBuamhwIn0.yzvv5xRDq657q45CDihkeA"
fig = go.Figure(go.Densitymapbox(lat=df.from_lat, lon=df.from_long, radius=10, 

                                 colorscale=[[0, 'rgb(255,255,0)'], [1, 'rgb(178,34,34)']]))

fig.update_layout(mapbox_accesstoken=accesstoken)

fig.update_layout(mapbox_style="streets", mapbox_center_lon=77.6, mapbox_center_lat=13, 

                  mapbox_zoom=9.4)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
by_area_id = df.groupby('from_area_id').count()['count'].sort_values(ascending = False).reset_index()

by_area_id_top_10 = by_area_id[:10]

by_area_id_top_10
by_area_id_top_10.from_area_id.to_list()
lat_long_avg = df.groupby('from_area_id').mean()[['from_lat','from_long']].reset_index()

lat_long_avg[lat_long_avg.from_area_id.isin(by_area_id_top_10.from_area_id.to_list())]
fig = go.Figure(data=[go.Pie(labels=by_area_id_top_10['from_area_id'], values=by_area_id_top_10['count'],

                            )])

fig.update_layout(title_text='Top 10 Areas with Most Pickups in Bangalore')

fig.show()
df['time'] = df.from_date.dt.time
time_vs_count = df.groupby("time").count()["count"].reset_index()
time_vs_count.rename(columns={"user_id": "count"}, inplace=True)
time_vs_count.head()
days_vs_count = df.groupby("from_area_id").count()["count"].reset_index()

days_vs_count.sort_values("count", ascending=False, inplace=True)

days_vs_count.head(10)
area_time_count = df.groupby(["from_area_id","time"]).count()["user_id"].reset_index()

area_time_count.sort_values("user_id", ascending=False).head(10)
area_time_count_no_outlier = area_time_count[area_time_count.from_area_id != 393.0]

area_time_count_no_outlier.sort_values("time", inplace=True)
area_time_count_outlier = area_time_count[area_time_count.from_area_id == 393.0]

area_time_count_outlier.sort_values("user_id", ascending=False).head(10)
fig = go.Figure()

fig.add_trace(go.Scatter(x=area_time_count_outlier.time, y=area_time_count_outlier.user_id, 

                         name="Total Kempegowda Pickups",

                         line_color='deepskyblue'))



# fig.add_trace(go.Scatter(x=df.Date, y=df['AAPL.Low'], name="AAPL Low",

#                          line_color='dimgray'))



fig.update_layout(title_text='Total Kempegowda Pickups',

                  xaxis_rangeslider_visible=True)

fig.show()
df.head()

top_10_areas = by_area_id.from_area_id.to_list()

area_time_count.head()

area_time_count_20_no_airport = area_time_count.loc[(area_time_count.user_id>20) & 

                    (area_time_count.from_area_id!=393.0)].sort_values(by='time')
fig = px.scatter(area_time_count_20_no_airport, x="time", y="user_id",

            size="user_id", color="from_area_id")
fig.update_layout(title_text='Areas with total count>20 excluding Kempegowda',  xaxis_title="Time", yaxis_title="Count")

fig.show()

area_time_count_top_10 = area_time_count[area_time_count.from_area_id.isin(top_10_areas)]

area_time_count_top_10 = area_time_count_top_10.sort_values(by="time", ascending=True)

fig = px.scatter(area_time_count_top_10, x="time", y="user_id",

            size="user_id", color="from_area_id"

                 )



fig.update_layout(title_text='Top 10 Popular Areas Pickups',  xaxis_title="Time", yaxis_title="Count")

fig.show()
area_time_count_top_10.head()

area_time_count_top_10_airport = area_time_count_top_10.loc[(area_time_count_top_10["user_id"]>20) &  

                                                       (area_time_count_top_10["from_area_id"]==393)]
fig = px.scatter(area_time_count_top_10_airport, x="time", y="user_id",

            size="user_id", color="from_area_id"

                 )

fig.update_layout(title_text='Kempegowda - Total Pickups > 20',  xaxis_title="Time", yaxis_title="Count", showlegend=False)

fig.update_layout(showlegend=False)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=area_time_count_top_10_airport.time, y=area_time_count_top_10_airport.user_id, 

                         name="Total Kempegowda Pickups",

                         line_color='deepskyblue'))



fig.update_layout(title_text='Kempegowda - Total Pickups > 20',

                  xaxis_rangeslider_visible=True)

fig.show()
area_time_count_top_10
area_time_count_top_10_others = area_time_count_top_10.loc[(area_time_count_top_10["user_id"]>20) &  

                                                       (area_time_count_top_10["from_area_id"]!=393)]
area_time_count_top_10_others
fig = px.scatter(area_time_count_top_10_others, x="time", y="user_id",

            size="user_id", color="from_area_id")

fig.update_layout(title_text='Total Pickups excluding Kempegowda')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=area_time_count_top_10_others.time, y=area_time_count_top_10_others.user_id, 

                         name="Excluding Kempegowda - Total Pickups > 20",

                         line_color='green'))



# fig.add_trace(go.Scatter(x=df.Date, y=df['AAPL.Low'], name="AAPL Low",

#                          line_color='dimgray'))



fig.update_layout(title_text='Excluding Kempegowda - Total Pickups > 20',

                  xaxis_rangeslider_visible=True)

fig.show()