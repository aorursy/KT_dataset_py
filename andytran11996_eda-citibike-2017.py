%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly import tools
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
df = pd.read_csv('../input/201701-citibike-tripdata.csv')
df.head(10)
df.isna().sum()
df_wo_na = df.dropna()

df['Birth Year'].apply(lambda y: y + 100 if y < 1918 else y)
df = df.reset_index(drop=True)
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(10, weights='distance')
for col in ['Start Time', 'Stop Time']:
    df_wo_na[col] = df_wo_na[col].apply(lambda x: pd.Timestamp(x).value)
for col in ['Start Station Name', 'End Station Name', 'User Type']:
    df_wo_na[col] = pd.Categorical(df_wo_na[col]).codes
knn.fit(df_wo_na.drop(['Birth Year'], axis=1).values, df_wo_na['Birth Year'].values)
nan_by = df[df['Birth Year'].isnull()]
for col in ['Start Time', 'Stop Time']:
    nan_by[col] = nan_by[col].apply(lambda x: pd.Timestamp(x).value)
for col in ['Start Station Name', 'End Station Name', 'User Type']:
    nan_by[col] = pd.Categorical(nan_by[col]).codes
nan_by = nan_by.drop(['Birth Year'], axis=1)
nan_by['Birth Year'] = knn.predict(nan_by)
df.loc[df['Birth Year'].isnull(), 'Birth Year'] = nan_by['Birth Year'].astype(int)
df['age'] = df['Birth Year'].apply(lambda y: 2017 - y)
df.isna().sum()
df = df.dropna()
df.isna().sum()


customer_type_df = pd.DataFrame(data=df['User Type'].value_counts())
customer_type_df = customer_type_df.reset_index()
customer_type_df.rename(columns={'User Type':'count', 'index': 'type'}, inplace=True)
layout = go.Layout(
    title='User Type',
)
trace = go.Pie(labels=customer_type_df['type'].values, values=customer_type_df['count'].values)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
y = list(range(0, 110, 10))
men_bins = []
women_bins = []
for i in range(0, len(y) - 1):
    df_gender = pd.DataFrame(data=df[( df['age'] >  y[i] ) & (df['age'] < y[i+1]) ]['Gender'].value_counts())
    df_gender = df_gender.reset_index()
    df_gender.rename(columns={'Gender':'count', 'index':'gender'}, inplace=True)
    count = df_gender[df_gender['gender'] == 1]['count']
    men_bins.append(0 if len(count) == 0 else count.values[0])
    count2 = df_gender[df_gender['gender'] == 2]['count']
    women_bins.append(0 if len(count2) == 0 else -count2.values[0])
    
layout = go.Layout(yaxis=go.layout.YAxis(title='Age'),
                  title="Gender",
                   barmode='overlay',
                   bargap=0.1)

data = [
        go.Bar(y=y,
               x=women_bins,
               orientation='h',
               name='Women',
               text=-1 * women_bins,
               hoverinfo='text',
               marker=dict(color='seagreen')
               ),go.Bar(y=y,
               x=men_bins,
               orientation='h',
               name='Men',
               hoverinfo='x',
               marker=dict(color='powderblue')
               )]
iplot(dict(data=data, layout=layout)) 
df['Start Time'] = df['Start Time'].apply(pd.to_datetime)
def extract_part_of_day(hour):
    if hour < 4:
        return 'early morning'
    if hour < 10:
        return 'morning'
    if hour < 14:
        return 'noon'
    if hour < 18:
        return 'afternoon'
    return 'evening'
df['part_of_day'] = df['Start Time'].apply(lambda t: extract_part_of_day(t.hour))
df_station_end = df.groupby(['End Station ID', 'End Station Name', 'End Station Latitude', 'End Station Longitude']).count().reset_index()[['End Station ID', 'End Station Name', 'End Station Latitude', 'End Station Longitude', 'age']]
df_station_end.rename(columns={
    'End Station ID': 'id',
    'End Station Name': 'name',
    'End Station Latitude':'lat',
    'End Station Longitude': 'lon'
}, inplace= True)
df_station_start = df.groupby(['Start Station ID', 'Start Station Name', 'Start Station Latitude', 'Start Station Longitude']).count().reset_index()[['Start Station ID', 'Start Station Name', 'Start Station Latitude', 'Start Station Longitude', 'age']]
df_station_start.rename(columns={
    'Start Station ID': 'id',
    'Start Station Name': 'name',
    'Start Station Latitude':'lat',
    'Start Station Longitude': 'lon'
}, inplace= True)

df_paths = df.groupby(['End Station ID', 'End Station Name', 'End Station Latitude', 'End Station Longitude','Start Station ID', 'Start Station Name', 'Start Station Latitude', 'Start Station Longitude']).count().reset_index()
mapbox_access_token = 'pk.eyJ1IjoiYW5keXRyYW4xMTk5NiIsImEiOiJjam9xeXg2aTMwOWRlM3FvOWk2NDF0N3F4In0.zvrXbjWVMU7dHWHAeLeA4w'
# sk.eyJ1IjoiYW5keXRyYW4xMTk5NiIsImEiOiJjam9yMGFlcW4wOW10M3hucmwwNm83bTJkIn0.z199ESfbVcWieB3qiOv67A
data = []
data.append(go.Scattermapbox(
        lat=df_station_start['lat'].values,
        lon=df_station_start['lon'].values,
        mode='markers',
        marker=dict(
                size=9
        ),
         text=df_station_start['name'].values
    ))
for i in range(len(df_paths)//2 - 1, len(df_paths)//2-100, -1):
     data.append(go.Scattermapbox(
        lat=[df_paths['Start Station Latitude'][i], df_paths['End Station Latitude'][i]],
        lon=[ df_paths['Start Station Longitude'][i], df_paths['End Station Longitude'][i]],
        mode='lines',
        line = dict(
                width = 1,
                color = 'red',
            ),
    ))
layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=40.76,
            lon=-73.99
        ),
        pitch=0,
        zoom=12
    ),
    showlegend = False
)


fig = dict(data=data, layout=layout)
iplot(fig, filename='Multiple Mapbox')
sns.pairplot(df[['age','Trip Duration', 'Start Station ID', 'End Station ID','Gender']])

df.head(5)
day_parts = ['early morning','morning', 'noon', 'afternoon','evening']
fig = tools.make_subplots(rows=1, cols=5, subplot_titles=day_parts)

df_station_end = df.groupby(['End Station ID', 'End Station Name', 'End Station Latitude', 'End Station Longitude','part_of_day']).count().reset_index()[['End Station ID', 'End Station Name', 'End Station Latitude', 'End Station Longitude', 'age','part_of_day']]
df_station_end.rename(columns={
    'End Station ID': 'id',
    'End Station Name': 'name',
    'End Station Latitude':'lat',
    'End Station Longitude': 'lon'
}, inplace= True)
df_station_start = df.groupby(['Start Station ID', 'Start Station Name', 'Start Station Latitude', 'Start Station Longitude','part_of_day']).count().reset_index()[['Start Station ID', 'Start Station Name', 'Start Station Latitude', 'Start Station Longitude', 'age','part_of_day']]
df_station_start.rename(columns={
    'Start Station ID': 'id',
    'Start Station Name': 'name',
    'Start Station Latitude':'lat',
    'Start Station Longitude': 'lon'
}, inplace= True)


for idx, daypart in enumerate(day_parts):
    df_start_top10 = df_station_start[df_station_start['part_of_day'] == daypart].sort_values(['age'], ascending=False).head(10)
    trace = go.Bar(
            x=df_start_top10.name,
            y=df_start_top10.age
    )
    fig.append_trace(trace, 1, idx + 1)

fig['layout'].update(title='Top 10 start station')

iplot(fig)

fig2 = tools.make_subplots(rows=1, cols=5, subplot_titles=day_parts)
for idx, daypart in enumerate(day_parts):
    df_start_top10 = df_station_end[df_station_end['part_of_day'] == daypart].sort_values(['age'], ascending=False).head(10)
    trace = go.Bar(
            x=df_start_top10.name,
            y=df_start_top10.age
    )
    fig2.append_trace(trace, 1, idx + 1)

fig2['layout'].update(title='Top 10 end station')

iplot(fig2)