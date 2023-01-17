import plotly.offline as ply
import plotly.graph_objs as go
from plotly.tools import make_subplots

ply.init_notebook_mode(connected=True)

import pandas as pd
import numpy as np

station = pd.read_csv('../input/station.csv')
trip = pd.read_csv('../input/trip.csv')
weather = pd.read_csv('../input/weather.csv')
import colorlover as cl
from IPython.display import HTML

chosen_colors=cl.scales['5']['qual']['Paired']
print('The color palette chosen for this notebook is:')
HTML(cl.to_html(chosen_colors))
station.head()
citygroup=station.groupby(['city'])
temp_df1=citygroup['id'].count().reset_index().sort_values(by='id', ascending=False)
temp_df2=citygroup['dock_count'].sum().reset_index().sort_values(by='dock_count', ascending=False)
trace1 = go.Bar(
    x=temp_df1.city,
    y=temp_df1.id,
    name='No. of stations',
    text=temp_df1.id,
    textposition='outside',
    marker=dict(
        color=chosen_colors[0]
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of bike stations in Bay Area cities',
    xaxis=dict(
        title='City'
    ),
    yaxis=dict(
        title='No. of bike stations'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trace2 = go.Bar(
    x=temp_df1.city,
    y=temp_df1.id.cumsum().shift(1),
    #name='No. of stations',
    hoverinfo=None,
    marker=dict(
        color='rgba(1,1,1,0.0)'
    )
)
trace1 = go.Bar(
    x=temp_df1.city,
    y=temp_df1.id,
    name='No. of stations',
    text=temp_df1.id,
    textposition='outside',
    marker=dict(
        color=chosen_colors[0]
    )
)


data=[trace2, trace1]

layout = go.Layout(
    title='No. of bike stations in Bay Area cities',
    xaxis=dict(
        title='City'
    ),
    yaxis=dict(
        title='No. of bike stations'
    ),
    barmode='stack',
    hovermode='closest',
    showlegend=False
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trace2 = go.Bar(
    x=temp_df2.city,
    y=temp_df2.dock_count,
    name='No. of docks',
    text=temp_df2.dock_count,
    textposition='auto',
    marker=dict(
        color=chosen_colors[1]
    )
)

data=[trace1, trace2]

figure = go.Figure(data=data, layout=layout)

figure['layout'].update(dict(title='No. of bike stations and docks in Bay Area cities'), barmode='group')

ply.iplot(figure)
trip.head()
trip.start_date=pd.to_datetime(trip.start_date,infer_datetime_format=True)
trip.end_date=pd.to_datetime(trip.end_date,infer_datetime_format=True)
#For some reason, the add_datepart function that I imported through fastai library in Kaggle does not have the 'time' argument which also extracts the time details such as the hour and minute
#Hence, I am copying the code from Github and pasting it here for using in this notebook.
import re
import datetime
def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
add_datepart(trip, 'start_date', drop=False, time=True)
add_datepart(trip, 'end_date', drop=False, time=True)
trip['duration_min']=trip.duration/60
trace1 = go.Histogram(
    x=trip[trip.duration_min<60].duration_min, #To remove outliers
    marker=dict(
        color=chosen_colors[0]
    )    
)

data=[trace1]

layout = go.Layout(
    title='Distribution of bike trip duration in Bay Area',
    xaxis=dict(
        title='Trip Duration (minutes)'
    ),
    yaxis=dict(
        title='Count'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Histogram(
            x=trip[(trip.subscription_type==trace_names[i]) & (trip.duration_min<60)].duration_min,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            ),
            opacity=0.5
        )
    )

layout = go.Layout(
    title='Distribution of bike trip duration in Bay Area',
    barmode='overlay',
    xaxis=dict(
        title='Trip Duration (minutes)'
    ),
    yaxis=dict(
        title='Count'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trace_names=['Subscriber', 'Customer']

figure=make_subplots(rows=2, cols=1, subplot_titles = ['Trip duration (minutes) - '+i for i in trace_names])


for i in range(2):
    figure.append_trace(
        go.Histogram(
            x=trip[(trip.subscription_type==trace_names[i]) & (trip.duration_min<60)].duration_min,
            name=trace_names[i],
            showlegend=False,
            marker=dict(
                color=chosen_colors[i]
            )
        ),
        i+1, 1
    )

figure['layout'].update(
    height=1000,
    title='Distribution of trip duration by subscription type', 
    xaxis1=dict(title='Duration'),
    xaxis2=dict(title='Duration'),
    yaxis1=dict(title='Count'),
    yaxis2=dict(title='Count'),
)

ply.iplot(figure)
trip_count_by_month=trip.groupby(['start_Year','start_Month'])['id'].count().reset_index()
trace1 = go.Bar(
    x=trip_count_by_month.start_Month.astype(str)+'-'+trip_count_by_month.start_Year.astype(str),
    y=trip_count_by_month.id,
    name='No. of trips',
    marker=dict(
        color=chosen_colors[0]
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of bike trips by month',
    xaxis=dict(
        title='Month'
    ),
    yaxis=dict(
        title='No. of bike trips'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trip_count_by_month_sub=trip.groupby(['start_Year','start_Month', 'subscription_type'])['id'].count().reset_index()
data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    temp_df=trip_count_by_month_sub[(trip_count_by_month_sub.subscription_type==trace_names[i])]
    data.append(
        go.Bar(
            x=temp_df.start_Month.astype(str)+'-'+temp_df.start_Year.astype(str),
            y=temp_df.id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='No. of trips per month in Bay Area cities',
    xaxis=dict(
        title='Month'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    barmode='stack'
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trip['start_date_dt']=[i.date() for i in trip.start_date]
trip['end_date_dt']=[i.date() for i in trip.end_date]
trip_count_by_date=trip.groupby(['start_date_dt'])['id'].count().reset_index()

trip_count_by_date['day_of_week']=[i.weekday() for i in trip_count_by_date.start_date_dt]

trip_count_by_date['is_weekend'] = (trip_count_by_date.day_of_week>4)*1

data=[]

trace_names=['Weekday', 'Weekend']

for i in range(2):
    data.append(
        go.Bar(
            x=trip_count_by_date[(trip_count_by_date.is_weekend==i) & (trip_count_by_date.start_date_dt<datetime.date(2014, 1, 1))].start_date_dt,
            y=trip_count_by_date[(trip_count_by_date.is_weekend==i)  & (trip_count_by_date.start_date_dt<datetime.date(2014, 1, 1))].id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='No. of trips per day in Bay Area cities',
    xaxis=dict(
        title='Date'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trip_count_by_date_sub=trip.groupby(['start_date_dt', 'subscription_type'])['id'].count().reset_index()

trip_count_by_date_sub['day_of_week']=[i.weekday() for i in trip_count_by_date_sub.start_date_dt]

trip_count_by_date_sub['is_weekend'] = (trip_count_by_date_sub.day_of_week>4)*1
data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Bar(
            x=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].start_date_dt,
            y=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='No. of trips per day in Bay Area cities',
    xaxis=dict(
        title='Date'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    barmode='stack'
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
def calc_percent(row, col):
    dt=row[col]
    total=trip_count_by_date[trip_count_by_date[col]==dt].id.iloc[0]
    count=row['id']
    
    return count*1./total*100
trip_count_by_date_sub['percent']=trip_count_by_date_sub.apply(lambda row: calc_percent(row, 'start_date_dt'), axis=1)
data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Bar(
            x=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].start_date_dt,
            y=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].percent,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='Percentage of trips per day in Bay Area cities',
    xaxis=dict(
        title='Date'
    ),
    yaxis=dict(
        title='% of trips',
        ticksuffix='%'
    ),
    barmode='stack'
    #hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
data=[]

trace_names=['Subscriber', 'Customer']
trace_names1=['weekdays', 'weekends']

weekend=[0,1]

for i in range(2):
    for j in range(2):
        data.append(
            go.Bar(
                x=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & 
                                         (trip_count_by_date_sub.is_weekend==weekend[j]) &
                                         (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].start_date_dt,
                y=trip_count_by_date_sub[(trip_count_by_date_sub.subscription_type==trace_names[i]) & 
                                         (trip_count_by_date_sub.is_weekend==weekend[j]) &
                                         (trip_count_by_date_sub.start_date_dt<datetime.date(2014, 1, 1))].percent,
                name=trace_names[i]+' on '+trace_names1[j],
                marker=dict(
                    color=chosen_colors[i*2+j]
                )
            )
        )

layout = go.Layout(
    title='Percentage of trips per day in Bay Area cities',
    xaxis=dict(
        title='Date'
    ),
    yaxis=dict(
        title='% of trips',
        ticksuffix='%'
    ),
    barmode='stack',
    #hovermode='closest',
    legend=dict(
        orientation="h",
        x=0,
        y=1.1
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trip_count_by_DOW=trip.groupby(['start_Dayofweek'])['id'].count().reset_index()

DOW=[
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]

data=[
    go.Bar(
        x=DOW,
        y=trip_count_by_DOW['id'],
        name='No. of trips',
        marker=dict(
            color=chosen_colors[0]
        )
    )
]

layout = go.Layout(
    title='No. of bike trips by day of week in Bay Area cities',
    xaxis=dict(
        title='Day of week'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    #hovermode='closest',
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trip_count_by_DOW_sub=trip.groupby(['start_Dayofweek','subscription_type'])['id'].count().reset_index()

data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Bar(
            x=DOW,
            y=trip_count_by_DOW_sub[(trip_count_by_DOW_sub.subscription_type==trace_names[i])].id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='No. of bike trips by day of week in Bay Area cities',
    xaxis=dict(
        title='Day of week'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
def calc_percent1(row, col):
    dt=row[col]
    total=trip_count_by_DOW[trip_count_by_DOW[col]==dt].id.iloc[0]
    count=row['id']
    
    return count*1./total*100
trip_count_by_DOW_sub['percent']=trip_count_by_DOW_sub.apply(lambda row:calc_percent1(row, 'start_Dayofweek'), axis=1)
data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    data.append(
        go.Bar(
            x=DOW,
            y=trip_count_by_DOW_sub[trip_count_by_DOW_sub.subscription_type==trace_names[i]].percent,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='Percentage of trips by day of week in Bay Area cities',
    xaxis=dict(
        title='Day of week',
    ),
    yaxis=dict(
        title='% of trips',
        ticksuffix='%'
    ),
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trip_count_by_hour=trip.groupby(['start_date_dt','start_Hour'])['id'].count().reset_index()
temp_df=trip_count_by_hour[(trip_count_by_hour.start_date_dt>datetime.date(2013, 11, 30)) & (trip_count_by_hour.start_date_dt<datetime.date(2013, 12, 8))]
trace1 = go.Bar(
    x=temp_df.start_Hour.astype(str)+'                     '+temp_df.start_date_dt.astype(str),
    #x=temp_df.start_Hour.astype(str),
    y=temp_df.id,
    text=['Hour:'+str(i) for i in temp_df.start_Hour],
    name='No. of trips',
    marker=dict(
        color=chosen_colors[0]
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of bike trips by hour',
    xaxis=dict(
        title='Hour',
        categoryorder='array',
        categoryarray=temp_df.start_Hour,
        type='category'
    ),
    yaxis=dict(
        title='No. of bike trips'
    ),
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
temp_df=trip.groupby(['start_Hour'])['id'].count().reset_index()

trace1 = go.Bar(
    x=temp_df.start_Hour,
    y=temp_df.id,
    name='No. of trips',
    marker=dict(
        color=chosen_colors[0]
    )
)

data=[trace1]

layout = go.Layout(
    title='No. of bike trips by hour',
    xaxis=dict(
        title='Hour',
    ),
    yaxis=dict(
        title='No. of bike trips'
    ),
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trip_count_by_hour_sub=trip.groupby(['start_Hour','subscription_type'])['id'].count().reset_index()
#temp_df=trip_count_by_hour_sub[(trip_count_by_hour_sub.start_date_dt>datetime.date(2013, 11, 30)) & (trip_count_by_hour_sub.start_date_dt<datetime.date(2013, 12, 8))]
temp_df=trip_count_by_hour_sub.copy()
data=[]

trace_names=['Subscriber', 'Customer']

for i in range(2):
    temp_df1=temp_df[(temp_df.subscription_type==trace_names[i])]
    data.append(
        go.Bar(
            x=temp_df1.start_Hour,
            y=temp_df1.id,
            name=trace_names[i],
            marker=dict(
                color=chosen_colors[i]
            )
        )
    )

layout = go.Layout(
    title='No. of trips per hour in Bay Area cities',
    xaxis=dict(
        title='Hour'
    ),
    yaxis=dict(
        title='No. of trips'
    ),
    barmode='stack'
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trip_count_start_station=trip.groupby(['start_station_name']).id.count().reset_index().sort_values(by='id', ascending=False)
trip_count_end_station=trip.groupby(['end_station_name']).id.count().reset_index().sort_values(by='id', ascending=False)
trace1 = go.Bar(
    x=trip_count_start_station[:10].start_station_name,
    y=trip_count_start_station[:10].id,
    name='No. of trips starting',
    marker=dict(
        color=chosen_colors[0]
    )
)

trace2 = go.Bar(
    x=trip_count_start_station[:10].start_station_name,
    y=trip_count_end_station[trip_count_end_station.end_station_name.isin(trip_count_start_station[:10].start_station_name)].id,
    name='No. of trips ending',
    marker=dict(
        color=chosen_colors[1]
    )
)

data=[trace1, trace2]

layout = go.Layout(
    title='No. of bike trips by starting station',
    xaxis=dict(
        title='Station name',
    ),
    yaxis=dict(
        title='No. of bike trips'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
trace1 = go.Bar(
    x=trip_count_start_station[:10].id,
    y=trip_count_start_station[:10].start_station_name,
    orientation='h',
    name='No. of trips starting',
    marker=dict(
        color=chosen_colors[0]
    )
)

trace2 = go.Bar(
    x=trip_count_end_station[trip_count_end_station.end_station_name.isin(trip_count_start_station[:10].start_station_name)].id,
    y=trip_count_start_station[:10].start_station_name,
    orientation='h',
    name='No. of trips ending',
    marker=dict(
        color=chosen_colors[1]
    )
)

data=[trace1, trace2]

layout = go.Layout(
    title='No. of bike trips by starting station',
    yaxis=dict(
        title='Station name',
    ),
    xaxis=dict(
        title='No. of bike trips'
    ),
    margin=dict(
        l=350
    ),
    legend=dict(
        orientation='h',
        x=0,
        y=1.1
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)