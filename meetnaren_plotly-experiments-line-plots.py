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



chosen_colors=cl.scales['7']['qual'][np.random.choice(list(cl.scales['7']['qual'].keys()))]



print('The color palette chosen for this notebook is:')

HTML(cl.to_html(chosen_colors))
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

    attr = ['Year', 'Month', 'Week', 'Day', 'Date', 'Dayofweek', 'Dayofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    if time: attr = attr + ['Hour', 'Minute', 'Second']

    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

    if drop: df.drop(fldname, axis=1, inplace=True)
add_datepart(trip, 'start_date', drop=False, time=True)
add_datepart(trip, 'end_date', drop=False, time=True)
trip.head()
trip_count_by_date=trip.groupby(['start_Date'])['id'].count().reset_index()
trace1 = go.Scatter(

    x=trip_count_by_date.start_Date,

    y=trip_count_by_date.id,

    mode='lines',

    line=dict(

        color=chosen_colors[0]

    ),

    name='Daily'

)



data=[trace1]



layout = go.Layout(

    title='No. of trips per day in Bay Area cities',

    xaxis=dict(

        title='Date',

        type='date',

        showgrid=False

    ),

    yaxis=dict(

        title='No. of trips'

    ),

    hovermode='closest',

)



figure = go.Figure(data=data, layout=layout)



ply.iplot(figure)
layout
figure['layout'].update(

    xaxis=dict(

        rangeslider=dict(

            visible = True

        ),

        type='date'

    )

)



ply.iplot(figure)
trip_count_by_date['weekly_avg']=trip_count_by_date.rolling(window=7, center=True)['id'].mean()

trip_count_by_date['monthly_avg']=trip_count_by_date.rolling(window=30, center=True)['id'].mean()

trip_count_by_date['quarterly_avg']=trip_count_by_date.rolling(window=90, center=True)['id'].mean()



trace2 = go.Scatter(

    x=trip_count_by_date.start_Date,

    y=trip_count_by_date.weekly_avg,

    mode='lines',

    line=dict(

        color=chosen_colors[1],

    ),

    name='Weekly'

)



trace3 = go.Scatter(

    x=trip_count_by_date.start_Date,

    y=trip_count_by_date.monthly_avg,

    mode='lines',

    line=dict(

        color=chosen_colors[3],

    ),

    name='Monthly'

)



trace4 = go.Scatter(

    x=trip_count_by_date.start_Date,

    y=trip_count_by_date.quarterly_avg,

    mode='lines',

    line=dict(

        color=chosen_colors[5],

    ),

    name='Quarterly'

)



data=[trace1, trace2, trace3, trace4]



figure = go.Figure(data=data, layout=layout)



figure['layout'].update(

    legend=dict(

        orientation="h",

        x=0,

        y=1.1

    ),

    xaxis=dict(

        rangeslider=dict(

            visible = True

        ),

        type='date',

    ),

)



ply.iplot(figure)
trip_count_q2_2014=trip[(trip.start_Date>=datetime.date(2014,4,1)) & (trip.start_Date<datetime.date(2014,7,1))].groupby(['start_Date'])['id'].count().reset_index()



trace1 = go.Scatter(

    x=trip_count_q2_2014.start_Date,

    y=trip_count_q2_2014.id,

    mode='markers+lines',

    line=dict(

        color=chosen_colors[0]

    ),

    marker=dict(

        color=chosen_colors[1],

    )

)



data=[trace1]



figure = go.Figure(data=data, layout=layout)



ply.iplot(figure)
trip_count_q2_2014['day_of_week']=[i.weekday() for i in trip_count_q2_2014.start_Date]



trip_count_q2_2014['is_weekend'] = (trip_count_q2_2014.day_of_week>4)*1



trace1 = go.Scatter(

    x=trip_count_q2_2014.start_Date,

    y=trip_count_q2_2014.id,

    mode='lines',

    line=dict(

        color=chosen_colors[-1]

    ),

    showlegend=False

)



data=[trace1]



trace_names=['Weekday', 'Weekend']



for i in range(2):

    data.append(

        go.Scatter(

            x=trip_count_q2_2014[trip_count_q2_2014.is_weekend==i].start_Date,

            y=trip_count_q2_2014[trip_count_q2_2014.is_weekend==i].id,

            mode='markers',

            marker=dict(

                color=chosen_colors[i]

            ),

            name=trace_names[i]

        )

    )



figure = go.Figure(data=data, layout=layout)



ply.iplot(figure)
trace1 = go.Scatter(

    x=trip_count_q2_2014.start_Date,

    y=trip_count_q2_2014.id,

    mode='markers+lines',

)



shapes=[]

weekend_dates=trip_count_q2_2014[trip_count_q2_2014.is_weekend==1].start_Date



box_color=chosen_colors[-1]



for i in weekend_dates[::2]:

    shapes.append(

        {

            'type':'rect',

            'xref':'x',

            'x0':i,

            'x1':i+datetime.timedelta(days=1),

            'yref':'paper',

            'y0':0,

            'y1':1,

            'fillcolor': box_color,

            'opacity':0.15,

            'line': {

                'width': 0,

            }

        }

    )



data=[trace1]



figure = go.Figure(data=data, layout=layout)



figure['layout'].update(

    shapes=shapes

)



ply.iplot(figure)
trace1 = go.Scatter(

    x=trip_count_q2_2014.start_Date,

    y=trip_count_q2_2014.id,

    mode='lines',

    line=dict(

        width=0.0,

        color=chosen_colors[0]

    ),

    fill='tozeroy',

)



data=[trace1]



figure = go.Figure(data=data, layout=layout)



ply.iplot(figure)
trip_count_q2_2014_sub=trip[(trip.start_Date>=datetime.date(2014,4,1)) & (trip.start_Date<datetime.date(2014,7,1))].groupby(['start_Date', 'subscription_type'])['id'].count().reset_index()



data=[]



trace_names=['Subscriber', 'Customer']

fillmodes=[None, 'tonexty']



for i in range(2):

    data.append(

        go.Scatter(

            x=trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].start_Date,

            y=trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].id,

            name=trace_names[i],

            line=dict(

                width=0.0,

                color=chosen_colors[i]

            ),

            fill='tozeroy',

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

    legend=dict(

        orientation='h',

        x=0,

        y=1.1

    )

    #hovermode='closest',

    #showlegend=True

)



figure = go.Figure(data=data, layout=layout)



ply.iplot(figure)
def calc_total(row):

    if row.subscription_type=='Customer':

        return trip_count_q2_2014_sub[(trip_count_q2_2014_sub.start_Date==row.start_Date) & (trip_count_q2_2014_sub.subscription_type=='Subscriber')].id.iloc[0]+row.id

    else:

        return row.id



trip_count_q2_2014_sub['total']=trip_count_q2_2014_sub.apply(lambda row:calc_total(row), axis=1)
data=[]



trace_names=['Subscriber', 'Customer']

fillmodes=['tozeroy', 'tonexty']



for i in range(2):

    data.append(

        go.Scatter(

            x=trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].start_Date,

            y=trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].total,

            name=trace_names[i],

            line=dict(

                color=chosen_colors[i],

                width=0.0

            ),

            fill=fillmodes[i],

            hoverinfo='text',

            text=[trace_names[i]+': '+k for k in trip_count_q2_2014_sub[trip_count_q2_2014_sub.subscription_type==trace_names[i]].id.astype(str)]

        )

    )



layout = go.Layout(

    title='No. of trips per day in Bay Area cities',

    xaxis=dict(

        title='Date',

        showgrid=False

    ),

    yaxis=dict(

        title='No. of trips'

    ),

    legend=dict(

        orientation='h',

        x=0,

        y=1.1

    )

)



figure = go.Figure(data=data, layout=layout)



ply.iplot(figure)
fillmodes[0]=None

for i in range(2): figure['data'][i].update(fill=fillmodes[i])

ply.iplot(figure)
trace1 = go.Scatter(

    x=trip_count_q2_2014.start_Date,

    y=trip_count_q2_2014.id,

    mode='markers+lines',

    line=dict(

        color=chosen_colors[0],

        shape='hv'

    ),

    marker=dict(

        color=chosen_colors[1]

    )

)



data=[trace1]



figure = go.Figure(data=data, layout=layout)



ply.iplot(figure)