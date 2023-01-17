import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

import plotly.figure_factory as ff
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
data = pd.read_csv('../input/bus-breakdown-and-delays.csv', low_memory=False)
data.sample(3).T
data['How_Long_Delayed'].sample(500).unique()
#### The string is searched for a sequence of numbers, the last is returned.
def minutes(x):
    flag = None
    num_list = list()
    num = str()
    for i in str(x):         
        if i.isnumeric():
            flag = 1
            num += i
        else:
            if num:
                num_list.append(num)
            flag = 0
            num = ''
    return int(num_list[-1]) if num_list else 0

data['How_Long_Delayed'] = data.How_Long_Delayed.fillna(0)
data['How_Long_Delayed'] = data.How_Long_Delayed.str.lower()
data['HLD'] = pd.to_numeric(data.How_Long_Delayed, errors='coerce')

data.loc[
    data.How_Long_Delayed.str.contains('m|-|in|/', na=False), 
    'HLD'
] = data.How_Long_Delayed.apply(minutes)
#### Sometimes, delay time indicated in hours. Basically - 1 hour.
data.loc[data.How_Long_Delayed.str.contains('hr|hour', na=False), 'HLD'] = 60
#### Explicit errors - equate to 0.
data.loc[data.HLD.isna(), 'HLD'] = 0
#### If the driver has not arrived at all, it may not be considered a delay.
data.loc[data.Breakdown_or_Running_Late =='Breakdown', 'HLD'] = 0
#### A delay of more than 250 minutes is considered an error.
data.loc[data.HLD > 250, 'HLD'] = 0
clean_delay = data.loc[data.HLD > 0]
clean_delay['School_Year'] = pd.to_datetime(clean_delay.School_Year.str[:4]).dt.year
delay_years = clean_delay.groupby(
    'School_Year', 
    as_index=True
)['HLD'].agg([('m', 'mean'), ('s', 'sum')])

trace1 = go.Scatter(
    x=delay_years.index.tolist(),
    y=delay_years.m.tolist(),
    showlegend=False
)
trace2 = go.Scatter(
    x=delay_years.index.tolist(),
    y=delay_years.s.tolist(),
    showlegend=False
)

fig = tools.make_subplots(
    rows=2, 
    cols=1,
    subplot_titles=('Mean', 'Sum'), 
    shared_xaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)

fig['layout']['yaxis1'].update(title='Mins')
fig['layout']['yaxis2'].update(title='Mins')

fig['layout'].update(height=600, width=700, title='"How long delayed" by years')
iplot(fig, filename='sss')
mean_del_per_cont = clean_delay.pivot_table(
    index='Boro', 
    columns='Reason', 
    values='HLD', 
    aggfunc='mean'
)

y = mean_del_per_cont.index.tolist()
traces = list()
for col in mean_del_per_cont.columns.tolist():
    traces.append(go.Bar(
        y = y,
        x = mean_del_per_cont[col].tolist(),
        name = col,
        showlegend=False,
        orientation='h'
    )
    )
layout = go.Layout(barmode='stack',
                   height=600, 
                   width=700, 
                   title='Mean delay for areas with reasons')

fig = go.Figure(data=traces, layout=layout)
iplot(fig, filename='stacked-bar')
    
sum_del_per_cont = clean_delay.pivot_table(
    index='Boro', 
    columns='Reason', 
    values='HLD', 
    aggfunc='sum'
)

y = sum_del_per_cont.index.tolist()
traces = list()
for col in sum_del_per_cont.columns.tolist():
    traces.append(go.Bar(
        y = y,
        x = sum_del_per_cont[col].tolist(),
        name = col,
        showlegend=False,
        orientation='h'
    )
    )
layout = go.Layout(barmode='stack',
                   height=600, 
                   width=700, 
                   title='Sum delay for areas with reasons')

fig = go.Figure(data=traces, layout=layout)
iplot(fig, filename='stacked-bar')
mean_delay_res = clean_delay.pivot_table(
    index=['Reason'], 
    values='HLD', 
    aggfunc='mean'
).sort_values(by='HLD')

mean_delay_bor = clean_delay.pivot_table(
    index=['Boro'], 
    values='HLD', 
    aggfunc='mean'
).sort_values(by='HLD')


trace1 = go.Bar(
    x=mean_delay_res.index.tolist(),
    y=mean_delay_res.HLD.tolist(),
    showlegend=False
)
trace2 = go.Bar(
    x=mean_delay_bor.index.tolist(),
    y=mean_delay_bor.HLD.tolist(),
    showlegend=False
)

fig = tools.make_subplots(
    rows=1, 
    cols=2,
    subplot_titles=('Reasons', 'Boro'), 
    shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout']['yaxis1'].update(title='Mins')

fig['layout'].update(height=600, width=700, title='Mean delayed time by:')
iplot(fig, filename='sss')
clean_delay.Created_On =  pd.to_datetime(clean_delay.Created_On)
clean_delay['Created_On_moun'] = clean_delay.Created_On.map(lambda x: x.strftime('%Y-%m'))
clean_delay['Created_On_day_of_week'] = clean_delay.Created_On.dt.dayofweek
clean_delay['Created_On_hour'] = clean_delay.Created_On.dt.hour
data_day = clean_delay.pivot_table(
    index='Created_On_moun',
    columns='Reason',
    values='HLD',
    aggfunc='sum'
)


x = data_day.index.tolist()
traces = list()
for col in data_day.columns.tolist():
    traces.append(go.Bar(
        x = x,
        y = data_day[col].tolist(),
        name = col,
        showlegend=False
    )
    )
layout = go.Layout(barmode='stack',
                   height=600, 
                   width=700, 
                   title='Total delay by month')

fig = go.Figure(data=traces, layout=layout)
iplot(fig, filename='stacked-bar')
delay_hours = clean_delay.groupby(
    'Created_On_hour', 
    as_index=True
)['HLD'].agg([('m', 'mean'), ('s', 'sum')])

trace1 = go.Scatter(
    x=delay_hours.index.tolist(),
    y=delay_hours.m.tolist(),
    showlegend=False
)
trace2 = go.Scatter(
    x=delay_hours.index.tolist(),
    y=delay_hours.s.tolist(),
    showlegend=False
)

fig = tools.make_subplots(
    rows=2, 
    cols=1,
    subplot_titles=('Mean', 'Sum'), 
    shared_xaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)

fig['layout']['yaxis1'].update(title='Mins')
fig['layout']['yaxis2'].update(title='Mins')

fig['layout'].update(height=600, width=700, title='Mean delay by hours')
iplot(fig, filename='sss')
