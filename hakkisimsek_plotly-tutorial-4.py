import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
color = sns.color_palette()

import os
import datetime

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

print(os.listdir("../input"))
df = pd.read_csv('../input/train-set/extracted_train/extracted_train', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str},
                    nrows=None)
df["totals.transactionRevenue"] = df["totals.transactionRevenue"].astype('float')
df['date'] = df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), 
                                    int(str(x)[4:6]), int(str(x)[6:])))
df['time'] = pd.to_datetime(df['visitStartTime'], unit='s')
df['month'] = df['time'].dt.month
df['dow'] = df['time'].dt.dayofweek
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

df['day_frame'] = 0
df['day_frame'] = np.where((df["hour"]>=0) & (df["hour"]<4), 'overnight', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=4) & (df["hour"]<8), 'dawn', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=8) & (df["hour"]<12), 'morning', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=12) & (df["hour"]<14), 'lunch', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=14) & (df["hour"]<18), 'afternoon', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=18) & (df["hour"]<21), 'evening', 
                           df['day_frame'])
df['day_frame'] = np.where((df["hour"]>=21) & (df["hour"]<24), 'night', 
                           df['day_frame'])
miss = pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/len(df)], 
              axis=1).rename(columns={0:'Missing Records', 
                        1:'Percentage (%)'}).sort_values(by='Percentage (%)',
                                                         ascending=False)[:8]
trace = go.Bar(
        y=miss.index[::-1],
        x=miss['Percentage (%)'][::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color='firebrick',
        )
    )

data = [trace]
layout = dict(
    title = 'Percentage (%) missing values',
    margin  = dict(l = 200
                                      )
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
def scatter_plot(data, color, name, mode=None):
    trace = go.Scatter(
        x=data.index[::-1],
        y=data.values[::-1],
        showlegend=False,
        name = name,
        mode = mode,
        marker=dict(
        color = color
        )
    )
    return trace
f, ax = plt.subplots(1,2, figsize=(18,5))
rev = df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

ax[0].scatter(range(rev.shape[0]), 
              np.sort(rev["totals.transactionRevenue"].values), color='navy')
ax[0].set_xlabel('index')
ax[0].set_ylabel('Revenue')
ax[0].set_title('Transaction revenue by visitors')

ax[1].scatter(range(rev.shape[0]), 
              np.sort(np.log1p(rev["totals.transactionRevenue"].values)), 
              color='navy')
ax[1].set_xlabel('index')
ax[1].set_ylabel('Revenue (log)')
ax[1].set_title('Transaction revenue (log) by visitors')
plt.show()

visit_group = df.groupby('fullVisitorId')['fullVisitorId'].count()

for i in [2, 10, 30, 50]:
    print('Visitors that appear less than {} times: {}%'.format(i, 
                                 round((visit_group < i).mean() * 100, 2)))
rev = df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
rev1 = np.sort(rev["totals.transactionRevenue"].values)
rev2 = rev1[rev1>0]
rev3 = np.sort(np.log1p(rev["totals.transactionRevenue"].values))
rev4 = rev3[rev3>0]

trace0 = scatter_plot(pd.DataFrame(rev2)[0],'red','revenue', mode='markers')
trace1 = scatter_plot(pd.DataFrame(rev4)[0],'red','revenue', mode='markers')

fig = tools.make_subplots(rows=1, cols=2, 
                          subplot_titles=('Total revenue by user level',
                                          'Total revenue (log) by user level'
                                         )
                         )

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(showlegend=False)
py.iplot(fig)
stats = df.groupby('date')['totals.transactionRevenue'].agg(['size', 'count'])
stats.columns = ["count total", "count non-zero"]
stats.index = pd.to_datetime(stats.index)
stats['per'] = stats['count non-zero'] / stats['count total']

df1dm = stats.reset_index().resample('D', on='date').mean()
df1wm = stats.reset_index().resample('W', on='date').mean()
df1mm = stats.reset_index().resample('M', on='date').mean()

trace0 = scatter_plot(df1mm['count total'].round(0), 'red', 'count')
trace1 = scatter_plot(df1wm['count total'].round(0), 'orange', 'count')
trace2 = scatter_plot(df1dm['count total'], 'indigo', 'count')

trace3 = scatter_plot(df1mm['count non-zero'].round(0), 'red', 'count')
trace4 = scatter_plot(df1wm['count non-zero'].round(0), 'orange', 'count')
trace5 = scatter_plot(df1dm['count non-zero'], 'indigo', 'count')

fig = tools.make_subplots(rows=2, cols=3, 
                          subplot_titles=('Monthly total','Weekly', 'Daily',
                                          'Monthly non-zero','Weekly', 'Daily'
                                         )
                         )

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 3)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 2, 3)

fig['layout'].update(showlegend=False, 
                     title='Total visitors vs non-zero revenue visitors')
py.iplot(fig)
trace0 = scatter_plot(df1mm['per'].round(4), 'red', 'count', 'markers')
trace1 = scatter_plot(df1wm['per'].round(4), 'orange', 'count', 'markers')
trace2 = scatter_plot(df1dm['per'].round(4), 'indigo', 'count', 'markers')

fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], 
                          [{'colspan': 2}, None]],
                          subplot_titles=('Monthly','Weekly', 'Daily'
                                         )
                         )
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=False, title='Non-zero/Total visitors')
py.iplot(fig)
trace = go.Scatter(
                   x=list(df1dm.index),
                   y=list(df1dm.per.round(4)), 
                   line=dict(color='red'))

data = [trace]
layout = dict(
    title='Non-zero/Total visitors',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=7,
                     label='7d',
                     step='day',
                     stepmode='backward'),
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=3,
                     label='3m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig)
trace = [
    go.Histogram(x=df['hour'],
                opacity = 0.7,
                 name="Total Visits",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='grey')
                ),
    
    go.Histogram(x=df[df['totals.transactionRevenue'].notnull()]['hour'],
                 visible=False,
                 opacity = 0.7,
                 name = "Non-zero revenue visits",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='red')
                ),
    
    go.Histogram(x=df[df['totals.transactionRevenue'].isnull()]['hour'],
                 visible=False,
                opacity = 0.7,
                 name = "Zero revenue visits",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='aqua')         
                )
]

layout = go.Layout(title='Visiting hours',
    paper_bgcolor = 'rgb(240, 240, 240)',
     plot_bgcolor = 'rgb(240, 240, 240)',
    autosize=True, xaxis=dict(tickmode="linear"),
                   yaxis=dict(title="# of Visits",
                             titlefont=dict(size=17)),
                  )

updatemenus = list([
    dict(
    buttons=list([
        dict(
            args = [{'visible': [True, False, False]}],
            label="Total visits",
            method='update',
        ),
        dict(
            args = [{'visible': [False, True, False]}],
            label="Non-zero revenue visits",
            method='update',
        ),
        dict(
            args = [{'visible': [False, False, True]}],
            label="Zero revenue visits",
            method='update',
        ),
        
    ]),
        direction="down",
        pad = {'r':10, "t":10},
        x=0.1,
        y=1.25,
        yanchor='top',
    )
])
layout['updatemenus'] = updatemenus

fig = dict(data=trace, layout=layout)
py.iplot(fig)
fv = df.pivot_table(index="device.operatingSystem",columns="day_frame",
                    values="totals.transactionRevenue",aggfunc=lambda x:x.sum())
fv = fv[['morning', 'lunch', 'afternoon', 'evening','night','overnight', 'dawn']]
fv = fv.sort_values(by='morning', ascending=False)[:6]

trace = go.Heatmap(z=[fv.values[0],fv.values[1],fv.values[2],fv.values[3],
                      fv.values[4],fv.values[5]],
                   x=['morning', 'lunch', 'afternoon', 'evening', 'night',
                      'overnight','dawn'],
                   y=fv.index.values, colorscale='Blues', reversescale = True
                  )

data=[trace]
layout = go.Layout(
    title='Total Revenue by Device OS<br>(parts of the day)')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
color = ['tomato',  'bisque','lightgreen', 'gold', 'tan', 'lightgrey', 'cyan']

def PieChart(column, title, limit):
    revenue = "totals.transactionRevenue"
    count_trace = df.groupby(column)[revenue].size().nlargest(limit).reset_index()
    non_zero_trace = df.groupby(column)[revenue].count().nlargest(limit).reset_index()
    rev_trace = df.groupby(column)[revenue].sum().nlargest(limit).reset_index()    

    trace1 = go.Pie(labels=count_trace[column], 
                    values=count_trace[revenue], 
                    name= "Visit", 
                    hole= .5, textfont=dict(size=10),
                    domain= {'x': [0, .32]},
                   marker=dict(colors=color))

    trace2 = go.Pie(labels=non_zero_trace[column], 
                    values=non_zero_trace[revenue], 
                    name="Revenue", 
                    hole= .5,  textfont=dict(size=10),
                    domain= {'x': [.34, .66]})
    
    trace3 = go.Pie(labels=rev_trace[column], 
                    values=rev_trace[revenue], 
                    name="Revenue", 
                    hole= .5,  textfont=dict(size=10),
                    domain= {'x': [.68, 1]})

    layout = dict(title= title, font=dict(size=15), legend=dict(orientation="h"),
                  annotations = [
                      dict(
                          x=.10, y=.5,
                          text='<b>Number of<br>Visitors', 
                          showarrow=False,
                          font=dict(size=12)
                      ),
                      dict(
                          x=.50, y=.5,
                          text='<b>Number of<br>Visitors<br>(non-zero)', 
                          showarrow=False,
                          font=dict(size=12)
                      ),
                      dict(
                          x=.88, y=.5,
                          text='<b>Total<br>Revenue', 
                          showarrow=False,
                          font=dict(size=12)
                      )
        ])
    
    fig = dict(data=[trace1, trace2,trace3], layout=layout)
    py.iplot(fig)
PieChart("device.operatingSystem", "Operating System", 4)
fv = df.pivot_table(index="trafficSource.source",columns="dow",values="totals.transactionRevenue",aggfunc=lambda x:x.sum())
fv = fv.sort_values(by=0, ascending=False)[:6]

trace = go.Heatmap(z=[fv.values[0],fv.values[1],fv.values[2], fv.values[3],fv.values[4],fv.values[5]],
                   x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'],
                   y=fv.index.values, colorscale='Reds'
                  )

data=[trace]
layout = go.Layout(dict(
    title='Total Revenue by Traffic Source<br>(day of week)'),
                margin  = dict(l = 150
                                      )  )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
PieChart("trafficSource.source", "Traffic Source", 5)
fv = df.pivot_table(index="trafficSource.medium",columns="month",values="totals.transactionRevenue",aggfunc=lambda x:x.sum())
fv = fv.sort_values(by=1, ascending=False)[:5]

trace = go.Heatmap(z=[fv.values[0],fv.values[1],fv.values[2], fv.values[3],fv.values[4]],
                   x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   y=fv.index.values, colorscale='Reds'
                  )

data=[trace]
layout = go.Layout(dict(
    title='Total Revenue by Source Medium<br>(months)'),
                margin  = dict(l = 150
                                      ))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
PieChart("trafficSource.medium", "Source Medium", 5)
count_geo = df.groupby('geoNetwork.country')['geoNetwork.country'].count()

data = [dict(
        type = 'choropleth',
        locations = count_geo.index,
        locationmode = 'country names',
        z = count_geo.values,
        text = count_geo.index,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of calls'),
      ) ]

layout = dict(
    title = 'Number of Visits by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict(data=data, layout=layout )
py.iplot(fig, validate=False)
count = df[['geoNetwork.continent','totals.transactionRevenue',
            'day']].groupby('geoNetwork.continent', as_index=False)['day'].\
                    count().sort_values(by='day', ascending=False)

sumrev = df[['geoNetwork.continent','totals.transactionRevenue']].groupby('geoNetwork.continent', 
                                                       as_index=False)['totals.transactionRevenue'].\
                    sum().sort_values(by='totals.transactionRevenue', ascending=False)

meanrev = df[['geoNetwork.continent','totals.transactionRevenue']].groupby('geoNetwork.continent', 
                                                       as_index=False)['totals.transactionRevenue'].\
                    mean().sort_values(by='totals.transactionRevenue', ascending=False)

trace = [
    go.Bar(x=count['geoNetwork.continent'],
    y=count['day'],
                opacity = 0.7,
                 name="COUNT",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='aqua')
                ),
    go.Bar(x=sumrev['geoNetwork.continent'],
    y=sumrev['totals.transactionRevenue'],
                 visible=False,
                 opacity = 0.7,
                 name = "TOTAL",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='navy')
                ),
    go.Bar(x=meanrev['geoNetwork.continent'],
    y=meanrev['totals.transactionRevenue'],
                 visible=False,
                opacity = 0.7,
                 name = "MEAN",
                 hoverinfo="y",
                 marker=dict(line=dict(width=1.6),
                            color='red')
                
                )
]

layout = go.Layout(title = 'Revenue Statistics of Continents',
    paper_bgcolor = 'rgb(240, 240, 240)',
     plot_bgcolor = 'rgb(240, 240, 240)',
    autosize=True,
                   xaxis=dict(title="",
                             titlefont=dict(size=20),
                             tickmode="linear")
                  )

updatemenus = list([
    dict(
    buttons=list([
        dict(
            args = [{'visible': [True, False, False]}],
            label="Count",
            method='update',
        ),
        dict(
            args = [{'visible': [False, True, False]}],
            label="Total Revenue",
            method='update',
        ),
        dict(
            args = [{'visible': [False, False, True]}],
            label="Mean Revenue",
            method='update',
        ),
        
    ]),
        direction="down",
        pad = {'r':10, "t":10},
        x=0.1,
        y=1.25,
        yanchor='top',
    )
])

layout['updatemenus'] = updatemenus

fig = dict(data=trace, layout=layout)
py.iplot(fig)