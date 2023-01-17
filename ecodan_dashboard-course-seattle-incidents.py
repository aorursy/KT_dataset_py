# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sea

# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/crisis-data.csv", parse_dates=['Reported Date', 'Occurred Date / Time'])
df.head()
# how big is the dataset?
print("Num rows: {0}".format(df.shape[0]))
# what range of dates do we have?
print("Min date: {0} | Max date: {1}".format(df['Reported Date'].min(), df['Reported Date'].max()))
# clean up bogus dates
df = df[df['Reported Date'].dt.year > 2000].copy()
print("Min date: {0} | Max date: {1}".format(df['Reported Date'].min(), df['Reported Date'].max()))
dfg = df[['Reported Date','Template ID']].groupby('Reported Date').count()
dfg.rename({'Template ID':'Incidents'},axis=1,inplace=True)
dfg.plot.line(
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day",
    legend=False
)
data = [go.Scatter(x=dfg.index.tolist(), y=dfg['Incidents'])]

# specify the layout of our figure
layout = dict(title = "Number of Incidents Per Day",
              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
dfp_precinct = pd.pivot_table(
    data=df[['Reported Date', 'Precinct']],
    index='Reported Date',
    columns=['Precinct'],
    aggfunc=len,
)
dfp_precinct.iloc[-30::,:].plot.bar(
    stacked=True, 
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day by precint (last 30 days)"
).legend(loc='center left', bbox_to_anchor=(1, 0.5))
dfp_precinct.iloc[-30::,:].div(dfp_precinct.iloc[-30::,:].sum(axis=1), axis=0).plot.bar(
    stacked=True, 
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day by precint (last 30 days)"
).legend(loc='center left', bbox_to_anchor=(1, 0.5))
data = [
    go.Bar(
        x=dfp_precinct.index.tolist(), 
        y=dfp_precinct[col],
        name=col
    ) for col in dfp_precinct.columns
]

# updatemenus = list([
#     dict(active=-1,
#          buttons=list([   
#             dict(label = 'High',
#                  method = 'update',
#                  args = [{'visible': [True, True, False, False]},
#                          {'title': 'Yahoo High',
#                           'annotations': high_annotations}]),
#             dict(label = 'Low',
#                  method = 'update',
#                  args = [{'visible': [False, False, True, True]},
#                          {'title': 'Yahoo Low',
#                           'annotations': low_annotations}]),
#             dict(label = 'Both',
#                  method = 'update',
#                  args = [{'visible': [True, True, True, True]},
#                          {'title': 'Yahoo',
#                           'annotations': high_annotations+low_annotations}]),
#             dict(label = 'Reset',
#                  method = 'update',
#                  args = [{'visible': [True, False, True, False]},
#                          {'title': 'Yahoo',
#                           'annotations': []}])
#         ]),
#     )
# ])

# specify the layout of our figure
layout = dict(
    title = "Number of Incidents Per Day",
    xaxis= dict(
        title= 'Date',
        ticklen= 5,
        zeroline= False,
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date',
    ),
    barmode='stack',
)

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)



dfp_sector = pd.pivot_table(
    data=df[['Reported Date', 'Sector']],
    index='Reported Date',
    columns=['Sector'],
    aggfunc=len,
)

dfp_sector.iloc[-30::,:].plot.bar(
    stacked=True, 
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day by sector (last 30 days)"
).legend(loc='center left', bbox_to_anchor=(1, 0.5))
dfp_sector.iloc[-30::,:].div(dfp_sector.iloc[-30::,:].sum(axis=1), axis=0).plot.bar(
    stacked=True, 
    figsize=(12,5), 
    colormap='tab20',
    title="Incidents per day by sector (last 30 days)"
).legend(loc='center left', bbox_to_anchor=(1, 0.5))
df_trend = pd.DataFrame(index=df['Sector'].sort_values().unique()).iloc[0:-1,:]
df_trend = df_trend.join(
    pd.Series(data=dfp_sector.iloc[-30::,:].T.mean(axis=1), name='last30')
)
df_trend = df_trend.join(
    pd.Series(data=dfp_sector.iloc[-3::,:].T.mean(axis=1), name='last3')
)
df_trend['trend'] = df_trend['last3'] - df_trend['last30'] 
df_trend['trend'].sort_values().plot.bar(title="Average daily incidents per sector over last three days vs last month", figsize=(12,5), colormap='tab20')
data = [
    go.Bar(
        x=df_trend.index.tolist(), 
        y=df_trend['trend'],
        marker={
            'color':['red' if x>0 else 'green' for x in  df_trend['trend']]
        }
    )
]

# specify the layout of our figure
layout = dict(title = "Average daily incidents over last three days vs last month",
              xaxis= dict(title= 'Sector',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)
