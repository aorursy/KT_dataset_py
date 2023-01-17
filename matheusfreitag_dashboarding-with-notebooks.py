import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# import plotly
import plotly
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

print(os.listdir("../input"))
restaurants = pd.read_csv('../input/restaurant-scores-lives-standard.csv')
howManyOfEach = restaurants.fillna(-1).groupby('risk_category').business_id.count()

labels = restaurants['risk_category'].unique()
values = howManyOfEach.values

trace = go.Pie(labels=labels, values=values)
plotly.offline.iplot([trace], filename='basic_pie_chart')
overview = restaurants[np.isfinite(restaurants['inspection_score'])]
data = [go.Bar(
            x=np.sort(overview['inspection_score'].unique()),
            y=overview.groupby('inspection_score').business_id.count().values
    )]
layout = go.Layout(
    title='How Many Restaurants Scored X',
    xaxis=dict(
        title='Score',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='How Many Restaurants',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig, filename='basic-bar')