import pandas as pd
import numpy as np
df = pd.read_excel('https://query.data.world/s/kbtrxotkhv7km3zp4oz5wkt72i42qm')

df['date1']=pd.to_datetime(df.date)

grouped=df.groupby(['date1', 'source', 'type']).amount.sum().reset_index().sort_values(by=['date1'])

grouped['month']=grouped.date1.dt.month
grouped['year']=grouped.date1.dt.year
grouped['yyyy-mm']=grouped.year.astype(str)+'-'+grouped.month.astype(str)
grouped['y']=1
grouped['source_cat']='Donald J. Trump for President, Inc.'

for i in grouped.index:
    if grouped.loc[i, 'type']=='FEC' and grouped.loc[i, 'source']!='Donald J. Trump for President, Inc.':
        grouped.loc[i, 'source_cat']='Other campaigns'
    elif grouped.loc[i, 'type']=='government':
        grouped.loc[i, 'source_cat']='Taxpayer dollars'

grouped=grouped.sort_values(by=['year', 'month', 'source_cat']).reset_index()

for i in range(1, len(grouped)):
    prev_ym=grouped.iloc[i-1]['yyyy-mm']
    prev_y=grouped.iloc[i-1]['y']
    if prev_ym==grouped.iloc[i]['yyyy-mm']:
        grouped.loc[i, 'y']=prev_y+1

grouped=grouped[grouped.amount>0].reset_index()

import plotly.offline as ply
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.tools import make_subplots

ply.init_notebook_mode(connected=True)

data=[]
source_cats=[
    'Donald J. Trump for President, Inc.',
    'Other campaigns',
    'Taxpayer dollars'
]

colors=[
    '#FFA73A',
    '#5CBFBF',
    '#4D5355'
]
for i in source_cats:
    data.append(
        go.Scatter(
            x=grouped[grouped.source_cat==i]['yyyy-mm'],
            y=grouped[grouped.source_cat==i].y*-1,
            mode='markers',
            marker=dict(
                symbol='square',
                size=np.floor(np.log10(grouped[grouped.source_cat==i].amount))*2,
                color=colors[source_cats.index(i)]
            ),
            name=i,
            hoverinfo='text',
            text='On '+grouped[grouped.source_cat==i].date1.astype(str)+',<br>'+grouped[grouped.source_cat==i].source.astype(str)+' spent<br><b>$'+grouped[grouped.source_cat==i].amount.astype(str)+'</b>'
        )
    
    )
layout=go.Layout(
    title='<b>Paying the President</b>',
    hovermode='closest',
    legend=dict(
        orientation='h',
        x=0,
        y=0
    ),
    xaxis=dict(
        side='top',
        showgrid=False,
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False
    ),
    font=dict(
        family='Segoe UI'
    )
)

figure=go.Figure(data=data, layout=layout)

ply.iplot(figure)
