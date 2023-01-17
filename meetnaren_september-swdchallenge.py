import pandas as pd
import numpy as np

regions=['Middle East', 'Latin America', 'Other', 'Asia Pacific', 'North America', 'Europe']
df=pd.DataFrame(
    {'Y2000':[3.0, 4.0, 10.0, 21.0, 27.0, 35.0],
    'Y2016':[3.4, 4.4, 9, 31, 25, 27]},
    index=regions
)

import plotly.offline as ply
import plotly.graph_objs as go
import colorlover as cl
from ipywidgets.widgets import *
ply.init_notebook_mode(connected=True)

colors=np.random.choice(cl.scales['5']['qual']['Set1'], 2, replace=False)
HTMLcolors=[]
for i in cl.to_numeric(colors):
    c='#'
    for j in i:
        c+=hex(int(j))[-2:]
    HTMLcolors.append(c)

marker_size=12

trace1=go.Scatter(
    x=regions,
    y=df.Y2000,
    mode='markers',
    marker=dict(
        size=marker_size,
        color=colors[0]
    ),
    name='2000',
    text=df.Y2000.astype(str)+'%',
    hoverinfo='text',
    error_y=dict(
        type='data',
        array=df.Y2016-df.Y2000,
        symmetric=False,
        width=0,
        color='#BBBBBB',
        thickness=4
    )
)

trace2=go.Scatter(
    x=regions,
    y=df.Y2016,
    mode='markers',
    marker=dict(
        size=marker_size,
        color=colors[1]
    ),
    name='2016',
    text=df.Y2016.astype(str)+'%',
    hoverinfo='text'
)

layout=go.Layout(
    #title=f'<b>Tourism in Asia Pacific and Europe has seen big changes between <font color="{HTMLcolors[0]}">2000</font> and <font color="{HTMLcolors[1]}">2016</font></b><br>',
    yaxis=dict(
        showgrid=False,
        showticklabels=False
    ),
    xaxis=dict(
        showgrid=False,
        #showticklabels=False
    ),
    showlegend=False,
    font=dict(
        family='Segoe UI',
    ),
)

figure1=go.Figure(data=[trace1, trace2], layout=layout)
figure=go.FigureWidget(data=[trace1, trace2], layout=layout)

viz_title=HTML(
    value=f'<div align="center" style="font-family:Segoe UI;font-size:200%"><b>Tourism in Asia Pacific and Europe has seen big changes between <font color="{HTMLcolors[0]}">2000</font> and <font color="{HTMLcolors[1]}">2016</font></b></div><br>',
    layout=Layout(
        display='flex',
        align_content='center',
        width='100%'
    )
)

viz=VBox(
    children=[
        viz_title,
        figure, 
    ],
    layout=Layout(
        display='flex',
        align_content='stretch',
    )    
)

display(viz)
#ply.iplot(figure1)
