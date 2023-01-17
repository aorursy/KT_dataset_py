import pandas as pd
import numpy as np
df=pd.read_excel('https://query.data.world/s/44dvolkjonenaewlpqxyjiet26kygp')

import plotly.offline as ply
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.tools import make_subplots
#!jupyter nbextension enable --py widgetsnbextension
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets.widgets import *
from IPython.display import display
#from IPython.display import HTML
import colorlover as cl
ply.init_notebook_mode(connected=True)

colors=np.random.choice(cl.scales['9']['qual']['Pastel1'], 5, replace=False)
HTMLcolors=[]
for i in cl.to_numeric(colors):
    c='#'
    for j in i:
        c+=hex(int(j))[-2:]
    HTMLcolors.append(c)

buttons=[]

button_list=['ALL'] + df['Product Type'].unique().tolist()

for i in button_list:
    buttons.append(
        Button(
            description=i,
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to see data for '+i[0]+i[1:].lower()+' products',
            icon='',
            overflow='visible',
        )
    )

for i in buttons:
    i.style.button_color=HTMLcolors[button_list.index(i.description)]

button_box=Box(
    children=buttons,
    layout=Layout(
        display='flex',
        #align_content='stretch',
        justify_content='space-around',
    )
)

top_factories=df.groupby(['Country']).count().sort_values(by=['Factory Name'], ascending=False)['Factory Name'].reset_index()[:5]
factories=go.Bar(
    x=top_factories['Country'],
    y=top_factories['Factory Name'],
    text=top_factories['Factory Name'],
    textposition='auto',
    marker=dict(
        color=colors[0]
    ),
    hoverinfo='text'
)

layout1=go.Layout(
    title='<b>Top 5 countries with Nike factories</b>',
    xaxis=dict(
        title='<b>Country</b>'
    ),
    yaxis=dict(
        title='<b>No. of factories</b>'
    ),
    font=dict(
        family='Segoe UI',
    ),
    autosize=True,
)

graph1=go.FigureWidget(data=[factories], layout=layout1)

viz_title=HTML(
    value='<div align="center" style="font-family:Segoe UI;font-size:200%"><b>Vietnam, China and Taiwan have the highest no. of Nike factories for various product types</b></div><br>',
    layout=Layout(
        display='flex',
        align_content='center',
        width='100%'
    )
)

viz=VBox(
    children=[
        viz_title,
        button_box,
        graph1, 
    ],
    layout=Layout(
        display='flex',
        align_content='stretch',
    )    
)

def update_graph(factory_df, p):
    if p=='ALL':
        t='<b>Top 5 countries with Nike factories</b>'
    else:
        t='<b>Top 5 countries with Nike '+p[0]+p[1:].lower()+' factories</b>'
    with graph1.batch_update():
        graph1.layout.title=t
        graph1.data[0].x=factory_df['Country']
        graph1.data[0].y=factory_df['Factory Name']
        graph1.data[0].text=factory_df['Factory Name']
        graph1.data[0].marker.color=colors[button_list.index(p)]

def product_button_click(b):
    p=b.description
    factory_df=df[df['Product Type']==p].groupby(['Country']).count().sort_values(by=['Factory Name'], ascending=False)['Factory Name'].reset_index()[:5]
    update_graph(factory_df, p)
    
def all_button_click(b):
    factory_df=df.groupby(['Country']).count().sort_values(by=['Factory Name'], ascending=False)['Factory Name'].reset_index()[:5]
    update_graph(factory_df, 'ALL')

for b in buttons[1:]:
    b.on_click(product_button_click)
buttons[0].on_click(all_button_click)

display(viz)