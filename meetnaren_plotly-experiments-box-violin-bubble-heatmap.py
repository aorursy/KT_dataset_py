import pandas as pd
import numpy as np

appstore=pd.read_csv('../input/app-store-apple-data-set-10k-apps/AppleStore.csv')
appstore.head()
from pandas_summary import DataFrameSummary

dfs=DataFrameSummary(appstore)

dfs.summary().T
appstore['paid_free']=np.where(appstore.price>0, 'paid', 'free')
appstore['size_MB']=appstore.size_bytes/(1024*1024)
import plotly.offline as ply
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.tools import make_subplots

ply.init_notebook_mode(connected=True)

import colorlover as cl
from IPython.display import HTML

colors=cl.scales['12']['qual']

chosen_colors=[j for i in colors for j in colors[i]]

print('The color palette chosen for this notebook is:')
HTML(cl.to_html(chosen_colors))
paid_apps=appstore[appstore.paid_free=='paid']

genres=list(paid_apps.groupby(['prime_genre']).price.quantile(.75, interpolation='linear').sort_values().reset_index().prime_genre)

data=[]
for i in genres:
    data.append(
        go.Box(
            y=paid_apps[appstore.prime_genre==i].price,
            name=i,
            marker=dict(
                color=chosen_colors[genres.index(i)]
            )
        )
    )

layout=go.Layout(
    title='<b>Price comparison across app genres</b>',
    xaxis=dict(
        title='App genre',
        showgrid=False
    ),
    yaxis=dict(
        title='Price'
    ),
    showlegend=False,
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
paid_apps=appstore[appstore.paid_free=='paid']
genres=list(appstore.prime_genre.unique())
data=[]
for i in genres:
    data.append(
        go.Box(
            x=paid_apps[appstore.prime_genre==i].price,
            name=i,
            marker=dict(
                color=chosen_colors[genres.index(i)]
            )
        )
    )

layout=go.Layout(
    title='<b>Price comparison across app genres</b>',
    yaxis=dict(
        title='App genre',
        showgrid=False
    ),
    xaxis=dict(
        title='Price'
    ),
    showlegend=False,
    margin=dict(
        l=150
    ),
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
paid_apps_lt_50=appstore[(appstore.paid_free=='paid') & (appstore.price<=50)]

genres=list(paid_apps_lt_50.groupby(['prime_genre']).price.quantile(.75, interpolation='linear').sort_values().reset_index().prime_genre)

data=[]
for i in genres:
    data.append(
        go.Box(
            x=paid_apps_lt_50[appstore.prime_genre==i].price,
            name=' '*15+i,
            marker=dict(
                color=chosen_colors[genres.index(i)]
            ),
        )
    )

layout['height']=800

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
paid_apps_lt_50[(paid_apps_lt_50.prime_genre == 'Medical') | (paid_apps_lt_50.prime_genre == 'Weather')][['prime_genre','price']].sort_values(by=['prime_genre', 'price'])
genres_count=appstore.groupby(['prime_genre']).id.count().reset_index().sort_values(by=['id'], ascending=False)

top_genres=list(genres_count[genres_count.id>100].prime_genre)

top_apps=appstore[appstore.prime_genre.isin(top_genres)].copy()

top_apps['text']=top_apps.track_name+': '+top_apps.size_MB.astype(str)+' MB'
data=[]

for g in top_genres:
    data.append(
        go.Violin(
            y=top_apps[top_apps.prime_genre==g].size_MB,
            x=top_apps[top_apps.prime_genre==g].prime_genre,
            name=g,
            text=top_apps[top_apps.prime_genre==g].text,
            marker=dict(
                color=chosen_colors[list(top_genres).index(g)]
            ),
            box=dict(
                visible=True
            ),
            jitter=1,
            #points=False
        )
    )

layout=go.Layout(
    title='<b>App size comparison across genres</b>',
    xaxis=dict(
        title='App genre',
        showgrid=False
    ),
    yaxis=dict(
        title='Size (MB)'
    ),
    showlegend=False,
    hovermode='closest',
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
top_apps_lt_500MB = top_apps[top_apps.size_MB<500].copy()
data=[]

for g in top_genres:
    data.append(
        go.Violin(
            y=top_apps_lt_500MB[top_apps_lt_500MB.prime_genre==g].size_MB,
            x=top_apps_lt_500MB[top_apps_lt_500MB.prime_genre==g].prime_genre,
            name=g,
            text=top_apps_lt_500MB[top_apps_lt_500MB.prime_genre==g].text,
            marker=dict(
                color=chosen_colors[list(top_genres).index(g)]
            ),
            box=dict(
                visible=True
            ),
            jitter=.75,
            #points=False
        )
    )

layout=go.Layout(
    title='<b>App size comparison across genres</b>',
    xaxis=dict(
        title='<b>App genre</b>',
        showgrid=False
    ),
    yaxis=dict(
        title='<b>Size (MB)</b>'
    ),
    showlegend=False,
    hovermode='closest',
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
insurance=pd.read_csv('../input/insurance/insurance.csv')
insurance.head()
def bubble(trace_col):
    data=[]
    trace_vals=list(insurance[trace_col].unique())
    for i in range(len(trace_vals)):
        data.append(
            go.Scatter(
                x=insurance[insurance[trace_col]==trace_vals[i]].age,
                y=insurance[insurance[trace_col]==trace_vals[i]].bmi,
                mode='markers',
                marker=dict(
                    color=chosen_colors[i*4+1],
                    opacity=0.5,
                    size=insurance[insurance[trace_col]==trace_vals[i]].charges/3000,
                    line=dict(
                        width=0.0
                    )
                ),
                text='Age:'+insurance[insurance[trace_col]==trace_vals[i]].age.astype(str)+'<br>'+'BMI:'+insurance[insurance[trace_col]==trace_vals[i]].bmi.astype(str)+'<br>'+'Charges:'+insurance[insurance[trace_col]==trace_vals[i]].charges.astype(str)+'<br>'+trace_col+':'+trace_vals[i],
                hoverinfo='text',
                name=trace_vals[i]
            )
        )

    layout=go.Layout(
        title='<b>Insurance cost comparison for different ages / BMIs</b>', 
        hovermode='closest',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(
            family='Segoe UI',
            color='#ffffff'
        ),
        xaxis=dict(
            title='<b>Age<b>'
        ),
        yaxis=dict(
            title='<b>BMI<b>'
        ),
        legend=dict(
            orientation='h',
            x=0,
            y=1.1
        ),
        shapes=[
            dict(
                type='rect',
                xref='x',
                x0=min(insurance.age)-1,
                x1=max(insurance.age)+1,
                yref='y',
                y0=18.5,
                y1=24.9,
                line=dict(
                    width=0.0
                ),
                fillcolor='rgba(255,255,255,0.2)'
            )
        ],
        annotations=[
            dict(
                xref='x',
                x=45,
                yref='y',
                y=18.5,
                text='Healthy BMI zone',
                ay=35
            )
        ]
    )
    
    figure = go.Figure(data=data, layout=layout)
    
    ply.iplot(figure)
bubble('sex')
bubble('region')
bubble('smoker')
ages=insurance.age.unique().tolist()
regions=insurance.region.unique().tolist()
charge_matrix=pd.DataFrame(data=0, index=ages, columns=regions).sort_index()
charge_count=pd.DataFrame(data=0, index=ages, columns=regions).sort_index()
def create_charge_matrix(row):
    a=row['age']
    r=row['region']
    c=row['charges']
    charge_matrix.loc[a, r]+=c
    charge_count.loc[a, r]+=1    
insurance.apply(lambda row: create_charge_matrix(row), axis=1)

#Calculating average charges
charge_matrix /= charge_count
z=[]
for i in range(len(charge_matrix)):
    z.append(charge_matrix.iloc[i].tolist())
trace1=go.Heatmap(
    x=charge_matrix.columns.tolist(),
    y=charge_matrix.index.tolist(),
    z=z,
    colorscale='Electric'
)

data=[trace1]

layout=go.Layout(
    title='<b>Average insurance charges by age and region<b>',
    height=800,
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(
        family='Segoe UI',
        color='#ffffff'
    ),
    xaxis=dict(
        title='<b>Region<b>'
    ),
    yaxis=dict(
        title='<b>Age<b>'
    )
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)