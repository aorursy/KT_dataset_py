import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
init_notebook_mode(connected=True)
import os
print(os.listdir("../input"))
data = pd.read_csv("../input/IHMStefanini_industrial_safety_and_health_database.csv", delimiter=',', header=0, parse_dates = ["Data"], index_col ="Data")
data.shape
data.head()
data.index
datadict = pd.DataFrame(data.dtypes)
datadict
datadict['MissingVal'] = data.isnull().sum()
datadict
datadict['NUnique']=data.nunique()
datadict
data.describe(include=['object'])
data['Day of the Week'] = data.index.dayofweek
grouped_data = pd.DataFrame(data.groupby(['Countries','Day of the Week']).count())
grouped_data
grouped_data = pd.DataFrame(data.groupby(['Industry Sector','Day of the Week']).count())
grouped_data
# Faz o resampling dos dados para 24h
df = data
df = df.Countries.resample('24H').count()

#Plot o gráfico
trace_high = go.Scatter(
                x=df.index,
                y=df,
                name = "AAPL High",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

dados= [trace_high]

layout = dict(
    title = "Number of Accidents/Day (all countries)",

)

fig = dict(data=dados, layout=layout)
iplot(fig, filename = "Manually Set Range")
# Faz o resampling dos dados para 24h
df2 = data
df2 = df2.Countries.resample('24H').count()
temp = df2.rolling(window=30)
b = temp.mean()

#Plot o gráfico
trace_high = go.Scatter(
                x=b.index,
                y=b,
                name = "AAPL High",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

dados= [trace_high]

layout = dict(
    title = "Moving Average of 30 Days of the number of accidents/Day (all countries)",

)

fig = dict(data=dados, layout=layout)
iplot(fig, filename = "Manually Set Range")
from IPython.display import Image
Image("../input/Accidents_Storyline_example.png")
g = sns.factorplot(data=data, kind="count", x="Countries", hue = "Local", size=8, aspect=1)
columns = ['total','cumulative_sum', 'cumulative_perc','demarcation']
paretodf = pd.DataFrame(columns=columns)
paretodf = paretodf.fillna(0)

paretodf['total'] = data["Risco Critico"].value_counts()
#print(paretodf)

paretodf['cumulative_sum'] = paretodf.cumsum()
#print(paretodf)

paretodf['cumulative_perc'] = 100*paretodf.cumulative_sum/paretodf.total.sum()
#print(paretodf)

paretodf['demarcation'] = 80
#print(paretodf)
trace1 = Bar(
    x=paretodf.index[0:7],
    y=paretodf.total[0:7],
    name='Count',
    marker=dict(
        color='rgb(34,163,192)'
               )
)
trace2 = Scatter(
    x=paretodf.index[0:7],
    y=paretodf.cumulative_perc[0:7],
    name='Cumulative Percentage',
    yaxis='y2',
    line=dict(
        color='rgb(243,158,115)',
        width=2.4
       )
)
trace3 = Scatter(
    x=paretodf.index[0:7],
    y=paretodf.demarcation[0:7],
    name='80%',
    yaxis='y2',
    line=dict(
        color='rgba(128,128,128,.45)',
        dash = 'dash',
        width=1.5
       )
)
dataplot = [trace1, trace2,trace3]
layout = Layout(
    title='Critical Risks Pareto',
    titlefont=dict(
        color='',
        family='',
        size=0
    ),
    font=Font(
        color='rgb(128,128,128)',
        family='Balto, sans-serif',
        size=12
    ),
    width=623,
    height=623,
    paper_bgcolor='rgb(240, 240, 240)',
    plot_bgcolor='rgb(240, 240, 240)',
    hovermode='compare',
    margin=dict(b=250,l=60,r=60,t=65),
    showlegend=True,
       legend=dict(
          x=.83,
          y=1.3,
          font=dict(
            family='Balto, sans-serif',
            size=12,
            color='rgba(128,128,128,.75)'
        ),
    ),
    annotations=[ dict(
                  text="Cumulative Percentage",
                  showarrow=False,
                  xref="paper", yref="paper",
                  textangle=90,
                  x=1.100, y=.75,
                  font=dict(
                  family='Balto, sans-serif',
                  size=14,
                  color='rgba(243,158,115,.9)'
            ),)],
    xaxis=dict(
      tickangle=-90
    ),
    yaxis=dict(
        title='Count',
        range=[0,300],
      tickfont=dict(
            color='rgba(34,163,192,.75)'
        ),
      tickvals = [0,6000,12000,18000,24000,30000],
        titlefont=dict(
                family='Balto, sans-serif',
                size=14,
                color='rgba(34,163,192,.75)')
    ),
    yaxis2=dict(
        range=[0,101],
        tickfont=dict(
            color='rgba(243,158,115,.9)'
        ),
        tickvals = [0,20,40,60,80,100],
        overlaying='y',
        side='right'
    )
)

fig = dict(data=dataplot, layout=layout)
iplot(fig)
