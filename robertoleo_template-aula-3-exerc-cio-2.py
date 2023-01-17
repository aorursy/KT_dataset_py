import pandas as pd
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import numpy as np
url = 'https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/cars.csv'
df = pd.read_csv(url)
df.head()
df.info()
line = df.groupby(['Year']).size().reset_index(name='Total')
line
trace1 = go.Scatter(
                    x = line.Year,
                    y = line.Total,
                    marker = dict(color = 'rgba(15, 120, 4, 0.8)'),
                    text= line.Year.values)

data = [trace1]
layout = dict(title = 'Quantidade de veículos por ano',
              xaxis= dict(title= 'Ano',zeroline= False,
                         ticks='outside',
                            tick0=0,
                            dtick=1,
                            ticklen=5,
                            tickwidth=1,
                            tickcolor='#000'),
              yaxis = dict(title= 'Total de veículos',
                            ticks='outside',
                            tick0=0,
                            dtick=100,
                            ticklen=5,
                            tickwidth=1,
                            tickcolor='#000'
                          )
             )
fig = dict(data = data, layout = layout)
iplot(fig)
bar = df.groupby('Driveline').size().reset_index(name='Total')
bar
trace1 = go.Bar(
            width  = 0.7,
            x = bar[bar['Driveline']=='All-wheel drive'].Driveline,
            y = bar[bar['Driveline']=='All-wheel drive'].Total,
            name='All-wheel drive',
            text= 'All-wheel drive')
trace2 = go.Bar(
            width  = 0.7,
            x = bar[bar['Driveline']=='Four-wheel drive'].Driveline,
            y = bar[bar['Driveline']=='Four-wheel drive'].Total,
            name='Four-wheel drive',
            text= 'Four-wheel drive')
trace3 = go.Bar(
            width  = 0.7,
            x = bar[bar['Driveline']=='Front-wheel drive'].Driveline,
            y = bar[bar['Driveline']=='Front-wheel drive'].Total,
            name='Front-wheel drive',
            text= 'Front-wheel drive')
trace4 = go.Bar(
            width  = 0.7,
            x = bar[bar['Driveline']=='Rear-wheel drive'].Driveline,
            y = bar[bar['Driveline']=='Rear-wheel drive'].Total,
            name='Rear-wheel drive',
            text= 'Rear-wheel drive')

data = [trace1,trace2,trace3,trace4]
layout = go.Layout(
    title = 'Quantidade de veículos por tipo de tração',
              xaxis= dict(title= 'Tipo de Tração'),
              yaxis = dict(title= 'Quantidade de veículos', ticks='outside'),
    barmode = "group")
fig = dict(data = data, layout = layout)
iplot(fig)
df.Horsepower.sort_values().unique()
hist = df.Horsepower.sort_values()
hist.head(10)
trace1 = go.Histogram(x=hist,nbinsx = 50)

data = [trace1]
layout = dict(title = 'Total de veículos por cavalos',
              xaxis= dict(title= 'Total de cavalos',ticklen= 5,zeroline= False, tickangle=20),
              yaxis = dict(title= 'Total de veículos',
                            ticks='outside',
                            tick0=0,
                            dtick=50,
                            ticklen=5,
                            tickwidth=1,
                            tickcolor='#000'
                          )
             )
fig = dict(data = data, layout = layout)
iplot(fig)
rdr = df.groupby(['Make']).size().reset_index(name='Total').nlargest(columns='Total',n=10)
rdr
trace1 = go.Scatterpolar(
  r = rdr.Total,
  theta = rdr.Make,
  fill = 'toself'
)
data = [trace1]

layout = go.Layout(
  title = "10 Maiores fabricantes de veículos por quantidade produzida",
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 650]
    )
  ),
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
df.Horsepower.value_counts()
trace1 = go.Box(
    y= df.Horsepower,
    name='Cavalos'
)
data = [trace1]
layout = dict(title = 'Quantidade de cavalos',
              xaxis= dict(ticklen= 5,zeroline= False),
              yaxis = dict(title= 'Total de cavalos',
                            ticks='outside',
                            tick0=0,
                            dtick=50,
                            ticklen=5,
                            tickwidth=1
                          ),
              boxmode='group'
             )

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df['Horsepower'][df['Hybrid'] == pd.unique(df['Hybrid'])[0]]
data = []
for i in range(0,len(pd.unique(df.Hybrid))):
    trace = {
            "type": 'violin',
            "x": df['Hybrid'][df['Hybrid'] == pd.unique(df['Hybrid'])[i]].astype(str),
            "y": df['Horsepower'][df['Hybrid'] == pd.unique(df['Hybrid'])[i]],
            "name": pd.unique(df['Hybrid'])[i].astype(str),
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        }
    data.append(trace)
        
fig = {
    "data": data,
    "layout" : {
        "title": "Quantidade de cavalos por veículo híbrido (True/False)",
        "yaxis": {
            "title":"Quantidade de cavalos",
            "zeroline": False,
        },
        "xaxis": {
            "title":"Híbrido",
            "zeroline": False,
        }
    }
}

iplot(fig, filename='violin/multiple', validate = False)

