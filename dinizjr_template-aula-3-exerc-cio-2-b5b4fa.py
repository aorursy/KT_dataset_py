%matplotlib inline
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import plotly.offline as py

from IPython.core.pylabtools import figsize

url ="https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/cars.csv"
car = pd.read_csv(url, sep=',')
car.columns = car.columns.str.replace(' ', '_')
car.head()
#car['Highway_mpg'].idxmax()#3686
cars = car.set_value(3686, 'Highway_mpg', 23)
cars.info()
col_used = cars[['City_mpg','Classification','Horsepower','Highway_mpg','Make']]
col_used.head()
marcas = pd.DataFrame(cars.Make.value_counts().reset_index(name='qtd').rename(columns={'index':'marcas'}))

trace = go.Scatter(x=marcas['marcas'],y=marcas['qtd'],mode = 'lines+markers')
data = [trace]
layout = go.Layout(
    title='Quantidade de veículos por marca',
    xaxis=dict(title='Marcas',titlefont=dict(size=20),tickangle=35),
    yaxis=dict(title='Quantidade',titlefont=dict(size=20)))
py.iplot(go.Figure(data = [trace], layout=layout))
tr = pd.DataFrame(cars.Classification.value_counts().reset_index(name='qtd').rename(columns={'index':'transmission'}))
trace = go.Bar(x=tr['transmission'],y=tr['qtd'])
data = [trace]
layout = go.Layout(
    title='Quantidade de veículos por transmissão',titlefont=dict(size=25),
    xaxis=dict(title='Transmissão',titlefont=dict(size=20)),
    yaxis=dict(title='Quantidade',titlefont=dict(size=20)))
py.iplot(go.Figure(data = [trace], layout=layout))
data = [go.Histogram(x=cars.Horsepower,nbinsx = 50)]
layout = dict(title='Histograma de potência dos veículos',titlefont=dict(size=25),
              xaxis= dict(title= 'Potência em cavalos',tickangle=25),
              yaxis = dict(title= 'Total de veículos'))

fig = dict(data = data, layout = layout)
iplot(fig)

cons_hwy = cars.Highway_mpg > 35
hwy = pd.DataFrame(cars[cons_hwy].Make.value_counts().nlargest(10).reset_index(name='qtd').rename(columns={'index':'marca'}))
hwy
cons_city = cars.City_mpg > 25
city = pd.DataFrame(cars[cons_city].Make.value_counts().nlargest(10).reset_index(name='qtd').rename(columns={'index':'marca'}))
city
trace1 = go.Scatterpolar(
  r = city.qtd,
  theta = city.marca,
  fill = 'toself',
  name='Consumo na Cidade (mais que 25 mpg)'
)
trace2 = go.Scatterpolar(
  r = hwy.qtd,
  theta = hwy.marca,
  fill = 'toself',
    name='Consumo na Estrada (mais que 35 mpg)'
)
data = [trace1,trace2]

layout = go.Layout(
  title = "Quantidade de veículos por marca que mais gastam(10 maiores)",
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 50]
    )
  ),
  showlegend = True
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Box(y= cars.Highway_mpg,name='Consumo na estrada')
trace2 = go.Box(y= cars.City_mpg,name='Consumo na cidade')
data = [trace1,trace2]
layout = dict(title = 'Consumo MPG(Milhas por galão)',
              xaxis= dict(ticklen= 5,zeroline= False),
              yaxis = dict(title= 'Total de consumo',
                            ticks='outside',
                            tick0=0,
                            dtick=5,
                            ticklen=10,
                            tickwidth=2
                          ),
              boxmode='group'
             )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
cons_h = cars.Highway_mpg
cons_c = cars.City_mpg
#cons_c.head()
fig = {
    "data": [{
            "type": 'violin',
            "y": cons_h,
            "name": 'consumo na estrada',
            "box": {"visible": True},
            "meanline": {"visible": True},
            "line": {"color": 'orange'}},
        {
            "type": 'violin',
            "y": cons_c,
            "name": 'consumo na cidade',
            "box": {"visible": True},
            "meanline": {"visible": True},
            "line": {"color": 'green'}
        }],
    "layout" : {"title": "Consumo em MPG na estrada e cidade", "yaxis": {"zeroline": False,},"violinmode": "group"}}
py.iplot(fig, filename = 'violin/multiple', validate = False)
