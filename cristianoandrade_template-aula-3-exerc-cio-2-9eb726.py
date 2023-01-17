import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
#Ultilizando o dataset cars
file_path = '../input/carros/'
file_name = 'cars.csv'
cars = pd.read_csv(file_path + file_name,delimiter = ',')

cars.head(10)
resposta = [["City mpg","Quantitativa Ordinal"],["Classification","Qualitativa Nominal"],["Driveline","Qualitativa Nominal"],
            ["Engine Type","Qualitativa Nominal"],["Fuel Type","Quantitativa Nominal"],["Height","Quantitativa Ordinal"],
            ["Highway mpg","Quantitativa Ordinal"],["Horsepower","Quantitativa Ordinal"],["Hybrid","Quantitativa Nominal"],
            ["ID","Quantitativa Ordinal"],["Length","Quantitativa Ordinal"],["Make","Quantitativa Nominal"],
            ["Model Year","Qualitativa Ordinal"],["Number of Forward Gears","Quantitativa Ordinal"],["Torque","Quantitativa Ordinal"],
           ["Transmission","Quantitativa Ordinal"],["Width","Quantitativa Ordinal"],["Year","Quantitativa Ordinal"]]

variaveis = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
variaveis
ClassificationYear = cars[['Classification', 'Year']].groupby(['Year'])['Classification'].value_counts().rename_axis(['Year','Classification2']).reset_index()
ClassificationYear.columns = ['Year', 'Classification', 'total']
ClassificationYear
#grafico de linha
trace0 = go.Scatter(
      x = ClassificationYear[ClassificationYear['Classification']=='Automatic transmission']['Year'],
      y = ClassificationYear[ClassificationYear['Classification']=='Automatic transmission']['total'],
      name ='Automatico',
      line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4))

trace1 = go.Scatter(
      x = ClassificationYear[ClassificationYear['Classification']=='Manual transmission']['Year'],
      y = ClassificationYear[ClassificationYear['Classification']=='Manual transmission']['total'],
      name ='Manual',
      line = dict(
        color = ('rgb(0, 255, 100)'),
        width = 4))
data = [trace0, trace1]

layout = dict(title = 'VENDAS - Manual X Automatico',
              xaxis = dict(title = 'Anos'),
              yaxis = dict(title = 'Total de vendas'),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-line')
FuelTypeYear = cars[['Fuel Type', 'Year']].groupby(['Year'])['Fuel Type'].value_counts().rename_axis(['Year','Classification2']).reset_index()
FuelTypeYear.columns = ['Year', 'Fuel Type', 'total']
FuelTypeYear
#grafico de barras
trace0 = go.Bar(
      x = FuelTypeYear[FuelTypeYear['Fuel Type']=='Gasoline']['Year'],
      y = FuelTypeYear[FuelTypeYear['Fuel Type']=='Gasoline']['total'],
      name ='Gasoline')

trace1 = go.Bar(
     x = FuelTypeYear[FuelTypeYear['Fuel Type']=='Diesel fuel']['Year'],
     y = FuelTypeYear[FuelTypeYear['Fuel Type']=='Diesel fuel']['total'],
     name ='Diesel fuel')

    
trace2 = go.Bar(
      x = FuelTypeYear[FuelTypeYear['Fuel Type']=='E85']['Year'],
      y = FuelTypeYear[FuelTypeYear['Fuel Type']=='E85']['total'],
      name ='E85')

trace3 = go.Bar(
      x = FuelTypeYear[FuelTypeYear['Fuel Type']=='Compressed natural gas']['Year'],
      y = FuelTypeYear[FuelTypeYear['Fuel Type']=='Compressed natural gas']['total'],
      name ='Compressed natural gas	')

data = [trace0, trace1, trace2, trace3]

layout = dict(title = 'Comparação entre vendas de carros por tipo de combustivel',
              xaxis = dict(title = 'Years'),
              yaxis = dict(title = 'Total Sales'),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar-direct-labels')
#Histograma
trace0 = go.Histogram(
      x = FuelTypeYear['Fuel Type'],
      name ='Year')

data = [trace0]
layout = dict(title = 'Histograma de combustiveis mais utilizados',
              yaxis = dict(title = 'Quantidade'))
              
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='stacked histogram')
#Lendo o dataset Pokemon
file_path = '../input/pokemon/'
file_name = 'Pokemon.csv'
df = pd.read_csv(file_path + file_name,delimiter = ',')

df.head(10)
#grafico de radar
mask = df['Name'] == 'Squirtle'
x0 = df[mask]
x1 = x0[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]

data = [go.Scatterpolar(
  r = x1.values.ravel(),
  theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'],
  fill = 'toself'
)]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 100],
    ),
  ),
  title = 'Skills do Squirtle',
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
#grafico boxplot
trace0 = go.Box( y=df['Attack'], name = 'Attack' )
trace1 = go.Box( y=df['Defense'], name = 'Defense')
py.iplot([trace0, trace1])
#grafico de violino
data = []
for p in df[0:0].keys().tolist()[5:11]:
    trace = {
        "type": 'violin', "y": df[p], "name": p,
        "box": { "visible": True },
        "meanline": { "visible": True },
    }
    data.append(trace)

py.iplot(data, filename = 'violin/basic', validate = False)
