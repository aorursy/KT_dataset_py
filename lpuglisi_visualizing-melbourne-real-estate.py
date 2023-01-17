import pandas as pd

from IPython.display import display

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()

import seaborn as sns

%matplotlib inline

import numpy as np
data = pd.read_csv("../input/Melbourne_housing_extra_data.csv")

data.Date = pd.to_datetime(data.Date)

data = data[data['Price']>0]
display(data.describe())

print("\r")

print("Count of null entries")

display(data.isnull().sum())
pl1 = go.Histogram(x = data.Price)

layout1 = go.Layout(title='Histogram of Housing Prices (01/2016 - 10/2017)',

                   xaxis=dict(title='Price',titlefont=dict(family='Courier New, monospace',size=16, color='#7f7f7f')),

                   yaxis=dict(title='Frequency',titlefont=dict(family='Courier New, monospace',size=16,color='#7f7f7f'))

                 )

dt1 = [pl1]

fig1 = go.Figure(data=dt1, layout=layout1)

iplot(fig1)
#Estimating the commission and plotting the top 20 agents by commission

data['comm'] = data.Price*0.02

dt2 = data.groupby("SellerG").comm.sum().sort_values(ascending=False)

dt2 = dt2.nlargest(20)

pl2 = go.Bar(x=dt2.index, y=dt2.values)

layout2 = go.Layout(title='Top 20 Total Commission by Agent (01/2016 - 10/2017)',

                   xaxis=dict(title='Real-Estate Agent',titlefont=dict(family='Courier New, monospace',size=16, color='#7f7f7f')),

                   yaxis=dict(title='Total Commission (Assuming 2%)',titlefont=dict(family='Courier New, monospace',size=16,color='#7f7f7f'))

                 )

dt2 = [pl2] 

fig2 = go.Figure(data=dt2, layout=layout2)

iplot(fig2)

#Finding percent of total for the top 20 agents

n = data.groupby("SellerG").comm.sum().nlargest(20).sum()

d = data.comm.sum()

print("Top 20 Agents Percent of Total: %.2f" %((n/d)*100))
dt31 = data.groupby(["Date"]).Price.mean()

dt31 = dt31.reset_index()

dt32 = data.groupby(["Date"]).Price.max()

dt32 = dt32.reset_index()

dt33 = data.groupby(["Date"]).Price.min()

dt33 = dt33.reset_index()



pl31 = go.Scatter(x = dt31.Date, y=dt31.Price, mode="lines+markers", name='Avg', line= dict(color = ('rgb(255,165,0)'), width=4))

pl32 = go.Scatter(x = dt32.Date, y=dt32.Price, mode="lines", name='Max', line= dict(color = ('rgb(192,192,192)'), width=2)) 

pl33 = go.Scatter(x = dt33.Date, y=dt33.Price, mode="lines", name='Min', line= dict(color = ('rgb(192,192,192)'), width=2)) 



dt3 = [pl31, pl32, pl33]

layout3 = go.Layout(title='Min/Max/Avg Melbourne Housing Prices',

                   xaxis=dict(title='Date Sold',titlefont=dict(family='Courier New, monospace',size=16, color='#7f7f7f')),

                   yaxis=dict(title='Price',titlefont=dict(family='Courier New, monospace',size=16,color='#7f7f7f'))

                 )



fig3 = go.Figure(data=dt3, layout=layout3)

iplot(fig3)
#Filtering the data to only include entries with the YearBuilt field populated

data2 = data[data['YearBuilt']>0]

#First, plotting the entries based on the distance from the city center

pl4 = go.Scatter(x = data2.Longtitude, 

                 y = data2.Lattitude, 

                 mode='markers',

                marker = dict(color = data2.Distance, colorscale='Jet', reversescale=True, showscale=True)

                )

layout4 = go.Layout(title='Homes Sold Showing Distance from Melbourne CBD',

                   xaxis=dict(title='Longitude',titlefont=dict(family='Courier New, monospace',size=16, color='#7f7f7f')),

                   yaxis=dict(title='Latitude',titlefont=dict(family='Courier New, monospace',size=16,color='#7f7f7f'))

                 )

#Next, plotting the entries based on the age of the homes sold

pl5 = go.Scatter(x = data2.Longtitude, 

                 y = data2.Lattitude, 

                 mode='markers',

                marker = dict(color = data2.YearBuilt, colorscale='Jet', reversescale=True, showscale=True)

                )

layout5 = go.Layout(title='Homes Sold Showing the Year the Home was Built',

                   xaxis=dict(title='Longitude',titlefont=dict(family='Courier New, monospace',size=16, color='#7f7f7f')),

                   yaxis=dict(title='Latitude',titlefont=dict(family='Courier New, monospace',size=16,color='#7f7f7f'))

                 )





fig4 = go.Figure(data=[pl4], layout=layout4)

fig5 = go.Figure(data=[pl5], layout=layout5)



iplot(fig4)

iplot(fig5)