##import the libary using to process the data

from pandas import ExcelWriter

from pandas import ExcelFile

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.tools as tls

data = pd.read_excel('../input/Chapter2OnlineData.xls', sheet_name='Figure2.6')

data
corr = data.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(data.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(data.columns)

ax.set_yticklabels(data.columns)

plt.show()
# Create data

x = data['Happiness score']

y = data['Explained by: GDP per capita']

colors = (0,0,0)



# Plot

plt.scatter(x, y,color = 'pink')

plt.title('GDP per capita effect to happiness score')

plt.xlabel('Happiness score')

plt.ylabel('GDP per capita')

plt.show()



# Create data

x = data['Happiness score']

y = data['Dystopia (1.88) + residual']

colors = (0,0,0)



# Plot

plt.scatter(x, y,color = 'm')

plt.title('Dystopia effect to happiness score')

plt.xlabel('Happiness score')

plt.ylabel('Dystopia')

plt.show()



# Create data

x = data['Happiness score']

y = data['Explained by: Social support']

colors = (0,0,0)



# Plot

plt.scatter(x, y,color = 'c')

plt.title('Social support effect to happiness score')

plt.xlabel('Happiness score')

plt.ylabel('Dystopia')

plt.show()



#!pip install bubbly

#from bubbly.bubbly import bubbleplot 

#gdp = pd.read_excel('../input/gdp-vs-happiness.csv')

#figure = bubbleplot(dataset=gdp-vs-happiness, x_column='gdp.iloc[:,3]', y_column='gdp.iloc[:,4]', 

#   bubble_column='country', time_column='year', size_column='pop', color_column='continent', 

#   x_title="GDP per Capita", y_title="Life Satisfaction", title='Gapminder Global Indicators',

#    x_logscale=True, scale_bubble=3, height=650)



#iplot(figure, config={'scrollzoom': True})