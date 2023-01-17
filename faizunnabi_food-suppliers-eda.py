import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
%matplotlib inline
import os
os.listdir()
df = pd.read_csv('../input/FAO.csv',encoding='latin-1')
df.head()
df.tail()
df.info()
df.describe()
df.columns
#list of countries in the dataset
temp = df['Area'].value_counts()
temp
print("Total countries in the list : %d" %len(df.Area.unique()))
sns.set_style('darkgrid')
plt.figure(figsize=(16,28))
sns.barplot(y=temp.index,x=temp.values)
#Top 10 food supplying coutries
temp5= temp.head(10)
plt.figure(figsize=(16,9))
sns.barplot(x=temp5.index,y=temp5.values)
tempI = df['Item']
items = tempI.values
items_text = ''
for i in items:
    items_text += ' '+i

wd = WordCloud().generate(items_text)
plt.figure(figsize=(16,9))
plt.imshow(wd, interpolation="bilinear")
plt.axis("off")
plt.show()
temp5 = df.groupby('Area').sum().iloc[:,5:]
tempT = temp5.nlargest(10,'Y2013').iloc[:,-1]
tempT
plt.figure(figsize=(16,9))
sns.barplot(x=tempT.index,y=tempT.values)
plt.title('Top 10 Food suppliers in 2013',{'fontsize': 20,'fontweight' :'bold'})
plt.ylabel('Quantity  (Tonnes)',{'fontweight' :'bold'})
plt.xlabel('Country',{'fontweight' :'bold'})
#Analysing year wise distribution
tempT = temp5.nlargest(10,'Y2013')
plt.figure(figsize=(16,9))
sns.heatmap(tempT,cmap='coolwarm',)
plt.title('Food suppliers trend',{'fontsize': 20,'fontweight' :'bold'})
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


init_notebook_mode(connected=True) 
data = [ dict(
        type = 'scattergeo',
        lon = df['longitude'],
        lat = df['latitude'],
        text = df['Item'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            )
            
        ))]

layout = dict(
    autosize=True,
    title="Top Suppliers Locations and Items Supplied",
    hovermode='closest',
    mapbox=dict(
        bearing=0,
        pitch=0,
        zoom=1,
        center =dict(lat=40.7143528,lon=-74.0059731)
    ),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = df['Area Abbreviation'].value_counts()
data = dict(
        type = 'choropleth',
        locations = temp.index,
        z = temp.values,
        text = temp.values,
        colorbar = {'title' : 'Suppliers'},
      ) 

layout = dict(
    title = 'Major Suppliers',
    geo = dict(
        showframe = False,
        projection = {'type':'Mercator'}
    )
)


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
