# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/zomato.csv',encoding = "ISO-8859-1")
Country = pd.read_excel('../input/Country-Code.xlsx')


data = pd.merge(df, Country,on='Country Code')
data.head()

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as go

Temp = data['Country'].value_counts()
data1 = [go.Choropleth(
    locationmode='country names',
    locations=Temp.index.values,
    text=Temp.index,
    z=Temp.values
)]
layout = go.Layout(
    title='Number of Resturants Registered on Zomato by Country',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)


data1 = [go.Bar(x=Temp.index,
            y=Temp.values)]
layout = go.Layout(
    title='Number of Resturants Registered on Zomato by Country',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)
Temp=(data.groupby(['Country'], as_index=False)['Aggregate rating'].mean())

data1 = ([go.Choropleth(
    locationmode='country names',
    locations=Temp['Country'],
    text=Temp['Country'],
    z=Temp['Aggregate rating']
)])
layout = go.Layout(
    title='Average Rating on Zomato by Country',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)


data1 = ([go.Bar(x=Temp['Country'],
    y=Temp['Aggregate rating'])])
layout = go.Layout(
    title='Average on Zomato by Country',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)
Cusine_data=(data.groupby(['Cuisines'], as_index=False)['Restaurant ID'].count())
Cusine_data.columns = ['Cuisines', 'Number of Resturants']
Cusine_data['Mean Rating']=(data.groupby(['Cuisines'], as_index=False)['Aggregate rating'].mean())['Aggregate rating']
#Cusine_data.sort_values(['Number of Resturants'],ascending=False).head(20)
TwentyMostPopularCusines = (Cusine_data.sort_values(['Number of Resturants'],ascending=False).head(20))['Cuisines']
Top20 = Cusine_data.sort_values(['Number of Resturants'],ascending=False).head(20)
Cusine_data.sort_values(['Number of Resturants'],ascending=False).head(20)
data1 = ([go.Bar(x=Top20['Cuisines'],
    y=Top20['Number of Resturants'])])
layout = go.Layout(
    title='Number of Resturants on Zomato by Cuisines',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)
DataForTwentyMostPopularCusines = (data[data['Cuisines'].isin(TwentyMostPopularCusines)])
TwentyMostPopularCusines= TwentyMostPopularCusines.reset_index()['Cuisines']
a=[None] * 20
b= [None] * 20
trace = [None] * 20
for i in range (0,20):
    TwentyMostPopularCusines[i]
    a[i]= DataForTwentyMostPopularCusines[DataForTwentyMostPopularCusines['Cuisines']==TwentyMostPopularCusines[i]]['Aggregate rating']
    b[i] =a[i].reset_index()['Aggregate rating']
    trace[i] = go.Box(
    y=b[i],
    name = TwentyMostPopularCusines[i],
        boxmean=True
    )

data1 = trace
layout = go.Layout(
    title='How are these 20 Cuisines rated ',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

#mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
             #72 


stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_font_size=200, 
                         ).generate((data['Restaurant Name']).to_string())

print(wordcloud)
fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
Data_India = data[data['Country']=='India']
Resturants_India = (Data_India.groupby(['City'], as_index=False)['Aggregate rating'].count()).sort_values(by='Aggregate rating',ascending=False)
Resturants_India.columns = ['City', 'Number of Resturants']

trace0 = go.Bar(
    x=Resturants_India['City'],
    y=Resturants_India['Number of Resturants'],
    )

data1 = [trace0]
layout = go.Layout(
    title='Number of Resturants on Zomato by City',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)
Resturants_India=(Data_India.groupby(['City'], as_index=False)['Aggregate rating'].mean()).sort_values(by='Aggregate rating',ascending=False)


trace0 = go.Bar(
    x=Resturants_India['City'],
    y=Resturants_India['Aggregate rating'],
    )

data1 = [trace0]
layout = go.Layout(
    title='Average Rating on Zomato by City',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)
Delhi_Data = data[data['City']=='New Delhi']
Ggn_Data = data[data['City']=='Gurgaon']
Noida_Data =data[data['City']=='Noida']
Delhi_Data=(Delhi_Data.groupby(['Cuisines'])['Aggregate rating'].agg(['mean', 'count'])).reset_index()
Delhi_Data = Delhi_Data[Delhi_Data['count']>10]
Delhi_Data= Delhi_Data.sort_values(by='mean',ascending=False)
trace0 = go.Bar(
    x=Delhi_Data['Cuisines'],
        y=Delhi_Data['mean'],
    )

data1 = [trace0]
layout = go.Layout(
    title='Popular Cuisines in Delhi',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)


Ggn_Data=(Ggn_Data.groupby(['Cuisines'])['Aggregate rating'].agg(['mean', 'count'])).reset_index()
Ggn_Data = Ggn_Data[Ggn_Data['count']>10]
Ggn_Data= Ggn_Data.sort_values(by='mean',ascending=False)
trace0 = go.Bar(
    x=Ggn_Data['Cuisines'],
        y=Ggn_Data['mean'],
    )

data1 = [trace0]
layout = go.Layout(
    title='Popular Cuisines in Gurgaon',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)
Noida_Data=(Noida_Data.groupby(['Cuisines'])['Aggregate rating'].agg(['mean', 'count'])).reset_index()
Noida_Data = Noida_Data[Noida_Data['count']>10]
Noida_Data= Noida_Data.sort_values(by='mean',ascending=False)
trace0 = go.Bar(
    x=Noida_Data['Cuisines'],
        y=Noida_Data['mean'],
    )

data1 = [trace0]
layout = go.Layout(
    title='Popular Cuisines in Noida',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig)
data.head()
data1 = [dict(
    type='scattergeo',
    lon = data['Longitude'],
    lat = data['Latitude'],
    text = data['Restaurant Name'],
    mode = 'markers',
    marker = dict(
    cmin = 0,
    color = data['Price range'],
    cmax = data['Price range'].max(),
    colorbar=dict(
                title="Price Range"
            )
    )
    
)]
layout = dict(
    title = 'Where are all these Resturants',
    hovermode='closest',
    geo = dict(showframe=False, countrywidth=1, showcountries=True,
               showcoastlines=True, projection=dict(type='Mercator'))
)
fig = go.Figure(data=data1, layout=layout)
iplot(fig)