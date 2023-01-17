# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/honeyproduction.csv")
df.head()#5 samples in data
df.info() #features and data types
#heatmap correlation of features
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt=".2f",ax=ax,linecolor="red")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

#Bubble Chart
y=df[df.year==2008].sort_values("numcol",ascending=False).iloc[:5,:]
total_numcol  = [each/max(y.numcol)*100 for each in y.numcol]
international_color = [each for each in y.yieldpercol]
data = [
    {
        'y': y.prodvalue,
        'x': y.totalprod,
        'mode': 'markers',
        'marker': {
            'color': international_color,
            'size': total_numcol,
            'showscale': True
        },
        "text" :  y.state    
    }
]
iplot(data)


#Histogram
x1998=df.priceperlb[df.year==1998]
x2012=df.priceperlb[df.year==2012]

trace1 = go.Histogram(
    x=x1998,
    opacity=0.75,
    name = "1998",
    marker=dict(color='rgba(171,100, 96, 0.6)'))
trace2 = go.Histogram(
    x=x2012,
    opacity=0.75,
    name = "2012",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' priceperlb rates for 1998 and 2012',
                   xaxis=dict(title='priceperlb'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Pie-Chart
prod=df[df.year==2011].sort_values("totalprod",ascending=False).iloc[:5,:]
prod.num_stocks=[int(each) for each in prod.stocks]
labels=prod.state
fig = {
  "data": [
    {
      "values": prod.num_stocks,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Percent Of Stocks Rates",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie",
    },],
  "layout": {
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Percent of Stocks",
                "x": 0.40,
                "y": 1.2
            },
        ]
    }
}
iplot(fig)
#Bar Chart
x2010=df[df.year==2010].sort_values("prodvalue",ascending=True).iloc[:3,:]
trace1=go.Bar(
        x=x2010.state,
        y=x2010.totalprod,
        name="Total Production",
        marker=dict(color="rgba(175, 174, 200, 0.5)",
                     line=dict(color="rgb(0,0,0)",width=1.5)),
        text=x2010.state)
        
trace2=go.Bar(
        x=x2010.state,
        y=x2010.stocks,
        name="Stocks",
        text=x2010.state,
        marker=dict(color="rgba(150, 175, 128, 0.5)",
                    line=dict(color="rgb(0,0,0)",width=1.5)))        

data=[trace1,trace2]
layout=go.Layout(barmode="group")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
#Word Cloud
x2009 = df[df.year == 2009].sort_values("numcol",ascending=False).iloc[:20,:]
plt.subplots(figsize=(8,8))
a = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x2009.state))
plt.imshow(a)
plt.axis('off')
plt.show()
#Scatter Plot Matrix
import plotly.figure_factory as ff
# prepare data
dataframe = df[df.year == 1999].sort_values("priceperlb",ascending=False).iloc[:20,:]
data1999 = dataframe.loc[:,["numcol","totalprod", "stocks"]]
data1999["index"] = np.arange(1,len(data1999)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data1999, diag='box', index='index',colormap='Portland',
                                  colormap_type='seq',
                                  height=700, width=700)
plt.savefig('graph.png')
iplot(fig)

#3D Scatter 
# totalprod,yieldpercol,numcol
dataframe=df[df.year==2000]
trace1 = go.Scatter3d(
    x=dataframe.numcol,
    y=dataframe.totalprod,
    z=dataframe.yieldpercol,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,120,50)',        
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )  
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Multiple Subplots
dataframe=df[df.year==2002]
trace1 = go.Scatter(
    x=dataframe.state,
    y=dataframe.numcol,
    name = "numcol"
)
trace2 = go.Scatter(
    x=dataframe.state,
    y=dataframe.totalprod,
    xaxis='x2',
    yaxis='y2',
    name = "Total Prod."
)
trace3 = go.Scatter(
    x=dataframe.state,
    y=dataframe.stocks,
    xaxis='x3',
    yaxis='y3',
    name = "Stocks"
)
trace4 = go.Scatter(
    x=dataframe.state,
    y=dataframe.priceperlb,
    xaxis='x4',
    yaxis='y4',
    name = "Price Per Lb"
)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = 'Numcol, Total Prod., Stocks and Pricce Per Lb score VS State'
)
fig = go.Figure(data=data, layout=layout)
plt.savefig('graph.png')
iplot(fig)