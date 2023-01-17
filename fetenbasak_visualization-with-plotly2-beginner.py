# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data2016=pd.read_csv('../input/2016.csv')
data2016.head()
data2016.info()

#Happiness Rank and Happiness Score for Each Country
#LINE PLOT
trace=go.Scatter(x=data2016['Happiness Rank'],
                y=data2016['Happiness Score'],
                text=data2016.Country,
                marker=dict(color='rgba(220,20,208,0.7)',size=10))
data=[trace]
layout={'title':'Happiness Rank and Happiness Score for Each Country',
       'xaxis':{'title':'World Rank','zeroline':False},
       'yaxis':{'title':'Happiness Score','zeroline':False}}
iplot({'data':data,'layout':layout})
#Happines Score in 2016
#HISTOGRAM
trace=go.Histogram(x=data2016['Happiness Score'])

data=[trace]
layout={'title':'Happiness Score in 2016',
       'xaxis':{'title':'Happiness Score'},
       'yaxis':{'title':'Value'}}
iplot({'data':data,'layout':layout})
#BAR CHART
#Average Happiness Score According to Region

region_list=list(data2016.Region.unique())
average_score=[]

for i in region_list:
    x=data2016[data2016.Region==i]
    average_score.append(sum(x['Happiness Score'])/len(x))
df1=pd.DataFrame({'region_list':region_list,'average_score':average_score})

new_index=df1.average_score.sort_values(ascending=True).index.values
sorted_data=df1.reindex(new_index)

trace=go.Bar(x=df1.region_list, y=df1.average_score,
            marker=dict(color='rgba(240,20,230,0.7)'))
data=[trace]
layout={'title':'Average Happiness Score According to Region',
       'xaxis':{'title':'Region','tickangle':-90,'zeroline':False},
       'yaxis':{'title':'Happinness Score'}}
iplot({'data':data,'layout':layout})
    

#Scatter Plot Matrix

# import figure factory
import plotly.figure_factory as ff

df=data2016.loc[:,['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom']]
df['index']=np.arange(1,len(df)+1)

fig=ff.create_scatterplotmatrix(df, index='index',colormap='Rainbow',colormap_type='cat',height=800,width=800)
iplot(fig)

#3d scatter plot matrix
#Happiness Score vs Happiness Rank and Freedom

trace=go.Scatter3d(x=data2016['Happiness Rank'],y=data2016['Happiness Score'],z=data2016['Freedom'],
                  mode='markers',
                  marker=dict(color='rgba(20,120,140,0.8)',size=10),
                  text=data2016.Country)
data=[trace]

layout={'title':'Happiness Score vs Happiness Rank and Freedom'}

iplot({'data':data,'layout':layout})
