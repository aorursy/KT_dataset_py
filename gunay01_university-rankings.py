# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import itertools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load data
data1=pd.read_csv("../input/cwurData.csv")
data2=pd.read_csv("../input/shanghaiData.csv")
data3=pd.read_csv("../input/timesData.csv")
# analyzing data1
data1.info()
data1.head()
data1.tail()
data1.corr()
#correleation map
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data1.corr(),annot=True,linewidths=.3,fmt=".1f",ax=ax)
data1.columns
year_2012=data1["year"]==2012
year_2013=data1["year"]==2013
year_2014=data1["year"]==2014
year_2015=data1["year"]==2015
data1[year_2015].world_rank.plot(kind = 'line', color = 'red',label = 'World rank',linewidth=1,alpha = 0.9,grid = True,linestyle = '-')
data1[year_2015].publications.plot(color = 'orange',label = 'Publications',linewidth=1, alpha = 0.9,grid = True,linestyle = '--')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('World rank-publications plot')            # title = title of plot
plt.show()
data1[year_2015].plot(kind='scatter', x='publications', y='world_rank',alpha = 0.5,color = 'blue')
plt.xlabel('Publications')              # label = name of label
plt.ylabel('World rank')
plt.title('World rank-publications with scatter plot')            # title = title of plot
plt.show()
data1[year_2015].score.plot(kind='hist',bins=20,figsize=(10,10),color='r')
plt.show()
data1[(data1["year"]==2015) & (data1["country"]=="USA") & (data1["world_rank"]<11)]
# prepare data frame
df = data2.iloc[4397:4899,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.pub,
                    mode = "lines",
                    name = "publications",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df.university_name)
# Creating trace2
trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.award,
                    mode = "lines+markers",
                    name = "Nobel prizes",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.university_name)
data = [trace1, trace2]
layout = dict(title = 'Publications and Nobel Prizes - World Rank of Top 100 Universities in 2015',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# prepare data frame
df2013 = data1[data1.year == 2013].iloc[:100,:]
df2014 = data1[data1.year == 2014].iloc[:100,:]
df2015 = data1[data1.year == 2015].iloc[:100,:]

# creating trace1
trace1 =go.Scatter(
                    x = df2013.world_rank,
                    y = df2013.publications,
                    mode = "markers",
                    name = "2013",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2013.institution)
# creating trace2
trace2 =go.Scatter(
                    x = df2014.world_rank,
                    y = df2014.publications,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2014.institution)
# creating trace3
trace3 =go.Scatter(
                    x = df2015.world_rank,
                    y = df2015.publications,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2015.institution)

data = [trace1, trace2, trace3]

layout = dict(title = 'Publications of top 100 universities in 2013, 2014 2015',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Publications',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
trace1 = go.Scatter3d(
    x=df2015.national_rank,
    y=df2015.alumni_employment,
    z=df2015.score,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(200,0,0)',                     
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
feature_names = "citations","alumni_employment","publications","quality_of_education"
features_size = [len(df2015.citations),len(df2015.alumni_employment),len(df2015.publications),len(df2015.quality_of_education)]
# create a circle for the center of plot
circle = plt.Circle((0,0),0.4,color = "white")
plt.pie(features_size, labels = feature_names, colors = ["red","green","blue","purple"] )
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("Number of Each Features")
plt.show()