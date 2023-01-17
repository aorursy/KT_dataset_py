# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


data1=pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding="windows-1252")
data1.head(50)
data1.info()
data1.state.unique()
data1.number.replace(['-'],0.0,inplace = True)

data1.number.replace([' '],0.0,inplace = True)

data1.head(-10)
state_list = list(data1.state.unique())
state_forest_fires_ratio = []

for i in state_list:

    x = data1[data1.state==i]

    area_forest_fire_rate=sum(x.number)/sum(data1.number)*100

    state_forest_fires_ratio.append(area_forest_fire_rate)

data2 = pd.DataFrame({'state_list': state_list,'state_forest_fires_ratio':state_forest_fires_ratio})

new_index = (data2.state_forest_fires_ratio.sort_values(ascending=False)).index.values

sorted_data = data2.reindex(new_index)
plt.figure(figsize=(15,15))

sns.barplot(x=sorted_data['state_list'], y=sorted_data['state_forest_fires_ratio'])

plt.xticks(rotation= 70)

plt.xlabel('States')

plt.ylabel('Forest Fire Rate')

plt.title('Forest Fire Rate Given States')

plt.show()
state_forest_fires_number = []

for i in state_list:

    xx = data1[data1.state==i]

    area_forest_fire_number=sum(xx.number)

    state_forest_fires_number.append(area_forest_fire_number)

data3 = pd.DataFrame({'state_list': state_list,'state_forest_fires_number':state_forest_fires_number})

data33 = data3.sort_values(['state_forest_fires_number']).reset_index(drop=True)

plt.figure(figsize=(15,15))

sns.barplot(x='state_list',y='state_forest_fires_number',data=data33)

plt.xticks(fontsize=15, rotation=80)

plt.yticks(fontsize=15, rotation=45)

plt.title('Fires by Year', fontsize = 20)

plt.ylabel('Number of fires', fontsize=15)

plt.xlabel('Years', fontsize=15)

plt.show()
datag=data1

data4=datag.groupby('year')['number'].sum().reset_index()

data4
f,ax1 = plt.subplots(figsize =(15,15))

sns.pointplot(x='year',y='number',data=data4,color='coral',alpha=0.9)

plt.xlabel('Years',fontsize = 15,color='red')

plt.ylabel('Numbers',fontsize = 15,color='red')

plt.title('Number of Fires Per Year',fontsize = 25,color='purple')

plt.show()

g = sns.jointplot(data4.year,data4.number, kind="kde", size=7)

g = sns.jointplot("year", "number", data=data4,size=5, ratio=3, color="r")

plt.show()
labels = data4.year

plt.figure(figsize = (15,15))

sizes = data4.number

explode = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

plt.pie(sizes,explode=explode, labels=labels, autopct='%1.2f%%')

plt.title('Forest Fires According to Years',color = 'grey',fontsize = 25,style='italic')

plt.show()
sns.lmplot(x="year", y="number", data=data4) #Show the results of a linear regression within each dataset

plt.show()
# Show each distribution with both violins and points

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3) 

sns.violinplot(data=data4, palette=pal, inner="points")

plt.show()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data4.corr(), annot=True, linewidths=0.5,linecolor="yellow", fmt= '.1f',ax=ax)

plt.show()
sns.pairplot(data4)

plt.show()
trace1 = go.Scatter(

                    x = data4.year,

                    y = data4.number,

                    mode = "lines",

                    name = "year",

                    marker = dict(color = 'rgba(15, 110, 2, 0.8)'),

                    text= data4.year)

data = [trace1]

layout = dict(title = 'Forest Fires According to Years',

              xaxis= dict(title= 'years',ticklen= 4,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

trace1 = go.Scatter(

                    x = data4.year,

                    y = data4.number,

                    mode = "markers",

                    name = "year",

                    marker = dict(color = 'rgba(15, 110, 2, 0.8)'),

                    text= data4.year)

data = [trace1]

layout = dict(title = 'Forest Fires According to Years',

              xaxis= dict(title= 'years',ticklen= 4,zeroline= False),

              yaxis= dict(title= 'numbers',ticklen= 4,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Bar(

                    x = data4.year,

                    y = data4.number,

                    

                    name = "year",

                    marker = dict(color = 'rgba(15, 110, 2, 0.8)'),

                    text= data4.year)

data = [trace1]

layout = dict(title = 'Forest Fires According to Years',

              xaxis= dict(title= 'years',ticklen= 4,zeroline= False),

              yaxis= dict(title= 'numbers',ticklen= 4,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
pie1 = data4.number

labels = data4.year

# figure

fig = {

  "data": [

    {

      "values": pie1,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of Forest Fires",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Forest Fires of Brazil Rates",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Forest Fires",

                "x": 2,

                "y": 2

            },

        ]

    }

}

iplot(fig)
x2010 = data1[data1.year == 2010]

trace1 = go.Box(

                    

                    y = x2010.number,

                    name = "number",

                    marker = dict(color = 'rgba(15, 110, 2, 0.8)'),

                    text= data4.year)

data = [trace1]

layout = dict(

              xaxis= dict(title= '2010',ticklen= 4,zeroline= False),

              yaxis= dict(title= 'numbers',ticklen= 4,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)