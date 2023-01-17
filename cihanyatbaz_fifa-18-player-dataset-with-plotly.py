# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected =True)
import plotly.graph_objs as go
from plotly import tools
from wordcloud import WordCloud     #word cloud library
import matplotlib.pyplot as plt     #matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Prepare Data
fifa_data = pd.read_csv('../input/CompleteDataset.csv')
plt.show()
fifa_data.head()
# In order to be able to start our ranking from 1, we have assigned the 'array' variable.
fifa_data['array'] = fifa_data['Unnamed: 0']+1
# Let's change the names of some columns
fifa_data.rename(columns={'Ball control':'ball_control','Free kick accuracy':'free_kick',
                     'Shot power':'shot_power'}, inplace=True)
# Just take the columns we'll use
fifa_data= fifa_data[['array','Name','Age','Nationality','Overall','Potential','Club','Value',
                      'Wage','Special','Acceleration','ball_control','Dribbling','free_kick',
                      'Penalties','shot_power']]
fifa_data.head()
fifa_data.info()
# Prepare Data Frame
d_frame = fifa_data.iloc[:100,:]

#Creating trace1
trace1 = go.Scatter(
    x= d_frame.array,
    y= d_frame.Overall,
    mode='lines',
    name='Overall',
    marker= dict(color='rgba(12, 255, 250,0.9)'),
    text= d_frame.Name
)
#Creating trace2
trace2 = go.Scatter(
    x= d_frame.array,
    y= d_frame.Potential,
    mode='lines+markers',
    name='Potential',
    
    text= d_frame.Name
)
data = [trace1,trace2]
layout= dict(title="Comparing the 'Overall' and 'Potential' of players",
            xaxis= dict(title='Player Rank', ticklen=5, zeroline=False)
            )
fig= dict(data=data, layout=layout)
iplot(fig)
# Prepare Data Frame
d_frame = fifa_data.iloc[90:100,:]

#create trace1
trace1 = go.Bar(
    x= d_frame.array,
    y= d_frame.Overall,
    name= 'Overall',
    marker= dict(color= 'rgba(255,106,0,0.9)',
                line= dict(color= 'rgb(0,0,0)', width=1)),
    text= d_frame.Name
)
#Create trace2
trace2 = go.Bar(
    x= d_frame.array,
    y= d_frame.Potential,
    name= 'Potential',
    marker= dict(color= 'rgba(148, 255, 130,0.9)',
                line= dict(color='rgb(0,0,0)', width=1)),
)
data= [trace1, trace2]
layout= go.Layout(barmode= "group")
fig= go.Figure(data=data, layout=layout)
iplot(fig)
#Prepare Data Frame
d_frame = fifa_data.iloc[:5,:]

#Create trace1
trace1= {
    'x': d_frame.array,
    'y': d_frame.free_kick,
    'name': 'Free Kick',
    'type': 'bar',
    'text': d_frame.Name
};
#Create trace2
trace2= {
    'x': d_frame.array,
    'y': d_frame.Penalties,
    'name': 'Penalty',
    'type': 'bar',
    'text': d_frame.Name,
    'marker': dict(color= 'rgba(148, 255, 130,0.9)'),
};
data= [trace1, trace2]
layout= {
    'xaxis': {'title':'First 5 Player'},
    'barmode': 'relative',
    'title': 'Top 5 Players frikik and penalty strokes comparison'
};
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Prepare Data Frame
d_frame = fifa_data.iloc[:100,:]
donut= d_frame.Club.value_counts()
labels = d_frame.Club.value_counts().index

#Creat figure
fig = {
    "data":
    [
        {
            "values": donut,
            "labels": labels,
            "domain": {"x": [0, 1]},
            "name": "Clubs Rate",
            "hoverinfo": "label+percent+name",
            "hole": .4,
            "type": "pie"
        }, 
    ],
    "layout":
    {
        "title":"Club rates of the top 100 players",
        "annotations":
        [
            { 
                "font":{"size":20},
                "showarrow":False,
                "text": "",
                "x": 0,
                "y": 1
            },
        ]
    }
}
iplot(fig)
#Prepare Data Frame
d_frame = fifa_data.iloc[:100,:]
donut= d_frame.Nationality.value_counts()
labels = d_frame.Nationality.value_counts().index #Country names of the top 100 players

#Creat Figure
fig = {
    "data":
    [
        {
            "values": donut,
            "labels": labels,
            "domain": {"x": [0, 1]},
            "name": "Clubs Rate",
            "hoverinfo": "label+percent+name",
            "hole": .4,
            "type": "pie"
        }, 
    ],
    "layout":
    {
        "title":"Nationality rates of the top 100 players",
        "annotations":
        [
            { 
                "font":{"size":20},
                "showarrow":False,
                "text": "",
                "x": 0,
                "y": 1
            },
        ]
    }
}
iplot(fig)
#Prepare Data Frame
d_frame = fifa_data.Name[:50]

plt.subplots(figsize=(10,10))
wordcloud = WordCloud(
                   background_color='White',
                        width = 700,
                        height = 400
    ).generate(" ".join(d_frame))

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
#Prepare Data Frame
d_frame = fifa_data.iloc[:10,:]

#Create trace
trace = go.Scatter3d(
    x=d_frame.ball_control,
    y=d_frame.Dribbling,
    z=d_frame.shot_power,
    text= d_frame.Name,
    mode='markers',
    marker=dict(
        size=12,
        #color= z,          #set color to an array/list of desired value (plotly.ly)
#When we enters 'Fork Notebook' he describes 'z'. But why doesn't he recognize this right now? 
        colorscale='Viridis',   #Choose a colorscale
        opacity=0.8
    )
)
data = [trace]
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
#We'll edit 'Wage' to compare our values                                       
#removes € -> it will be like that: '£100K' -> '100K'
wage = fifa_data['Wage'].map(lambda x: x.replace("€", ""))
wage.head()                                                                
                                                                                        
#removes K -> it will be like that: '100K' -> '100'
wage = wage.map(lambda x: x.replace("K", ""))
wage.head()
#We'll edit 'Value' to compare our values                                                 
#removes € -> it will be like that: '£100M or £100K' -> '100M or 100K'
value = fifa_data['Value'].map(lambda x: x.replace("€", ""))
                                                                       
#removes M -> it will be like that: '100M' -> '100 or 100K'
value = value.map(lambda x: x.replace("M", ""))
value.head()
#removes K -> it will be like that: '100K' -> '100'
value = value.map(lambda x: x.replace("K", ""))
value.head()                                                                            
#We multiply by '1000' because we remove 'K'
wage = wage.astype("int")*1000
wage.head()
#We multiply by '1000000' because we remove 'M'
value = value.astype("float")*1000000
value.head()
#Let's create new columns now
fifa_data['wage']=wage
fifa_data['value']=value
fifa_data.head()
#Now let's see 20 players with the highest 'Wage'
fifa_data.sort_values("wage", ascending=False).head(20)
#Prepare Data Frame
d_frame = fifa_data.sort_values("wage", ascending=False).head(20)

#Create trace1
trace1 = go.Bar(
    x= d_frame.wage,
    opacity = 0.75,
    name= 'Wage',
    text= d_frame.Name,
    marker = dict(color='rgba(0, 250, 0,0.6)'))
#Create trace2
trace2 = go.Bar(
    x=d_frame.value,
    opacity = 0.75,
    name= "Value",
    text= d_frame.Name,
    marker= dict(color='rgba(26, 26, 26,0.6)'))

data = [trace1,trace2]
layout = go.Layout(barmode='stack',
                  title="Comparison of 'Wage' and 'Value' Among 20 Players with the Highest Wage",
                  xaxis= dict(title= 'Wage - Value'),
                  yaxis= dict(title='Array'),)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Prepare Data Frame
d_frame = fifa_data.iloc[:100,:]

#Create trace1
trace1 = go.Scatter(
    x=d_frame.array,
    y=d_frame.value,
    text= d_frame.Name,
    name='value',
    marker= dict(color='rgba(26, 163, 242,0.9)'),
    mode= 'lines+markers'
)
#Create trace2
trace2 = go.Scatter(
    x=d_frame.array,
    y=d_frame.wage,
    xaxis='x2',
    yaxis='y2',
    text= d_frame.Name,
    name='wage',
    marker= dict(color='rgba(70, 71, 71,0.9)'),
    mode= 'markers'
)
data = [trace1, trace2]
layout = go.Layout(
    title = "Top 100 players' wage and values",
    xaxis2=dict(
        domain=[0.3, 1],
        anchor='y2'
    ),
    yaxis2=dict(
        domain=[0.6, 1],
        anchor='x2'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
fifa_data.groupby("Club")["wage"].mean().sort_values(ascending=False).head(10)
#Prepare Data Frame
d_frame = fifa_data.groupby("Club")["wage"].mean().sort_values(ascending=False).head(10)

#Create trace
trace = go.Bar(
    x= fifa_data.array,
    y= d_frame,
    opacity = 0.75,
    text= fifa_data.Club,
    marker = dict(color='rgba(0, 250, 100,0.6)'))

data = [trace]
layout = go.Layout(barmode='group',
                  title='Top 10 teams paying maximum wage',
                  xaxis= dict(title= 'Array'),
                  yaxis= dict(title='Total Wage'),)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Now let's see the top 10 countries with the have more players
fifa_data.groupby("Nationality").Name.count().sort_values(ascending=False).head(10)
#How many players are from the same country?
df = fifa_data['Nationality'].value_counts()

iplot([
    go.Choropleth(
    locationmode='country names',
    locations=df.index.values,
    text= df.index,
    z=df.values,
    colorscale= 'Jet'
    )
])