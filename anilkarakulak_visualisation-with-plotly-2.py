import numpy as np 
import pandas as pd 

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
data=pd.read_csv("../input/data.csv")
data.info()
nationality=list(data["Nationality"].unique())
overall_average=[]
potential_average=[]
for i in nationality:
    x=data[data["Nationality"]==i]
    if len(x)!=0:
        overalls=round((sum(x.Overall)/len(x)),2)
        overall_average.append(overalls)
        potentials=round((sum(x.Potential)/len(x)),2)
        potential_average.append(potentials)
    else:
        overall_average.append("0")
        potential_average.append("0")
for i in range(len(nationality)):
    print(nationality[i],overall_average[i],potential_average[i])
df=pd.DataFrame({"nationality":nationality,"overall_average":overall_average,"potential_average":potential_average})

trace1 = go.Scatter(
                    x = df.nationality,
                    y = df.overall_average,
                    mode = "lines+markers",
                    name = "overall",
                    marker = dict(color = 'blue'),
                    text= df.nationality)

trace2 = go.Scatter(
                    x = df.nationality,
                    y = df.potential_average,
                    mode = "lines+markers",
                    name = "potential",
                    marker = dict(color = 'green'),
                    text= df.nationality)
data1 = [trace1, trace2]
layout = dict(title = 'Overall Abilities and Potential Average by Country',
              xaxis= dict(title= 'Nationality',ticklen= 5,zeroline= False)
             )
fig = dict(data = data1, layout = layout)
iplot(fig)
barcelona = data[data.Club == "FC Barcelona"].iloc[:20,:]
juventus = data[data.Club == "Juventus"].iloc[:20,:]
psg = data[data.Club == "Paris Saint-Germain"].iloc[:20,:]
man_utd = data[data.Club == "Manchester United"].iloc[:20,:]
man_city = data[data.Club == "Manchester City"].iloc[:20,:]

trace1 =go.Scatter(
                    x = barcelona.Overall,
                    y = barcelona.Potential,
                    mode = "markers",
                    name = "FC Barcelona",
                    marker = dict(color = 'green'),
                    text= barcelona.Name)
trace2 =go.Scatter(
                    x = juventus.Overall,
                    y = juventus.Potential,
                    mode = "markers",
                    name = "Juventus",
                    marker = dict(color = 'red'),
                    text= juventus.Name)
trace3 =go.Scatter(
                    x = psg.Overall,
                    y = psg.Potential,
                    mode = "markers",
                    name = "Paris Saint-Germain",
                    marker = dict(color = 'blue'),
                    text= psg.Name)
data1 = [trace1, trace2, trace3]
layout = dict(title = 'Abilities and Potential by Clubs',
              xaxis= dict(title= 'Potential',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Ability',ticklen= 5,zeroline= False)
             )
fig = dict(data = data1, layout = layout)
iplot(fig)
barcelona = data[data.Club == "FC Barcelona"].iloc[:20,:]
juventus = data[data.Club == "Juventus"].iloc[:20,:]
psg = data[data.Club == "Paris Saint-Germain"].iloc[:20,:]
man_utd = data[data.Club == "Manchester United"].iloc[:2,:]
man_city = data[data.Club == "Manchester City"].iloc[:2,:]

trace1 =go.Bar(
                    x = barcelona.Overall,
                    y = barcelona.Potential,
                    name = "FC Barcelona",
                    marker = dict(color = 'green'),
                    text= barcelona.Name)
trace2 =go.Bar(
                    x = juventus.Overall,
                    y = juventus.Potential,
                    name = "Juventus",
                    marker = dict(color = 'red'),
                    text= juventus.Name)
trace3 =go.Bar(
                    x = psg.Overall,
                    y = psg.Potential,
                    name = "Paris Saint-Germain",
                    marker = dict(color = 'blue'),
                    text= psg.Name)
data1 = [trace1, trace2, trace3]
layout1 = go.Layout(barmode="relative")
layout2 = go.Layout(barmode="group")
fig1=go.Figure(data = data1, layout = layout1)
fig2=go.Figure(data = data1, layout = layout2)
iplot(fig1)
iplot(fig2)
best_players = data.Nationality[data.Overall > 70]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(best_players))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
dataframe = data[data.Overall > 90]
trace1 = go.Scatter3d(
    x=dataframe.Overall,
    y=dataframe.Potential,
    z=dataframe.Age,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',     
    )
)

data2 = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data2, layout=layout)
iplot(fig)
