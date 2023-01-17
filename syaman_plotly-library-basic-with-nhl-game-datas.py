
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud

import os
print(os.listdir("../input"))

#our datas that we can use in this kernel
#game=pd.read_csv("../input/nhl-game-data/game.csv")
#goalie_stats=pd.read_csv("../input/nhl-game-data/game_goalie_stats.csv")
plays=pd.read_csv("../input/nhl-game-data/game_plays.csv")
plays_players=pd.read_csv("../input/nhl-game-data/game_plays_players.csv")
#shifts=pd.read_csv("../input/game_shifts.csv")
skater_stats=pd.read_csv("../input/nhl-game-data/game_skater_stats.csv")
teams_stats=pd.read_csv("../input/nhl-game-data/game_teams_stats.csv")
player_info=pd.read_csv("../input/nhl-game-data/player_info.csv")
team_info=pd.read_csv("../input/nhl-game-data/team_info.csv")
index=pd.read_csv("../input/ckdisease/kidney_disease.csv")
#Line plot
df=teams_stats.iloc[:30,:] # in team_info, we will use first 30 raw and all columns 

trace1=go.Scatter(x=df.shots,y=df.hits,mode="lines",name="hits",marker=dict(color="rgba(16,112,2,0.8)"),text=df.head_coach)
trace2=go.Scatter(x=df.shots,y=df.faceOffWinPercentage,mode="lines+markers",name="faceOffWinPercentage",marker=dict(color="rgba(80,26,80,0.8)"),text=df.head_coach)

data=[trace1,trace2]
layout=dict(title="hits and faceOffWinPercentage vs shots of top 30 teams_stats",xaxis=dict(title="shot",ticklen=5,zeroline=False))
fig=dict(data = data,layout=layout)
iplot(fig)
#Creating traces
#x = x axis
#y = y axiss
#mode = type of plot like marker, line or line + markers
#name = name of the plots
#marker = marker is used with dictionary.
#color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#text = The hover text (hover is curser)
#data = is a list that we add traces into it
#layout = it is dictionary.
#title = title of layout
#x axis = it is dictionary
#title = label of x axis
#ticklen = length of x axis ticks
#zeroline = showing zero line or not
#fig = it includes data and layout
#iplot() = plots the figure(fig) that is created by data and layout


#scatter plot
df2goals=teams_stats[teams_stats.goals==2].iloc[:30,:]
df3goals=teams_stats[teams_stats.goals==3].iloc[:30,:]
df4goals=teams_stats[teams_stats.goals==4].iloc[:30,:]

trace1=go.Scatter(x=df2goals.shots,y=df2goals.hits,
                  mode="markers",name="2goals",
                  marker=dict(color="rgba(16,112,2,0.8)"),text=df2goals.head_coach)

trace2=go.Scatter(x=df3goals.shots,y=df3goals.hits,
                  mode="markers",name="3goals",
                  marker=dict(color="rgba(160,112,40,0.8)"),text=df3goals.head_coach)

trace3=go.Scatter(x=df4goals.shots,y=df4goals.hits,
                  mode="markers",name="4goals",
                  marker=dict(color="rgba(60,200,80,0.8)"),text=df4goals.head_coach)

data=[trace1,trace2,trace3]
layout=dict(title="hits and faceOffWinPercentage vs shots of top 30 teams_stats for 2,3,4 goals",
           xaxis=dict(title="shot",ticklen=5,zeroline=False),
           yaxis=dict(title="hits",ticklen=5,zeroline=False))
fig=dict(data = data,layout=layout)
iplot(fig)
#bar plot
#we will compare shots and hitting rate with 2 goals
df2goals=teams_stats[teams_stats.goals==2].iloc[:5,:]

trace1=go.Bar(
    x=df2goals.head_coach,
    y=df2goals.hits,
    name="hits",
    marker=dict(color="rgba(255,112,255,0.8)",
              line=dict(color="rgba(0,0,0)",width=1.5)),
                        text=df2goals.team_id)

trace2=go.Bar(
    x=df2goals.head_coach,
    y=df2goals.shots,
    name="shots",
    marker=dict(color="rgba(255,255,122,0.8)",
              line=dict(color="rgba(0,0,0)",width=1.5)),
                        text=df2goals.team_id)

data=[trace1,trace2]
layout=go.Layout(barmode="group")
fig=go.Figure(data=data,layout=layout)
iplot(fig)

# we will make different type bar plot with above datas of fig
df2goals=teams_stats[teams_stats.goals==2].iloc[:5,:]
x=df2goals.head_coach

trace1={
    "x":x,
    "y":df2goals.hits,
    "name":"hits",
    "type":"bar"
};

trace2={
    "x":x,
    "y":df2goals.shots,
    "name":"shots",
    "type":"bar"
};
data=[trace1,trace2]

layout={
    "xaxis":{"title":"head_coach"},
    "barmode":"relative",
    "title":"hits and faceOffWinPercentage vs shots of top 30 teams_stats for 2 goals"
};
fig=go.Figure(data=data,layout=layout)
iplot(fig)
#pie plot
df4goals=teams_stats[teams_stats.goals==4].iloc[:30,:]

WinPercentage_list= [float(face) for face in df4goals.faceOffWinPercentage]
labels=df4goals.head_coach

fig={
    "data":[
        {
            "values":WinPercentage_list,
            "labels":labels,
            "domain":{"x":[0,0.5]},
            "hoverinfo":"label+percent+name",
            "hole":0.3,
            "type":"pie"
        },],
    "layout":{
        "title":"Win Percentage of teams",
        "annotations": [
            {
                "font":{"size":20},
                "showarrow":False,
                "text":"Win Percentage %",
                "x":0.2,
                "y":1
            },
        ]
    }
}
iplot(fig)
#Buble chart
df4goals=teams_stats[teams_stats.goals==4].iloc[:30,:]
winpercentagesize=[float(face) for face in df4goals.faceOffWinPercentage]
pim_color=df4goals.pim

data=[
    {
        "y":df4goals.faceOffWinPercentage,
        "x":df4goals.shots,
        "mode":"markers",
        "marker":{
            "color":pim_color,
            "size":winpercentagesize,
            "showscale":True
        },
    "text":df4goals.head_coach
    }
]
iplot(data)
#histogram ploting
#we will look shots that are results for 2 goals and 3 goals
goals2 = teams_stats.shots[teams_stats.goals==2]
goals3=teams_stats.shots[teams_stats.goals==3]

trace1=go.Histogram(
    x=goals2,
    opacity=0.7,
    name="2goals",
    marker=dict(color="rgba(180,200,25,0.6)")) #color of hist

trace2=go.Histogram(
    x=goals3,
    opacity=0.7,
    name="3goals",
    marker=dict(color="rgba(10,200,250,0.6)"))

data=[trace1,trace2]
layout=go.Layout(barmode="overlay",
                title="hist poting for team",
                xaxis=dict(title="pim"),
                yaxis=dict(title="count")
                )
fig=go.Figure(data=data,layout=layout)
iplot(fig)
#to plot wordcloud first I add two data(data1 and data2) in one cvs file(new data)
data1=team_info.shortName
data2=teams_stats
newdata=pd.concat([data1.head(30),data2.head(30)],axis=1,ignore_index =False)

#word cloud
goals2=newdata.shortName[newdata.goals==2]
plt.subplots(figsize=(8,8))
wordcloud= WordCloud(
    background_color="white",
    width=512,
    height=314,
    ).generate(" ".join(goals2))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("graph.png")
plt.show()
#preparing data to add plays.event datas in newdata
data1=plays.event
data2=newdata
newdata1=pd.concat([data1.head(30),data2.head(30)],axis=1,ignore_index =False)

#box plotting
goals4=newdata1[newdata1.goals==4]

trace0=go.Box(
    y=goals4.shots,
    name="shots for making 4 goals",
    marker=dict(color="rgb(255,0,255)",)
)

trace1=go.Box(
    y=goals4.hits,
    name="hits for making 4 goals",
    marker=dict(color="rgb(0,255,255)",)
)

data=[trace0,trace1]
iplot(data)
#for scatter plot we didn't have index number in our data. 
#to add index number we will make some preperas
data1=index.id
data2=newdata1
newdata2=pd.concat([data1.head(30),data2.head(30)],axis=1,ignore_index =False)
#scatter plot Matrix
#with scatter plot matrix we can see both box plot and scatter plot 
import plotly.figure_factory as ff

goals3=newdata2[newdata2.goals==3]
datagoals=goals3.loc[:,["faceOffWinPercentage","goals","hits","shots"]]
datagoals["id"]=np.arange(1,len(datagoals)+1)
fig=ff.create_scatterplotmatrix(datagoals,diag="box",index="id",
                                colormap="Portland",colormap_type="cat",
                               height=700,width=750)
iplot(fig)
#first plotting datas
trace1=go.Scatter(
x=newdata2.powerPlayOpportunities,
y=newdata2.shots,
name="shots",
    marker=dict(color="rgba(255,255,0,0.7)"),
)
#second plotting datas
trace2=go.Scatter(
x=newdata2.powerPlayOpportunities,
y=newdata2.faceOffWinPercentage,
xaxis="x2",
yaxis="y2",
name="win percentage",
    marker=dict(color="rgba(255,0,255,0.7)"),
)
data=[trace1,trace2]
layout=go.Layout(
xaxis2=dict(domain=[0.6,0.95],anchor="y2"),
yaxis2=dict(domain=[0.6,0.95],anchor="x2"),
    title="shots and win pertentage"
)
fig=go.Figure(data=data,layout=layout)
iplot(fig)

trace1=go.Scatter3d(
x=newdata2.powerPlayOpportunities,
y=newdata2.shots,
z=newdata2.hits,
    mode="markers",
    marker=dict(size=10,
               color=newdata2.pim,
               colorscale="Jet")
)
data=[trace1]
layout=go.Layout(margin=dict(l=0,#left
                             r=0,#right
                             b=0,#buttom
                             t=0#top
                            ))
fig=go.Figure(data=data,layout=layout)
iplot(fig)
newdata2.head()
trace1=go.Scatter(x=newdata2.giveaways,
y=newdata2.shots,name="shots")

trace2=go.Scatter(x=newdata2.giveaways,
y=newdata2.faceOffWinPercentage,
xaxis="x2",yaxis="y2",name="win percentage")

trace3=go.Scatter(x=newdata2.giveaways,
y=newdata2.goals,
xaxis="x3",yaxis="y3",name="goals")

trace4=go.Scatter(x=newdata2.giveaways,
y=newdata2.hits,
xaxis="x4",yaxis="y4",name="hits")

data=[trace1,trace2,trace3,trace4]

layout=go.Layout(xaxis=dict(domain=[0,0.45]),yaxis=dict(domain=[0,0.45]),
    xaxis2=dict(domain=[0.55,1],anchor="y2"),yaxis2=dict(domain=[0,0.45],anchor="x2"),
    xaxis3=dict(domain=[0,0.45],anchor="y3"),yaxis3=dict(domain=[0.55,1],anchor="x3"),
     xaxis4=dict(domain=[0.55,1],anchor="y4"),yaxis4=dict(domain=[0.55,1],anchor="x4"),
                 title="Subplot"
                )
fig=go.Figure(data=data,layout=layout)
iplot(fig)