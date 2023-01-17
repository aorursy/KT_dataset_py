# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# plotly

import plotly as py

from plotly.offline import init_notebook_mode, iplot

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
#Read our data here and I made arrangements in the columns.



data2015 = pd.read_csv("../input/2015.csv")

data2016 = pd.read_csv("../input/2016.csv")

data2017 = pd.read_csv("../input/2017.csv")

#columns name change

data2015.columns=[each.split()[0] if(len(each.split())>2) else each.replace(" ","_") for each in data2015.columns]

data2016.columns=[each.split()[0] if(len(each.split())>2) else each.replace(" ","_") for each in data2016.columns]

data2017.columns=[each.replace("."," ") for each in data2017.columns]

data2017.columns=[each.split()[0] if(len(each.split())>2) else each.replace(" ","_") for each in data2017.columns]
data2015.head(10)
data2015.info()
data2015.columns
data2015.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data2015.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# prepare data frame

df = data2015.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1=go.Scatter(x=data2015.Happiness_Rank,

                  y=data2015.Happiness_Score,

                  mode="lines",

                  name="citation",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=data2015.Country)

data = [trace1]

layout=dict(title="Happines Rank and Happiness Score of Top 100 Country",xaxis=dict(title="World Rank",ticklen=5,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
# first line plot

trace1=go.Scatter(x=df.Happiness_Rank,

                  y=df.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df.Country)

# second line plot

trace2 = go.Scatter(

        x=df.Happiness_Rank,

        y=df.Economy,

        xaxis="x2",

        yaxis="y2",

        name="Economy",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Economy and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.

df1 = data2015.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df1["Country"],y=df1["Economy"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Economy")

plt.title("Economy Given States")
# first line plot

trace1=go.Scatter(x=df.Happiness_Rank,

                  y=df.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df.Country)

# second line plot

trace2 = go.Scatter(

        x=df.Happiness_Rank,

        y=df.Family,

        xaxis="x2",

        yaxis="y2",

        name="Family",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Family and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.

df2 = data2015.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Family"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Family")

plt.title("Family Given States")
# first line plot

trace1=go.Scatter(x=df.Happiness_Rank,

                  y=df.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df.Country)

# second line plot

trace2 = go.Scatter(

        x=df.Happiness_Rank,

        y=df.Health,

        xaxis="x2",

        yaxis="y2",

        name="Health",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Health and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
#We used the first 50 countries because 100 countries did not fit.

df2 = data2015.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Health"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Health")

plt.title("Health Given States")
# first line plot

trace1=go.Scatter(x=df.Happiness_Rank,

                  y=df.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df.Country)

# second line plot

trace2 = go.Scatter(

        x=df.Happiness_Rank,

        y=df.Freedom,

        xaxis="x2",

        yaxis="y2",

        name="Freedom",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Freedom and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
#We used the first 50 countries because 100 countries did not fit.

df2 = data2015.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Freedom"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Freedom")

plt.title("Freedom Given States")
# first line plot

trace1=go.Scatter(x=df.Happiness_Rank,

                  y=df.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df.Country)

# second line plot

trace2 = go.Scatter(

        x=df.Happiness_Rank,

        y=df.Trust,

        xaxis="x2",

        yaxis="y2",

        name="Trust",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Trust and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
#We used the first 50 countries because 100 countries did not fit.

df2 = data2015.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Trust"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Trust")

plt.title("Trust Given States")
# first line plot

trace1=go.Scatter(x=df.Happiness_Rank,

                  y=df.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df.Country)

# second line plot

trace2 = go.Scatter(

        x=df.Happiness_Rank,

        y=df.Generosity,

        xaxis="x2",

        yaxis="y2",

        name="Generosity",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Generosity and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.

df2 = data2015.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Generosity"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Generosity")

plt.title("Generosity Given States")
data2016.head(10)
data2016.tail()
data2016.info()
data2016.columns
data2016.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data2016.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# prepare data frame

df2016 = data2016.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1=go.Scatter(x=data2016.Happiness_Rank,

                  y=data2016.Happiness_Score,

                  mode="lines",

                  name="citation",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=data2016.Country)

data = [trace1]

layout=dict(title="Happines Rank and Happiness Score of Top 100 Country",xaxis=dict(title="World Rank",ticklen=5,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
# first line plot

trace1=go.Scatter(x=df2016.Happiness_Rank,

                  y=df2016.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2016.Country)

# second line plot

trace2 = go.Scatter(

        x=df2016.Happiness_Rank,

        y=df2016.Economy,

        xaxis="x2",

        yaxis="y2",

        name="Economy",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2016.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Economy and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.

df2 = data2016.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Economy"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Economy")

plt.title("Economy Given States")
# first line plot

trace1=go.Scatter(x=df2016.Happiness_Rank,

                  y=df2016.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2016.Country)

# second line plot

trace2 = go.Scatter(

        x=df2016.Happiness_Rank,

        y=df2016.Family,

        xaxis="x2",

        yaxis="y2",

        name="Family",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2016.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Family and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.

df2 = data2016.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Family"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Family")

plt.title("Poverty Rate Given States")
# first line plot

trace1=go.Scatter(x=df2016.Happiness_Rank,

                  y=df2016.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2016.Country)

# second line plot

trace2 = go.Scatter(

        x=df2016.Happiness_Rank,

        y=df2016.Health,

        xaxis="x2",

        yaxis="y2",

        name="Health",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2016.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Health and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.

df2 = data2016.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Health"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Health")

plt.title("Health Rate Given States")
# first line plot

trace1=go.Scatter(x=df2016.Happiness_Rank,

                  y=df2016.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2016.Country)

# second line plot

trace2 = go.Scatter(

        x=df2016.Happiness_Rank,

        y=df2016.Freedom,

        xaxis="x2",

        yaxis="y2",

        name="Freedom",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2016.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Freedom and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
#We used the first 50 countries because 100 countries did not fit.

df2 = data2016.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Freedom"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Freedom")

plt.title("Freedom Given States")
# first line plot

trace1=go.Scatter(x=df2016.Happiness_Rank,

                  y=df2016.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2016.Country)

# second line plot

trace2 = go.Scatter(

        x=df2016.Happiness_Rank,

        y=df2016.Trust,

        xaxis="x2",

        yaxis="y2",

        name="Trust",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2016.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Trust and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
#We used the first 50 countries because 100 countries did not fit.



df2 = data2016.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Trust"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Trust")

plt.title("Trust Given States")
# first line plot

trace1=go.Scatter(x=df2016.Happiness_Rank,

                  y=df2016.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2016.Country)

# second line plot

trace2 = go.Scatter(

        x=df2016.Happiness_Rank,

        y=df2016.Generosity,

        xaxis="x2",

        yaxis="y2",

        name="Generosity",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2016.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Generosity and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.



df2 = data2016.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Generosity"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Generosity")

plt.title("Generosity Given States")
data2017.head(10)
data2017.tail()
data2017.info()
data2017.columns
data2017.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data2017.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# prepare data frame

df2017 = data2017.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1=go.Scatter(x=data2017.Happiness_Rank,

                  y=data2017.Happiness_Score,

                  mode="lines",

                  name="citation",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=data2017.Country)

data = [trace1]

layout=dict(title="Happines Rank and Happiness Score of Top 100 Country",xaxis=dict(title="World Rank",ticklen=5,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
# first line plot

trace1=go.Scatter(x=df2017.Happiness_Rank,

                  y=df2017.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2017.Country)

# second line plot

trace2 = go.Scatter(

        x=df2017.Happiness_Rank,

        y=df2017.Economy,

        xaxis="x2",

        yaxis="y2",

        name="Economy",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2017.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Economy and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.



df1 = data2017.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df1["Country"],y=df1["Economy"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Economy")

plt.title("Economy Given States")
# first line plot

trace1=go.Scatter(x=df2017.Happiness_Rank,

                  y=df2017.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2017.Country)

# second line plot

trace2 = go.Scatter(

        x=df2017.Happiness_Rank,

        y=df2017.Family,

        xaxis="x2",

        yaxis="y2",

        name="Family",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2017.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Family and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.



df2 = data2017.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Family"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Family")

plt.title("Family Given States")
# first line plot

trace1=go.Scatter(x=df2017.Happiness_Rank,

                  y=df2017.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2017.Country)

# second line plot

trace2 = go.Scatter(

        x=df2017.Happiness_Rank,

        y=df2017.Health,

        xaxis="x2",

        yaxis="y2",

        name="Health",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2017.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Health and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
#We used the first 50 countries because 100 countries did not fit.



df2 = data2017.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Health"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Health")

plt.title("Health Given States")
# first line plot

trace1=go.Scatter(x=df2017.Happiness_Rank,

                  y=df2017.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2017.Country)

# second line plot

trace2 = go.Scatter(

        x=df2017.Happiness_Rank,

        y=df2017.Freedom,

        xaxis="x2",

        yaxis="y2",

        name="Freedom",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2017.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Freedom and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
#We used the first 50 countries because 100 countries did not fit.



df2 = data2017.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Freedom"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Freedom")

plt.title("Freedom Given States")
# first line plot

trace1=go.Scatter(x=df2017.Happiness_Rank,

                  y=df2017.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2017.Country)

# second line plot

trace2 = go.Scatter(

        x=df2017.Happiness_Rank,

        y=df2017.Trust,

        xaxis="x2",

        yaxis="y2",

        name="Trust",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2017.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Trust and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
#We used the first 50 countries because 100 countries did not fit.



df2 = data2017.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Trust"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Trust")

plt.title("Trust Given States")
# first line plot

trace1=go.Scatter(x=df2017.Happiness_Rank,

                  y=df2017.Happiness_Score,

                  mode="lines",

                  name="HappinessScore",

                  marker=dict(color="rgba(16,122,2,0.8)"),

                  text=df2017.Country)

# second line plot

trace2 = go.Scatter(

        x=df2017.Happiness_Rank,

        y=df2017.Generosity,

        xaxis="x2",

        yaxis="y2",

        name="Generosity",

        marker=dict(color ="rgba(160,112,20,0.8)"),

        text=df2017.Country

)



data=[trace1,trace2]

layout = go.Layout(

        xaxis2= dict(

                    domain=[0.6,0.95],

                    anchor="y2",

        ),

        yaxis2=dict(

                domain=[0.6,0.95],

                anchor="x2",

        ),

        title = 'Generosity and Happiness Score vs Happiness Rank  of Top 100 Country'

)



fig= go.Figure(data=data,layout=layout)

iplot(fig)
# We used the first 50 countries because 100 countries did not fit.



df2 = data2017.iloc[:50,:]



plt.figure(figsize=(15,10))

sns.barplot(x=df2["Country"],y=df2["Generosity"])

plt.xticks(rotation=90)

plt.xlabel("States")

plt.ylabel("Generosity")

plt.title("Generosity Given States")