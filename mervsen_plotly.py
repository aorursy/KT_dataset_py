# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

from plotly import tools

from matplotlib import pyplot as plt

from plotly.offline import init_notebook_mode, iplot

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
soccer=pd.read_csv("/kaggle/input/soccer-players-statistics/FullData.csv")
soccer.head()
soccer.info()
#Top 20 Countries of Soccer Player Number

import plotly.express as px

x1=list(soccer.Nationality.unique())

y1=list(len(soccer[soccer["Nationality"]==each]) for each in x1)

df=pd.DataFrame()

df["Country"]=x1

df["NumberOfPlayers"]=y1

df_n=df.sort_values("NumberOfPlayers",ascending=False)

fig = px.bar(df_n.head(20), x="NumberOfPlayers", y="Country", orientation='h',title="Top 20 Countries with the Most Players")

fig.show()
from wordcloud import WordCloud
#Country of Origins of Top Rated 100 Players

df_r=soccer.sort_values(by="Rating",ascending=False).head(100)

plt.subplots(figsize=(8,8))

text=""

for i in df_r.Nationality:

    text= text +" " +i

wc=WordCloud(width=512,height=384).generate(text)

plt.imshow(wc) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
#Top Rated Players' Properties

df2=soccer.sort_values(by="Rating",ascending=False).head(20)

df2.info()

df2.Height= df2.Height.apply(lambda x: x.replace(" cm",""))
df2.Weight= df2.Weight.apply(lambda x: x.replace(" kg",""))
df2.Weight =df2.Weight.astype(int)
df2.Height = df2.Height.astype(int)
trace1=go.Scatter(

                    x = df2.Name,

                    y = df2.Rating,

                    mode = "markers",

                    name = "Ratings",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= "Ratings")

trace2=go.Scatter(

                    x = df2.Name,

                    y = df2.Skill_Moves ,

                    mode = "markers",

                    name = "Skill Moves",

                   marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= "Skill Moves")

trace3=go.Scatter(

                    x = df2.Name,

                    y = df2.Ball_Control ,

                    mode = "markers",

                    name = "Ball Control",

                   marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= "Ball Control")

trace5=go.Scatter(

                    x = df2.Name,

                    y = df2.Marking,

                    mode = "markers",

                    name = "Marking",

                   marker = dict(color = 'rgba(0, 0, 128, 0.8)'),

                    text= "Marking")

trace6=go.Scatter(

                    x = df2.Name,

                    y = df2.Acceleration,

                    mode = "markers",

                    name = "Acceleration",

                   marker = dict(color = 'rgba(255, 0, 50, 0.8)'),

                    text= "Acceleration")

data=[trace1,trace2,trace3,trace4,trace5,trace6]

layout = dict(title = 'Properties of Top Rated Players',

              xaxis= dict(title= 'Player Name',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Properties',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)