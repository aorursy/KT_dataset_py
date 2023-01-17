import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization tools

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import seaborn as sns

import pandas_profiling as pp

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df2015 = pd.read_csv("../input/world-happiness/2015.csv")

df2016 = pd.read_csv("../input/world-happiness/2016.csv")

df2017 = pd.read_csv("../input/world-happiness/2017.csv")
df2017.sample(5)
df2017.info()
df2017.isnull().sum()
report = pp.ProfileReport(df2017)



report.to_file("report.html")



report
import missingno as msno

msno.matrix(df2017)

plt.show()
df2017.columns
df2017.columns=[each.replace(".","") for each in df2017.columns]

df2016.columns=[each.replace(" ","") for each in df2016.columns]

df2015.columns=[each.replace(" ","") for each in df2015.columns]
df2017.columns
f,ax = plt.subplots(figsize=(25, 15))

sns.heatmap(df2017.corr(), annot=True, linewidths=0.5,linecolor="blue", fmt= '.3f',ax=ax,cmap= 'YlGnBu')

plt.show()
plt.figure(figsize=(25,15))

sns.barplot(x=df2017.Country, y=df2017.HappinessScore)

plt.xticks(rotation= 90)

plt.xlabel('Country')

plt.ylabel('Happiness Score')

plt.show()
fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=df2017.HappinessRank, y=df2017.EconomyGDPperCapita,

                    mode='lines+markers',

                    name='2017'))

fig.add_trace(go.Scatter(x=df2016.HappinessRank, y=df2016["Economy(GDPperCapita)"],

                    mode='lines+markers',

                    name='2016'))

fig.add_trace(go.Scatter(x=df2015.HappinessRank, y=df2015["Economy(GDPperCapita)"],

                    mode='lines+markers',

                    name='2015'))

fig.show()
categories = ['HappinessScore','EconomyGDPperCapita','Generosity','Freedom', 'Family']

r1=[df2017[each][df2017["Country"]=="Turkey"].mean()/df2017[each].max()  for each in categories]

r2=[df2017[each][df2017["Country"]=="Norway"].mean()/df2017[each].max()  for each in categories]



fig = go.Figure()



fig.add_trace(go.Scatterpolar(

      r=r1,

      theta=categories,

      fill='toself',

      name='Turkey'

))

fig.add_trace(go.Scatterpolar(

      r=r2,

      theta=categories,

      fill='toself',

      name='Norway'

))



fig.update_layout(

  polar=dict(

    radialaxis=dict(

      visible=True,

      range=[0, 1]

    )),

  showlegend=False

)



fig.show()
fig = go.Figure()

# Use x instead of y argument for horizontal plot

fig.add_trace(go.Box(x=df2017.HappinessScore, name='2017'))

fig.add_trace(go.Box(x=df2016.HappinessScore, name='2016'))

fig.add_trace(go.Box(x=df2015.HappinessScore, name='2015'))



fig.show()
# trace1

trace1 = go.Scatter3d(

    x=df2017.EconomyGDPperCapita,

    y=df2017.HealthLifeExpectancy,

    z=df2017.Family,

    mode='markers',

    name = "2017",

    marker=dict(

        color='rgb(217, 100, 100)',

        size=12,

        line=dict(

            color='rgb(255, 255, 255)',

            width=0.1

        )

    )

)

# trace2

trace2 = go.Scatter3d(

    x=df2016["Economy(GDPperCapita)"],

    y=df2016["Health(LifeExpectancy)"],

    z=df2016.Family,

    mode='markers',

    name = "2016",

    marker=dict(

        color='rgb(54, 170, 127)',

        size=12,

        line=dict(

            color='rgb(204, 204, 204)',

            width=0.1

        )

    )

)

# trace3

trace3 = go.Scatter3d(

    x=df2015["Economy(GDPperCapita)"],

    y=df2015["Health(LifeExpectancy)"],

    z=df2015.Family,

    mode='markers',

    name = "2015",

    marker=dict(

        color='rgb(150, 170, 25)',

        size=12,

        line=dict(

            color='rgb(204, 204, 204)',

            width=0.1

        )

    )

)

data = [trace1, trace2, trace3]

layout = go.Layout(

    title = '2015 , 2016 and 2017 values',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)