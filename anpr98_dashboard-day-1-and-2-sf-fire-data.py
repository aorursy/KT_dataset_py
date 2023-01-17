#importing all the necessary packages

import pandas as pd

from dateutil import parser

import matplotlib.pyplot as plt

import numpy as np



# import plotly

import plotly.plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
#read the data

data = pd.read_csv("../input/fire-department-calls-for-service.csv")
df = data.sample(10000)

print(df.head())



df = df[['Response DtTm', 'On Scene DtTm', 'Box', 'Number of Alarms']]
print(df.count())



df = df.dropna().reset_index()



print(df.count())
box = df.groupby('Box').size()



plt.scatter(box.index, box)

plt.xlabel("Box No.")

plt.ylabel("Count of incidents")

plt.title("Scatterplot showing counts of incidents in each box")

plt.show()
# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis

data = [go.Scatter(x=box.index, y=box, mode='markers')]



# specify the layout of our figure

layout = dict(title = "Scatterplot showing counts of incidents in each box",

              xaxis= dict(title= 'Box No.'), yaxis= dict(title= 'Count of incidents')  )



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
print(np.mean(box))



risk = box[box > 5]

print(len(risk))
df['Time Diff'] = 0



for i in range(len(df)):

    dt1 = parser.parse(df['On Scene DtTm'][i])

    dt2 = parser.parse(df['Response DtTm'][i])

    td = (dt1 - dt2).total_seconds()

    df['Time Diff'][i] = td/60

    
d_ = df[df["Number of Alarms"] == 1]



plt.scatter(d_["Box"],d_["Time Diff"])

plt.plot(risk, "ro")

plt.xlabel("Box No.")

plt.ylabel("Time taken to respond(in mins.)")

plt.title("Scatterplot showing time taken to respond(in mins.) vs box no.")

plt.show()
trace1 = go.Scatter(

   x = d_["Box"],

   y = d_["Time Diff"],

   mode = 'markers', 

   name = 'low risk boxes' 

   )



trace2 = go.Scatter(

   x = risk.index,

   y = risk,

   mode = 'markers',

   name = 'high risk boxes' 

   )



layout = dict(title = "Scatterplot showing time taken to respond(in mins.) vs box no.",

              xaxis= dict(title= 'Box No.'), yaxis= dict(title= 'Time taken to respond(in mins.)')  )



data = [trace1, trace2]

fig = dict(data = data, layout = layout)

iplot(fig)