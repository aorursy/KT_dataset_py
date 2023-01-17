import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# read in data
data = pd.read_csv("../input/openpowerlifting.csv")

## Hacky data munging
data['W10'] = ((data['BodyweightKg'][data['BodyweightKg'] > 0] / 10).apply(round) * 10).apply(int)
data['A5'] = ((data['Age'][data['Age'] > 0] / 5).apply(round) * 5).apply(int)

byW = data.groupby(['W10','Sex'])['BestDeadliftKg'].median().reset_index()
WM = byW[byW["Sex"] == "M"]
WF = byW[byW["Sex"] == "F"]
byA = data.groupby(['A5', 'Sex'])['BestDeadliftKg'].median().reset_index()
AM = byA[byA["Sex"] == "M"]
AF = byA[byA["Sex"] == "F"]

data['Age'] = ((data['Age'][data['Age'] > 0]).apply(round)).apply(int)
ageDist = data.groupby(['Age','Sex'])['BodyweightKg'].describe().reset_index()

DM = ageDist[ageDist["Sex"] == "M"]
DF = ageDist[ageDist["Sex"] == "F"]

DM = DM[["min", "mean", "max", "Age"]][DM["count"] > 3]
DM["Mean"] = DM["mean"]
DM["Min"] = DM["min"]
DM["Max"] = DM["max"]
DF = DF[["min", "mean", "max", "Age"]][DF["count"] > 3]
DF["Mean"] = DF["mean"]
DF["Min"] = DF["min"]
DF["Max"] = DF["max"]


# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis
data = [go.Scatter(x=WM.W10, y=WM.BestDeadliftKg, name='Male'),
        go.Scatter(x=WF.W10, y=WF.BestDeadliftKg, name='Female'),
       ]

# specify the layout of our figure
layout = dict(title = "Deadlift median by body weight",
              xaxis= dict(title= 'Weight',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)

data = [go.Scatter(x=AM.A5, y=AM.BestDeadliftKg, name='Male'),
        go.Scatter(x=AF.A5, y=AF.BestDeadliftKg, name='Female'),
       ]

# specify the layout of our figure
layout = dict(title = "Deadlift median by age",
              xaxis= dict(title= 'Age',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)

data = [go.Scatter(x=DM.Age, y=DM.Min, name='Min weight'),
        go.Scatter(x=DM.Age, y=DM.Mean, name='Median weight'),
        go.Scatter(x=DM.Age, y=DM.Max, name='Max weight'),
       ]

# specify the layout of our figure
layout = dict(title = "Male weight by age",
              xaxis= dict(title= 'Age',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)

data = [go.Scatter(x=DF.Age, y=DF.Min, name='Min weight'),
        go.Scatter(x=DF.Age, y=DF.Mean, name='Median weight'),
        go.Scatter(x=DF.Age, y=DF.Max, name='Max weight'),
       ]

# specify the layout of our figure
layout = dict(title = "Female weight by age",
              xaxis= dict(title= 'Age',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)