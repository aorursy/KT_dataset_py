import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from plotly import tools

import plotly.offline as py

import plotly.graph_objs as go



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
py.init_notebook_mode(connected=True)
df = pd.read_csv("../input/Iris.csv")
df.head(10)
df.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
set(df.Species)
fig = tools.make_subplots(rows=1, cols=2)



for spec in set(df.Species):

    subp = go.Scatter(x=df[df.Species == spec]['SepalLengthCm'],

                      y=df[df.Species == spec]['SepalWidthCm'],

                      mode='markers',

                      name=spec)

    fig.append_trace(subp, 1, 1)



for spec in set(df.Species):

    subp = go.Scatter(x=df[df.Species == spec]['PetalLengthCm'],

                      y=df[df.Species == spec]['PetalWidthCm'],

                      mode='markers',

                      name=spec)

    fig.append_trace(subp, 1, 2)



py.iplot(fig)