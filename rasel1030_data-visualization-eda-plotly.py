# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
# importing data visualization library

import matplotlib.pyplot as pt

import plotly.graph_objects as go

%matplotlib inline
# reading data

df = pd.read_csv("../input/bangladesh-covid19-daily-dataset/COVID-19 BD Dataset-4 april.csv")
# check about informatin of the data

df.info()
# checking the top 5 rows of the data

df.head()
# important statistical information

df.describe()
# showing number of unique value for every column

df.nunique()
# checking if every date has only one occurence 

df.Date.value_counts()
x=df[df["Daily new confirmed cases"] != 0]["Date"]

y = df[df["Daily new confirmed cases"] != 0]["Daily new confirmed cases"]

                    





fig = go.Figure()



fig.add_trace(go.Bar(x=x,

                    y = y,

                    text = y,

                    textposition = "auto"))



fig.update_layout(title = dict(text = "Daily confirmed cases",

                              font_size = 25

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))

x=df[df["Daily new deaths"] != 0]["Date"]

y = df[df["Daily new deaths"] != 0]["Daily new deaths"]





fig = go.Figure()



fig.add_trace(go.Bar(x=x,

                    y = y,

                    text = y,

                    textposition = "auto"))



fig.update_layout(title = dict(text = "Daily new deaths",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))

x=df[df["Daily new confirmed cases"] != 0]["Date"]

y1 = df[df["Daily new confirmed cases"] != 0]["Daily new confirmed cases"]

y2 = df[df["Daily new confirmed cases"] != 0]["Daily new deaths"]





fig = go.Figure()



fig.add_trace(go.Bar(x=x,

                    y = y1,

                    name = "confirmed cases"

                    ))



fig.add_trace(go.Bar(x=x,

                    y = y2,

                    name = "death cases"

                    ))



fig.update_layout(title = dict(text = "Daily confirmed vs death cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))
x=df["Date"]

y1 = df["Daily new confirmed cases"]

y2 = df["Daily new recovered"]





fig = go.Figure()



fig.add_trace(go.Bar(x=x,

                    y = y1,

                    name = "confirmed cases",

                    marker = dict(color="#b30000")

                    ))



fig.add_trace(go.Bar(x=x,

                    y = y2,

                    name = "recovered cases",

                    marker = dict(color="#158000")

                    ))



fig.update_layout(title = dict(text = "Daily confirmed vs recovered cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))
x=df["Date"]

y1 = df["Daily new recovered"]

y2 = df["Daily new deaths"]





fig = go.Figure()



fig.add_trace(go.Bar(x=x,

                    y = y1,

                    name = "recovered cases",

                    marker = dict(color="#158000")

                    ))



fig.add_trace(go.Bar(x=x,

                    y = y2,

                    name = "death cases",

                    marker = dict(color="#b30000")

                    ))



fig.update_layout(title = dict(text = "Daily recovered cases vs death  cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))
x=df["Date"]

y1 = df["Daily New Tests"]

y2 = df["Daily new confirmed cases"]





fig = go.Figure()



fig.add_trace(go.Bar(x=x,

                    y = y1,

                    name = "test cases",

                    marker = dict(color="#158000")

                    ))



fig.add_trace(go.Bar(x=x,

                    y = y2,

                    name = "confirmed cases",

                    marker = dict(color="#b30000")

                    ))



fig.update_layout(title = dict(text = "Daily test vs confirmed cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))
confirmed = df["Total confirmed cases"].max()

death = df["Total deaths"].max()

recoverd = df["Total recovered"].max()
x = ["Confirmed", "Recovered", "Death" ]

y = [confirmed, recoverd, death]



fig = go.Figure()



fig.add_trace(go.Bar(x= x,

                    y = y))



fig.update_layout(title = dict(text = "Comparision between confirmed, death and recovered")

                 )
x=df["Date"]

y = df["Daily new confirmed cases"]





fig = go.Figure()



fig.add_trace(go.Scatter(x=x,

                    y = y,

                    mode = "lines+markers",

                    marker = dict(size = 10)

                    ))



fig.update_layout(title = dict(text = "Daily new confirmed cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))



x=df["Date"]

y = df["Daily new deaths"]





fig = go.Figure()



fig.add_trace(go.Scatter(x=x,

                    y = y,

                    mode = "lines+markers",

                    marker = dict(size = 10)

                    ))



fig.update_layout(title = dict(text = "Daily new deaths",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))
x=df["Date"]

y = df["Daily new recovered"]





fig = go.Figure()



fig.add_trace(go.Scatter(x=x,

                    y = y,

                    mode = "lines+markers",

                    marker = dict(size = 10)

                    ))



fig.update_layout(title = dict(text = "Daily recovered cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))
x=df["Date"]

y = df["Active Cases"]





fig = go.Figure()



fig.add_trace(go.Scatter(x=x,

                    y = y,

                    mode = "lines+markers",

                    marker = dict(size = 10)

                    ))



fig.update_layout(title = dict(text = "Daily active cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))



x=df["Date"]

y = df["Daily New Tests"]





fig = go.Figure()



fig.add_trace(go.Scatter(x=x,

                    y = y,

                    mode = "lines+markers",

                    marker = dict(size = 10)

                    ))



fig.update_layout(title = dict(text = "Daily new tests",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))
x=df["Date"]



y2 = df["Daily new confirmed cases"]

y3 = df["Daily new deaths"]

y4 = df["Daily new recovered"]







fig = go.Figure()





fig.add_trace(go.Scatter(x=x,

                    y = y2,

                    mode = "lines+markers",

                    name = "Daily confirmed cases"

                    ))



fig.add_trace(go.Scatter(x=x,

                    y = y3,

                    mode = "lines+markers",

                    name = "Daily death cases"

                    ))



fig.add_trace(go.Scatter(x=x,

                    y = y4,

                    mode = "lines+markers",

                    name = "Daily recovered cases"

                    ))







fig.update_layout(title = dict(text = "Line chart of confirmed, death and recovered  cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))
x=df["Date"]

y1 = df["Daily new confirmed cases"]

y2 = df["Daily new deaths"]

y3 = df["Daily new recovered"]

y4 = df["Active Cases"]





fig = go.Figure()





fig.add_trace(go.Scatter(x=x,

                    y = y1,

                    mode = "lines+markers",

                    name = "Daily confirmed cases"

                    ))



fig.add_trace(go.Scatter(x=x,

                    y = y2,

                    mode = "lines+markers",

                    name = "Daily death cases"

                    ))



fig.add_trace(go.Scatter(x=x,

                    y = y3,

                    mode = "lines+markers",

                    name = "Daily recovered cases"

                    ))



fig.add_trace(go.Scatter(x=x,

                    y = y4,

                    mode = "lines+markers",

                    name = "Daily active cases"

                    ))





fig.update_layout(title = dict(text = "Line chart of confirmed, death, recovered and active cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))
x=df["Date"]

y1 = df["Daily New Tests"]

y2 = df["Daily new confirmed cases"]

y3 = df["Daily new deaths"]

y4 = df["Daily new recovered"]

y5 = df["Active Cases"]





fig = go.Figure()



fig.add_trace(go.Scatter(x=x,

                    y = y1,

                    mode = "lines+markers",

                    name = "Daily test cases"

                    ))



fig.add_trace(go.Scatter(x=x,

                    y = y2,

                    mode = "lines+markers",

                    name = "Daily confirmed cases"

                    ))



fig.add_trace(go.Scatter(x=x,

                    y = y3,

                    mode = "lines+markers",

                    name = "Daily death cases"

                    ))



fig.add_trace(go.Scatter(x=x,

                    y = y4,

                    mode = "lines+markers",

                    name = "Daily recovered cases"

                    ))



fig.add_trace(go.Scatter(x=x,

                    y = y5,

                    mode = "lines+markers",

                    name = "Daily active cases"

                    ))





fig.update_layout(title = dict(text = "Line chart of test, confirmed, death, recovered and active cases",

                              font_size = 20

                              ),

                 yaxis = dict(title = "Number of cases"),

                 xaxis = dict(title = "Date"))