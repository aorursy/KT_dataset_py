# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/zomato.csv")

df.shape
df.head()
df.tail()
df.info()
semiclean = ["url", "address", "phone", "location", "dish_liked"]

df_semiclean = df.drop(semiclean, axis = 1)

df_semiclean.info()
df_clean = df_semiclean.dropna()

df_clean.info()
rates = list()

for fila in df_clean.rate:

    aux = fila[:3]

    rates.append(aux)

#df_clean = df_clean.drop(["rate"], axis = 1)

df_clean = df_clean.assign(rate = rates)
df_clean.head(20)
# Distribution of rates and prices



import plotly.plotly as py

import plotly.graph_objs as go

import plotly.offline as ply

ply.init_notebook_mode(connected=True)





rate = go.Histogram(x=df_clean["rate"],

                   opacity=0.75,

                   name = "Rate Values",

                   )



price = go.Histogram(x=df_clean["approx_cost(for two people)"],

                   opacity=0.75,

                   name = "Price Values",

                   )



layout = go.Layout(barmode='overlay',

                   title='Histogram',

                   xaxis=dict(title='Values'),

                   yaxis=dict(

                        title='yaxis title',

                    ),

                  )



data = [rate]

fig = go.Figure(data=data, layout=layout)                   

ply.iplot(fig)



data = [price]

fig = go.Figure(data=data, layout=layout)                   

ply.iplot(fig)
trace = [go.Scattergl(

                    x = df_clean["rate"],

                    y = df_clean["approx_cost(for two people)"],

                    mode = "markers"

                    )]

ply.iplot(trace)
cuisines_ = np.unique(df_clean.cuisines)

counter = list()

for i in cuisines_:

    aux = len(df_clean.index[df_clean["cuisines"]==i])

    counter.append(aux)
df_count = pd.DataFrame({"Q": counter}, index = cuisines_)

df_count.head()
cuisin = [go.Histogram(x=df_count["Q"],

                   opacity=0.75,

                   name = "Cusine counter",

                   text = df_count.index,

                   )]



fig = go.Figure(data=cuisin)                   

ply.iplot(fig)
cities_ = np.unique(df_clean["listed_in(city)"])

counter = list()

for i in cities_:

    aux = len(df_clean.index[df_clean["listed_in(city)"]==i])

    counter.append(aux)



df_city = pd.DataFrame({"Q": counter}, index = cities_)

df_city.head()
trace = [go.Pie(labels=df_city.index, values=df_city["Q"])]



ply.iplot(trace)