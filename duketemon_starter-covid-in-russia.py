import numpy as np

import pandas as pd



import plotly.express as px

import plotly.graph_objects as go
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_infected = pd.read_csv('/kaggle/input/covid19-in-russia/data-infected.csv')
df_infected.head()
fig = px.bar(df_infected, x="Субъект", y="03.05")

fig.show()
layout = {

  "title": "Number of infected people in Moscow", 

  "xaxis": {

    "type": "category", 

    "title": "Date", 

    "autorange": True

  }, 

  "yaxis": {

    "type": "linear", 

    "title": "Number of infected people", 

    "autorange": True

  }, 

  "autosize": True

}

x_values = list(df_infected.columns)[2:]

y_values = df_infected[df_infected['Субъект'] == 'Москва'].values[0].tolist()[2:]



fig = go.Figure([go.Bar(x=x_values, y=y_values)], layout=layout)

fig.show()