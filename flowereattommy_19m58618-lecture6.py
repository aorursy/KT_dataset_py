import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)





df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)

df = df[df["Country/Region"]=="New Zealand"]

df = df.groupby("ObservationDate").sum()

print(df)
df["daily_confirmed"] = df["Confirmed"].diff()

df["daily_deaths"] = df["Deaths"].diff()

df["daily_recovery"] = df["Recovered"].diff()

df["daily_confirmed"].plot()

df["daily_deaths"].plot()

df["daily_recovery"].plot()

plt.show
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df["daily_confirmed"].values,name="Daily confirmed")

daily_deaths_object = go.Scatter(x=df.index,y=df["daily_deaths"].values,name="Daily deaths")

daily_recovery_object = go.Scatter(x=df.index,y=df["daily_recovery"].values,name="Daily recovery")



layout_object = go.Layout(title="New Zealand daily cases 19M58618",xaxis=dict(title="Date"),yaxis=dict(title="Number of people"))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)

iplot(fig)

fig.write_html("New Zealand_daily_cases_19M58618.html")
df1 = df#[["daily_confirmed"]]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap="gist_ncar").highlight_max("daily_confirmed").set_caption("Daily Summaries")

display(styled_object)

f = open("table_19M58618.html","w")

f.write(styled_object.render())

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)

df = df[df["ObservationDate"]=="06/10/2020"]

df = df.groupby(["Country/Region"]).sum()

df1 = df.sort_values(by=["Confirmed"],ascending=False).reset_index()

print("Confirmed Ranking of New Zealand in 20200610: ",df1[df1["Country/Region"]=="New Zealand"].index.values[0]+1)






