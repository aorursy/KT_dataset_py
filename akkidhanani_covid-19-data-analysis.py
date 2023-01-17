import os

import matplotlib.pyplot as plt

from matplotlib import style

style.use("fivethirtyeight")

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
coviddataset=pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
coviddataset.head(5)
coviddataset.isnull().sum()
coviddataset["day"]=coviddataset["Date"].str.split('/').str[1].astype(int)

coviddataset["month"]=coviddataset["Date"].str.split('/').str[0].astype(int)

coviddataset["year"]=coviddataset["Date"].str.split('/').str[2].astype(int)
coviddataset.head()
plt.figure(figsize=(10,6))

coviddataset.groupby("month").mean()["Confirmed"].plot()

plt.xlabel("month")

plt.ylabel("cases conformed")

plt.title("No of positive cases conformed")
plt.figure(figsize=(10,6))

coviddataset.groupby("month").mean()["Deaths"].plot()

plt.xlabel("month")

plt.ylabel("Deaths conformed")

plt.title("No of Deaths")
plt.figure(figsize=(10,6))

coviddataset.groupby("month").mean()["Recovered"].plot()

plt.xlabel("month")

plt.ylabel("Recovered cases")

plt.title("Recovered")
plt.figure(figsize=(10,6))

coviddataset.groupby("month")["Confirmed"].plot()

plt.title("no of cases conformed")
plt.figure(figsize=(10,6))

coviddataset.groupby("month")["Deaths"].plot()

plt.title("no of deaths conformed")
plt.figure(figsize=(10,6))

coviddataset.groupby("month")["Recovered"].plot()

plt.title("no of Recovered")
ax=plt.figure(figsize=(19,12.5))

ax.add_subplot(121)

sns.lineplot(x="month",y="Confirmed",data=coviddataset,color="r")

ax.add_subplot(122)

sns.lineplot(x="month",y="Deaths",data=coviddataset,color="black")
fig=plt.figure(figsize=(19,10))

ax=fig.add_subplot(121,projection="3d")

ax.scatter(coviddataset["Confirmed"],coviddataset["Recovered"],coviddataset["Deaths"],color="r")

ax.set(xlabel='\nConfirmed',ylabel='\nRecovered',zlabel='\nDeaths')
from plotly.subplots import make_subplots
dataset=coviddataset[["Recovered","Deaths","Confirmed"]]
#make figure with subplots

fig=make_subplots(rows=1,cols=1,specs=[[{"type":"surface"}]])

#adding surface

fig.add_surface(z=dataset)

fig.update_layout(

                   showlegend=False,

                   height=800,

                   width=800,

                   title_text="3D model"

                 )

fig.show()
import plotly.express as px
df=coviddataset

fig=px.scatter_3d(df,x="Confirmed",y="Recovered",z="Deaths")

fig.show()
plt.figure(figsize=(10,6))

sns.scatterplot(x="Confirmed",y="Long",data=coviddataset,color="red")
plt.figure(figsize=(10,6))

sns.scatterplot(x="Confirmed",y="Lat",data=coviddataset,color="green")
plt.figure(figsize=(10,6))

sns.scatterplot(x="Long",y="Lat",hue="Confirmed",data=coviddataset)