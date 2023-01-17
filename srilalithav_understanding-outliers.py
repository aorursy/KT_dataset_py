import numpy as np

import pandas as pd

np.random.seed(42)

import requests

import io,os

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import seaborn as sns

from sklearn import preprocessing 

from sklearn.cluster import KMeans

import scipy.cluster.hierarchy as shc

from sklearn.cluster import DBSCAN

from scipy.spatial.distance import cdist

from scipy.cluster.hierarchy import dendrogram, cophenet,linkage
data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv")

deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv")

recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv")
confirmed.head(3)
data.head(3)
data=data.dropna()

data.shape
data.groupby(by='Country').groups.keys()

total=data.groupby(['Country'])['Deaths'].count().reset_index()
sns.barplot(x='Country',y='Deaths',data=total)
top5_confirmed =  data.groupby(['Country']).sum().nlargest(5,['Confirmed'])['Confirmed']

print("Top 5 Countries were affected most")

top5_confirmed=top5_confirmed.reset_index()

print(top5_confirmed)
sns.catplot(x="Country", y="Confirmed",kind="swarm",data=top5_confirmed);
top5_deaths =  data.groupby(['Country']).sum().nlargest(5,['Deaths'])['Deaths']

print("Top 5 Countries were affected most")

top5_deaths=top5_deaths.reset_index()

print(top5_deaths)

sns.catplot(x="Country", y="Deaths",kind="swarm",data=top5_deaths);
top5_recovered =  data.groupby(['Country']).sum().nlargest(5,['Recovered'])['Recovered']

print("Top 5 Countries were affected most")

top5_recovered=top5_recovered.reset_index()

print(top5_recovered)

sns.catplot(x="Country", y="Recovered",kind="swarm",data=top5_recovered);
sns.pairplot(data)
by_dates=data.groupby(by=['Date'],axis=0).sum().reset_index()

print(by_dates.head(3))
plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

plt.title('Province/State has reported confirmed cases by date')

sns.barplot(data=by_dates, x='Date', y='Confirmed',palette="muted",log=True)
plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

plt.title('Province/State has reported Deaths case by dates')

sns.barplot(data=by_dates, x='Date', y='Deaths',log=True,palette="muted")
plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

plt.title('Province/State has reported Deaths case by dates')

sns.barplot(data=by_dates, x='Date', y='Deaths',log=True,palette="muted")