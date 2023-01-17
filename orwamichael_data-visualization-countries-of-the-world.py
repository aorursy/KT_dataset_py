import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/countries/countries of the world.csv")

data.head()
data.isnull().sum()
data["Climate"] = data.groupby("Region")["Climate"].fillna(method='backfill')

data.isnull().sum()
data["Climate"] = data.groupby("Region")["Climate"].fillna(method='backfill')

data.isnull().sum()
data["Climate"] = data.groupby("Region")["Climate"].fillna(method='ffill')

data.isnull().sum()
data.fillna(0, inplace=True)

data.isnull().sum()
import plotly.express as px

px.bar(data, x="Region", y="Pop. Density (per sq. mi.)", hover_name="Country", color="GDP ($ per capita)", height=700)
px.sunburst(data, path=["Region", "Country"], values="Population", hover_name="Country")
px.treemap(data, path=["Region", "Country"], values="Population", hover_name="Country")
px.scatter(data, x="GDP ($ per capita)", size="Population", size_max=65, hover_name="Country", hover_data=data.columns, log_x=True, color="Region", height=700)