import os

import math

import numpy as np

import pandas as pd

import seaborn as sns

import calendar



%matplotlib inline

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from bokeh.charts import TimeSeries, show
for f in os.listdir('../input'):

    size_bytes = round(os.path.getsize('../input/' + f)/ 1000, 2)

    size_name = ["KB", "MB"]

    i = int(math.floor(math.log(size_bytes, 1024)))

    p = math.pow(1024, i)

    s = round(size_bytes / p, 2)

    print(f.ljust(25) + str(s).ljust(7) + size_name[i])
train_df = pd.read_csv("../input/train_1.csv")

key_df = pd.read_csv("../input/key_1.csv")
print("Train".ljust(15), train_df.shape)

print("Key".ljust(15), key_df.shape)
print(train_df[:4].append(train_df[-4:], ignore_index=True))
print(key_df[:4].append(key_df[-4:], ignore_index=True))
page_details = pd.DataFrame([i.split("_")[-3:] for i in train_df["Page"]])

page_details.columns = ["project", "access", "agent"]

page_details.describe()
project_columns = page_details['project'].unique()

access_columns = page_details['access'].unique()

agents_columns = page_details['agent'].unique()

print(list(page_details['project'].unique()))

print(list(page_details['access'].unique()))

print(list(page_details['agent'].unique()))
train_df = train_df.merge(page_details, how="inner", left_index=True, right_index=True)
def graph_by(plot_hue, graph_columns):

    train_project_df = train_df.groupby(plot_hue).sum().T

    train_project_df.index = pd.to_datetime(train_project_df.index)

    train_project_df = train_project_df.groupby(pd.TimeGrouper('M')).mean().dropna()

    train_project_df['month'] = 100*train_project_df.index.year + train_project_df.index.month

    train_project_df = train_project_df.reset_index(drop=True)

    train_project_df = pd.melt(train_project_df, id_vars=['month'], value_vars=graph_columns)

    fig = plt.figure(1,figsize=[12,10])

    ax = sns.pointplot(x="month", y="value", hue=plot_hue, data=train_project_df)

    ax.set(xlabel='Year-Month', ylabel='Mean Hits')
graph_by("project", project_columns)
graph_by("project", [x for i,x in enumerate(project_columns) if i!=2])
graph_by("access", access_columns)
graph_by("agent", agents_columns)
graph_by("agent", agents_columns[0])
graph_by("agent", agents_columns[1])