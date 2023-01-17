# importing libraries



import numpy as np

import pandas as pd

import plotly.express as px
# importing data



df = pd.read_csv("../input/habermans-survival-data-set/haberman.csv", names = ['age', 'year', 'nodes', 'status'])
# inspecting data



df.head()
# inspecting size of the dataset



df.shape
# inspecting datatypes of columns



df.dtypes
# renaming status



df.rename(columns = {'status': 'survived'}, inplace = True)
# mapping status for better understanding



df['survived'].replace({1 : 1, 2 : 0}, inplace = True)
df.describe()
df['survived'].value_counts()
px.box(x = df['nodes'], color = df['survived'], notched=True, title = 'Box plot of nodes')
px.bar(x = df['year'].value_counts().index.tolist(), y = df['year'].value_counts().values.tolist(), title = 'Number of operations per year')
df_byyear = df.groupby(df['year']).sum()

px.bar(x = df_byyear.index.tolist(), y = df_byyear['nodes'], title = 'Total number of nodes per year')
df_byyear = df.groupby(df['year']).median()

mi = min(df_byyear.index.tolist())

ma = max(df_byyear.index.tolist())

theta = []

for value in df_byyear.index.tolist():

    theta.append(((value - mi) * 360 ) / (ma - mi))

px.bar_polar(theta = theta, r = df_byyear['age'], title = 'Average age per year')
px.histogram(x = df['age'], color = df['survived'], title = 'histogram of age')
px.scatter(x = df['age'], y = df['nodes'], color = df['survived'], trendline = "ols", size = df['nodes'], title = 'Age vs Nodes scatter plot')
px.box(x = df['year'], y = df['nodes'], color = df['survived'], title = 'Box plot of age vs year')
px.parallel_coordinates(df, color = 'survived')