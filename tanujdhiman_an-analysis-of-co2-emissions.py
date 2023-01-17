# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
data = pd.read_csv('../input/co2-ghg-emissionsdata/co2_emission.csv')
data.head()
data.info()
data.describe()
sns.heatmap(data.corr(), annot = True, cmap = 'coolwarm', center = 0)
data.hist(figsize= (10, 3))
fig = px.scatter(data, x="Year", y="Entity", 
                 color="Year",
                 size='Year', 
                 hover_data=['Annual CO₂ emissions (tonnes )', 'Code'], 
                 title = "CO2 Emissions")
fig.show()
fig3 = go.Figure(data=go.Scatter(
    y = data['Entity'],
    mode='markers',
    marker=dict(
        size=16,
        color=data['Year'], #set color equal to a variable
        colorscale='Viridis', # one of plotly colorscales
        showscale=True
    )
))
fig3.show()