# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/2017.csv")
df.head(10)
sum(df["Country"].value_counts())
df['Country'].isnull().sum() #clean dataset yippueee
dl = df.drop(["Happiness.Rank"],axis=1)
# Compute the correlation matrix
corr = dl.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
scatter=plt.plot(df["Economy..GDP.per.Capita."],df["Happiness.Score"],'*')
plt.xlabel('Economy..GDP.per.Capita.')
plt.ylabel('Happiness.Score')
plt.setp(scatter, color='r')
plt.show()
scatter=plt.plot(df["Health..Life.Expectancy."],df["Happiness.Score"],'*')
plt.xlabel('Health..Life.Expectancy.')
plt.ylabel('Happiness.Score')
plt.setp(scatter, color='b')
plt.show()
scatter=plt.plot(df["Freedom"],df["Happiness.Score"],'*')
plt.xlabel('Freedom.')
plt.ylabel('Happiness.Score')
plt.setp(scatter, color='g')
plt.show()


#we see from 3 plot GDP has solid linear relationship with Happiness Score, while Life Expectancy and Freedom doesn't have a defined relationship
scatter=plt.plot(df["Generosity"],df["Happiness.Score"],'*')
plt.xlabel('Generosity')
plt.ylabel('Happiness.Score')
plt.setp(scatter, color='y')
plt.show()
#plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as py

#preparing longitude and latitude 
df1 = df.head(10)
df1['latitude']= pd.Series([60.4720,56.2639,64.9631,46.8182,61.9241,52.1326,56.1304,-40.9006,60.1282,-25.2744],index=df1.index)
df1['longitude'] = pd.Series([8.4689,9.5018,-19.0208,8.2275,25.7482,5.2913,-106.3468,174.8860,18.6435,133.7751],index=df1.index)
data1 = [dict(
    type='scattergeo',
    lon = df1['longitude'],
    lat = df1['latitude'],
    text = df1['Country'],
    mode = 0,
    marker = dict(
    symbol = "triangle-up",    
    cmin = 0,
    color = "green",
    cmax = df1['Happiness.Rank'].min(),
    colorbar=dict(
                title="Happiness"
            )
    )
    
)]
layout = dict(
    title = 'Where the heck is happiness in this world',
    hovermode='closest',
    geo = dict(showframe=False, countrywidth=1, showcountries=True,
               showcoastlines=True, projection=dict(type='Mercator'))
)
fig = py.Figure(data=data1, layout=layout)
iplot(fig)
