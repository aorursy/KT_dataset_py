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
df = pd.read_csv('../input/who_suicide_statistics.csv')
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.barplot(x='age',y='suicides_no',data=df,palette='Blues_d',
                order=['5-14 years','15-24 years','25-34 years','35-54 years',
                      '55-74 years','75+ years'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
ax.set_xlabel('Age')
ax.set_ylabel('Number of Suicides')
ax.set_title('Dependence of Suicide on Age')
import geopandas as gpd
df2 = df.groupby('country')['suicides_no'].sum()
df2 = df2.to_frame()
df2['country']=df2.index
df2 = df2.reset_index(drop=True)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
merged = world.set_index('name').join(df2.set_index('country'))
merged = merged.fillna(value=0)
variable = 'suicides_no'
vmin, vmax = 120, 220
fig, ax = plt.subplots(1, figsize=(10, 6))
merged.plot(column='suicides_no', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
# import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
data = [go.Bar(x=df.age,y=df.suicides_no)]
layout = dict(title = "Dependence of Suicide on Age",
              xaxis= dict(title= 'Age',ticklen= 5,zeroline= False))
fig = dict(data = data, layout = layout)
iplot(fig)
data = [ dict(
        type = 'choropleth',
        locations = merged['iso_a3'],
        z = np.log(merged['suicides_no']+1),
        autocolorscale = False,
locationmode = 'ISO-3') ]

layout = dict(
    title = 'Number of Suicides (log)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig)
