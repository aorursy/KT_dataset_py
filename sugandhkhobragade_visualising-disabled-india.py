

#importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import geopandas as gpd

from matplotlib import cm

import matplotlib

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING

import plotly.graph_objects as go

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/disabled-community-dataset/disabled_community_dataset.csv')

df.head()
df.info()
df.isna().sum()
df.columns
df = df[df.state != 'india']
import plotly.graph_objects as go



state = list(df.state)

disabled = list(df.number_disabled)



fig= go.Figure(go.Treemap(

    

    labels =  state,

    parents=[""]*len(state),

    values =  disabled,

    textinfo = "label+percent entry"

))

fig.update_layout(

    title_text= 'Distribution of Disabled people in country',

    paper_bgcolor='rgb(233,233,233)',

    autosize=False,

    width= 800,

    height=700,)



fig.show()
gdf = gpd.read_file('../input/india-states/Igismap/Indian_States.shp')

gdf['st_nm'] = gdf['st_nm'].str.lower()

gdf.at[0, 'st_nm']= 'andaman and nicobar islands'

gdf.at[1,'st_nm'] = 'arunachal pradesh'

gdf.at[23,'st_nm'] = 'delhi'
new_row = {'state':'telangana', 'number_disabled': 2266607}

df = df.append(new_row , ignore_index=True )



merged = gdf.merge(df , left_on='st_nm', right_on='state')
from matplotlib.colors import Normalize

from matplotlib import cm

cmap = 'YlGn'



merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[:])

merged['coords'] = [coords[0] for coords in merged['coords']]



sns.set_context("poster")

sns.set_style("darkgrid")

plt.style.use('fivethirtyeight')



figsize = (25, 25)





ax= merged.plot(column= 'number_disabled', cmap= cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title("Number of Disabled people ", size = 25)

for idx, row in merged.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['number_disabled'], horizontalalignment='center', bbox={'facecolor': 'gold', 'alpha':0.8, 'pad': 2, 'edgecolor':'blue'})



norm = Normalize(vmin=merged['number_disabled'].min(), vmax= merged['number_disabled'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
df2 = df.sort_values('percent_disabled', ascending = False)
sns.set_context("talk")

sns.set(rc={'axes.facecolor':'khaki', 'figure.facecolor':'khaki'})

plt.figure(figsize=(12,14))

#plt.style.use('fivethirtyeight')



ax = sns.barplot( x = 'percent_disabled', y = 'state', data = df2, palette = 'dark')

plt.title('Percentage of Disabled people in state (wrt population)',size = 20)

plt.xlabel('percentage of population')



for p in ax.patches:

        ax.annotate("%.2f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")
sns.set_context("poster")

sns.set_style("darkgrid")

plt.style.use('fivethirtyeight')



figsize = (25, 25)





ax= merged.plot(column= 'literacy_rate_disabled', cmap= cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title("Literacy rate of Disabled people ", size = 25)

for idx, row in merged.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['literacy_rate_disabled'], horizontalalignment='center', bbox={'facecolor': 'yellow', 'alpha':0.8, 'pad': 2, 'edgecolor':'blue'})



norm = Normalize(vmin=merged['literacy_rate_disabled'].min(), vmax= merged['literacy_rate_disabled'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
df['diff_literacy'] = df['literacy_rate_general'] - df['literacy_rate_disabled']

df3 = df.sort_values('diff_literacy', ascending = False)

df3['diff_literacy'] = df3['diff_literacy'].round(decimals=3)
sns.set_context("talk")

sns.set(rc={'axes.facecolor':'aqua', 'figure.facecolor':'aqua'})

plt.figure(figsize=(12,14))

#plt.style.use('bmh')



ax = sns.barplot( x = 'diff_literacy', y = 'state', data = df3, palette = 'bright')

plt.title('Difference of Literacy of Disabled people in state (wrt general population)',size = 20)

plt.xlabel('percentage')



for p in ax.patches:

        ax.annotate("%.2f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")
sns.set_context("poster")

sns.set_style("darkgrid")

plt.style.use('fivethirtyeight')



figsize = (25, 25)





ax= merged.plot(column= 'workforce_rate_disabled', cmap= cmap, figsize=figsize, scheme='equal_interval',edgecolor='black')

ax.set_title("Workforce rate of Disabled people ", size = 25)

for idx, row in merged.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['workforce_rate_disabled'], horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'blue'})



norm = Normalize(vmin=merged['workforce_rate_disabled'].min(), vmax= merged['workforce_rate_disabled'].max())

n_cmap = cm.ScalarMappable(norm=norm, cmap= cmap)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)

ax.set_axis_off()

plt.axis('equal')

plt.show()
df['diff_work'] = df['workforce_rate_general'] - df['workforce_rate_disabled']

df4 = df.sort_values('diff_work', ascending = False)

df4['diff_work'] = df4['diff_work'].round(decimals=3)
sns.set_context("talk")

sns.set(rc={'axes.facecolor':'snow', 'figure.facecolor':'snow'})

plt.figure(figsize=(12,14))

plt.style.use('fivethirtyeight')



ax = sns.barplot( x = 'diff_work', y = 'state', data = df4, palette = 'dark')

plt.title('Difference of Workforce of Disabled people in state (wrt general population)',size = 20)

plt.xlabel('percentage')



for p in ax.patches:

        ax.annotate("%.2f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),

            xytext=(5, 0), textcoords='offset points', ha="left", va="center")