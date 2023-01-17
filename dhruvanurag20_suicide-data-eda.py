# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import cm 

import seaborn as sns





from plotly.offline import init_notebook_mode, plot

import plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True)



import geopandas as gpd
data = pd.read_csv('../input/suicides-in-india/Suicides in India 2001-2012.csv')

data.head()
data.shape
data['State'].unique()
frame = data[data['State'] != 'Total (All India)']

frame = frame[frame['State'] != 'Total (States)']

frame = frame[frame['State'] != 'Total (Uts)']

frame.shape
frame['State'].unique()
final_frame = frame[frame['Type_code'] == 'Causes']

final_frame.shape
final_frame_new = final_frame[final_frame['Type'] != 'Other Causes (Please Specity)']

final_frame_new.shape
final_frame_new = final_frame_new[final_frame_new['Type'] != 'Causes Not known']

final_frame_new = final_frame_new[final_frame_new['Type'] != 'Other Prolonged Illness']

final_frame_new = final_frame_new[final_frame_new['Type'] != 'Not having Children(Barrenness/Impotency']

final_frame_new = final_frame_new[final_frame_new['Type'] != 'Not having Children (Barrenness/Impotency']

final_frame_new.shape

data_new = pd.DataFrame(final_frame_new)

data_new.shape
data_new
import plotly.express as px

fig = px.bar(data_new,

    x = data_new['Year'].unique(),

    y = data_new.groupby('Year').sum()['Total'],

    color_discrete_sequence=["#fc032c"]

)



fig.update_layout(

    title="Year-wise suicide",

    xaxis_title="Year",

    yaxis_title="Suicide Count",

)



fig.show()
fig = px.bar(data_new,

    x = data_new['Age_group'].unique(),

    y = data_new.groupby('Age_group').sum()['Total'],

    color_discrete_sequence=["#516882"],

)



fig.update_layout(

    title="Age Group suicide",

    xaxis_title="Age group",

    yaxis_title="Suicide Count",

)



fig.show()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Bar(

    x= data_new['Year'].unique(),

    y= data_new.groupby([data_new.Gender == 'Male' , 'Year']).sum()['Total'],

    name='Female',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x= data_new['Year'].unique(),

    y= data_new.groupby([data_new["Gender"] == 'Female' , 'Year']).sum()['Total'],

    name='Male',

    marker_color='lightsalmon'

))





fig.update_layout(barmode='group', xaxis_tickangle=-90)

fig.show()
india_map = gpd.read_file('../input/final-shp/Indian_States.shp')

data_new['State'].replace({'A & N Islands':'Andaman & Nicobar Island',

                        'Delhi (Ut)':'NCT of Delhi',

                        'D & N Haveli':'Dadra and Nagar Haveli',

                       }, inplace = True)



india_map['st_nm'].replace({'Telangana':'Andhra Pradesh',

                        'Dadara & Nagar Havelli': 'Dadra and Nagar Haveli',

                       }, inplace = True)



india_map.st_nm.unique()
data_new.State.unique()
india_map.rename(columns = {'st_nm':'State'}, inplace = True)

suicide_data_states = data_new.groupby(['State']).agg({'Total':'sum'})
suicide_data_map = india_map.merge(suicide_data_states, left_on='State', right_on='State')



suicide_data_map['coords'] = suicide_data_map['geometry'].apply(lambda x: x.representative_point().coords[:])

suicide_data_map['coords'] = [coords[0] for coords in suicide_data_map['coords']]



fig, ax = plt.subplots(figsize=(22, 15))



cmap = 'Reds'



ax = suicide_data_map.plot(ax=ax, cmap=cmap,column = 'Total',scheme = 'equal_interval',edgecolor = 'black')

ax.set_facecolor('white')

ax.set_title('Suicide Cases per State')



for idx, row in suicide_data_map.iterrows():

    ax.text(row.coords[0], row.coords[1], s=row['Total'], 

           horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = matplotlib.colors.Normalize(vmin=suicide_data_map['Total'].min(), vmax= suicide_data_map['Total'].max())

n_cmap = cm.ScalarMappable(cmap= cmap, norm = norm)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)



#suicide_map[suicide_map['Total'] > 0].plot(ax=ax, cmap=cmap, markersize=1)



plt.xticks([])

plt.yticks([])

plt.show()
q_stats = data_new.groupby(['Type']).sum()



q_stats.Total.sort_values(ascending = False)
f_p = data_new[data_new['Type'] == 'Family Problems']

f_p = f_p.groupby(['Type','State']).sum()



stats_1 = f_p.drop(['Year'], axis = 1)



f_p_stats = pd.DataFrame(stats_1)
suicide_data_map = india_map.merge(f_p_stats, left_on='State', right_on='State')



suicide_data_map['coords'] = suicide_data_map['geometry'].apply(lambda x: x.representative_point().coords[:])

suicide_data_map['coords'] = [coords[0] for coords in suicide_data_map['coords']]



fig, ax = plt.subplots(figsize=(22, 15))



cmap = 'RdPu'



ax = suicide_data_map.plot(ax=ax, cmap=cmap,column = 'Total',scheme = 'equal_interval',edgecolor = 'black')

ax.set_facecolor('white')

ax.set_title('Suicide Cases per State')



for idx, row in suicide_data_map.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Total'], 

           horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = matplotlib.colors.Normalize(vmin=suicide_data_map['Total'].min(), vmax= suicide_data_map['Total'].max())

n_cmap = cm.ScalarMappable(cmap= cmap, norm = norm)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)



#suicide_map[suicide_map['Total'] > 0].plot(ax=ax, cmap=cmap, markersize=1)



plt.xticks([])

plt.yticks([])

plt.show()
mi = data_new[data_new['Type'] == 'Insanity/Mental Illness']

mi = mi.groupby(['Type','State']).sum()



stats_2 = mi.drop(['Year'], axis = 1)



mi_stats = pd.DataFrame(stats_2)
suicide_data_map = india_map.merge(mi_stats, left_on='State', right_on='State')



suicide_data_map['coords'] = suicide_data_map['geometry'].apply(lambda x: x.representative_point().coords[:])

suicide_data_map['coords'] = [coords[0] for coords in suicide_data_map['coords']]



fig, ax = plt.subplots(figsize=(22, 15))



cmap = 'Blues'



ax = suicide_data_map.plot(ax=ax, cmap=cmap,column = 'Total',scheme = 'equal_interval',edgecolor = 'black')

ax.set_facecolor('white')

ax.set_title('Suicide Cases per State')



for idx, row in suicide_data_map.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Total'], 

           horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = matplotlib.colors.Normalize(vmin=suicide_data_map['Total'].min(), vmax= suicide_data_map['Total'].max())

n_cmap = cm.ScalarMappable(cmap= cmap, norm = norm)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)



#suicide_map[suicide_map['Total'] > 0].plot(ax=ax, cmap=cmap, markersize=1)



plt.xticks([])

plt.yticks([])

plt.show()
love = data_new[data_new['Type'] == 'Love Affairs']

love = love.groupby(['Type','State']).sum()



stats_3 = mi.drop(['Year'], axis = 1)



love_stats = pd.DataFrame(stats_3)
suicide_data_map = india_map.merge(love_stats, left_on='State', right_on='State')



suicide_data_map['coords'] = suicide_data_map['geometry'].apply(lambda x: x.representative_point().coords[:])

suicide_data_map['coords'] = [coords[0] for coords in suicide_data_map['coords']]



fig, ax = plt.subplots(figsize=(22, 15))



cmap = 'BuGn'



ax = suicide_data_map.plot(ax=ax, cmap=cmap,column = 'Total',scheme = 'equal_interval',edgecolor = 'black')

ax.set_facecolor('white')

ax.set_title('Suicide Cases per State')



for idx, row in suicide_data_map.iterrows():

   ax.text(row.coords[0], row.coords[1], s=row['Total'], 

           horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})



norm = matplotlib.colors.Normalize(vmin=suicide_data_map['Total'].min(), vmax= suicide_data_map['Total'].max())

n_cmap = cm.ScalarMappable(cmap= cmap, norm = norm)

n_cmap.set_array([])

ax.get_figure().colorbar(n_cmap)



#suicide_map[suicide_map['Total'] > 0].plot(ax=ax, cmap=cmap, markersize=1)



plt.xticks([])

plt.yticks([])

plt.show()