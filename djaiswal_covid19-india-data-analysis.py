# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
df_complete = pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv')
df_patient_wise = pd.read_csv('../input/covid19-corona-virus-india-dataset/patients_data.csv')
df_complete.head()
print(list(df_complete.columns))
df_complete.rename(columns={"Name of State / UT": "state", "Total Confirmed cases (Indian National)": "total_confirmed_cases_indian","Total Confirmed cases ( Foreign National )":"total_confirmed_cases_foreign", "Cured/Discharged/Migrated" : "Cured","Total Confirmed cases": "total_confirmed_cases"}, inplace=True)
print(list(df_complete.columns))
# calculate correlation matrix
corr = df_complete.corr()
# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
df_complete.plot(kind='scatter', x='total_confirmed_cases', y='Death')

#histogram : distribution of total_confirmed_cases_indian
df_complete['total_confirmed_cases_indian'].plot(kind='hist', bins=50, figsize=(12,6), facecolor='grey',edgecolor='black')
#histogram : distribution of total_confirmed_cases_foreign
df_complete['total_confirmed_cases_foreign'].plot(kind='hist', bins=50, figsize=(12,6), facecolor='grey',edgecolor='black')
sns.catplot(x="Death",  hue="state", data=df_complete, kind="count", height=5, aspect=2.8)


sns.pairplot(df_complete)

df_complete.boxplot('Death')
df_complete['total_confirmed_cases'].describe()
df_complete['total_confirmed_cases'].describe().plot()
top=df_complete.nlargest(20,'total_confirmed_cases')
sns.stripplot(x='state',y='total_confirmed_cases',data=top)
fig,ax = plt.subplots(1)
fig.set_size_inches(20,12)
sns.barplot(df_complete["state"],df_complete["total_confirmed_cases"])
plt.xticks(rotation=65,fontsize=8)
plt.title("Total Confirmed Cases Statewise as on ",fontsize=12)
plt.xlabel("State/Union Territory",fontsize=14)
plt.ylabel("Total Confirmed Cases",fontsize=14)
plt.show()
from matplotlib import style
style.use('ggplot')

df_complete.plot(x='Date',y='total_confirmed_cases',kind='line',linewidth=5,color='R',figsize=(25,15))
plt.ylabel('Corona Cases')

plt.grid()
plt.show()
sns.lmplot(x='total_confirmed_cases',y='Death',data=df_complete,hue='state',size=14)

from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

%matplotlib inline
plt.style.use('ggplot')

geometry = [Point(xy) for xy in zip(df_complete['Longitude'], df_complete['Latitude'])]
geo_df = GeoDataFrame(df_complete, geometry=geometry)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
geo_df.plot(ax=world.plot(figsize=(15, 15)), marker='o', color='red', markersize=15);
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.plotting import gmap

output_file("gmap.html")

map_options = GMapOptions(lat=20.5937, lng=-78.9629, map_type="roadmap", zoom=11)

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
p = gmap("AIzaSyC0-k3vOhCIAKkJMhinVaKByflrwMNA9Lw", map_options, title="India")

source = ColumnDataSource(
    data=dict(lat= df_complete['Latitude'],
              lon=df_complete['Longitude'])
)

p.circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, source=source)

show(p)
import plotly.graph_objects as go


df_complete['text'] = df_complete['state'] + '<br>Total Confirmed cases ' + (df_complete['total_confirmed_cases']).astype(str)

limits = [(100,200),(200,300),(300,400),(400,500),(500,3000)]
colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]
cities = []
scale = 5000


fig = go.Figure()

for i in range(len(limits)):
    lim = limits[i]
    
    fig.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = df_complete['Longitude'],
        lat = df_complete['Latitude'],
        text = df_complete['text'],
        marker = dict(
            size = df_complete['total_confirmed_cases']/10,
            color = colors[i],
            line_color='rgb(40,40,40)',
            line_width=1,
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])))

fig.update_layout(
        title_text = 'COVID19 Confirmed cases India<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope = 'asia',
            landcolor = 'rgb(217, 217, 217)',
        )
    )

fig.show()
