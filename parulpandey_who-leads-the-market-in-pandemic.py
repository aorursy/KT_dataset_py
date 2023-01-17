import numpy as np 
import pandas as pd 
import os

# Visualisation libraries
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
!pip install chart_studio
import chart_studio.plotly as py
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
#py.init_notebook_mode(connected=True)

#Geographical Plotting
import folium
from folium import Choropleth, Circle, Marker
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster


# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
plt.style.use("fivethirtyeight")# for pretty graphs

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Disable warnings 
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/market-capital-addition-during-pandemic/companies_market_cap.csv')
df.head()
df.info()

fig = px.bar(df, x='Company', y='Market cap added',color='Change',)
fig.show()
df_change = df.sort_values('Change')
fig = px.bar(df_change, y='Company', x='Change',height=800)
fig.show()
labels = df['Sector'].value_counts().index
values = df['Sector'].value_counts().values
fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label',
                             insidetextorientation='radial')])
fig.show()
df_tech = df[df['Sector']=='Technology']
fig = px.bar(df_tech, x='Company', y='Market cap added',color='Company',height=800)
fig.show()




fig = px.scatter(df_tech, x="Company", y="Change",
        size="Change", color="Company",
                 hover_name="Company",size_max=60)
fig.show()
df_health = df[df['Sector']=='Healthcare']
fig = px.bar(df_health, x='Company', y='Market cap added',color='Company',height=800)
fig.show()



fig = px.scatter(df_health, x="Company", y="Change",
        size="Change", color="Company",
                 hover_name="Company",size_max=60)
fig.show()
df_cd = df[df['Sector']=='Consumer discretionary']
fig = px.bar(df_cd, x='Company', y='Market cap added',color='Company',height=800)
fig.show()

fig = px.scatter(df_cd, x="Company", y="Change",
        size="Change", color="Company",
                 hover_name="Company",size_max=60)
fig.show()
