import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
#import the data
athlete = pd.read_csv('../input/athlete_events.csv',index_col = False)
regions = pd.read_csv('../input/noc_regions.csv',index_col = False)
athlete.head()
athlete.Medal = athlete.Medal.fillna(0)

summ = athlete[athlete.Season == 'Summer'] 
wint = athlete[athlete.Season == 'Winter']
# Group data according to gold
counts_gold = {}
for key,values in athlete.iterrows():
    if(values['Medal'] == 'Gold'):
        if(counts_gold.get(values.Team) == None):
            counts_gold[values.Team] = 1
        else:
            counts_gold[values.Team] +=1  
gdata = pd.concat({k:pd.Series(v) for k, v in counts_gold.items()}).unstack().astype(float).reset_index()
gdata.columns = ['Team', 'gold']
gdata.info()
g_data = gdata.sort_values(by='gold',ascending=False)

counts_silver = {}
for key,values in athlete.iterrows():
    if(values['Medal'] == 'Silver'):
        if(counts_silver.get(values.Team) == None):
            counts_silver[values.Team] = 1
        else:
            counts_silver[values.Team] +=1
            
sdata = pd.concat({k:pd.Series(v) for k, v in counts_silver.items()}).unstack().astype(float).reset_index()
sdata.columns = ['Team', 'silver']
s_data = sdata.sort_values(by='silver',ascending=False)

counts_bronze = {}
for key,values in athlete.iterrows():
    if(values['Medal'] == 'Bronze'):
        if(counts_bronze.get(values.Team) == None):
            counts_bronze[values.Team] = 1
        else:
            counts_bronze[values.Team] +=1
            
bdata = pd.concat({k:pd.Series(v) for k, v in counts_bronze.items()}).unstack().astype(float).reset_index()
bdata.columns = ['Team', 'bronze']
b_data = bdata.sort_values(by='bronze',ascending=False)

data1 = {}
data1['Team'] = athlete.Team.unique()

data1=pd.DataFrame(data1)

data1 = pd.merge(data1, gdata, on='Team')
data1 =pd.merge(data1, sdata, on='Team')
data1 =pd.merge(data1, bdata, on='Team')

data1 = data1.sort_values(by=['gold','silver','bronze'],ascending=False)
data2 = athlete.loc[:,['Team','NOC']]

df = pd.merge(data1, data2, on='Team')
df = df.drop_duplicates()
df.head()

import plotly.graph_objs as go
data = [ dict(
        type = 'choropleth',
        locations = df.NOC,
        z = df.gold,
        text = df.Team,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],[0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            )
        ),
        
        zmin = 0,
        
        colorbar = dict(
            title = 'Medal'
        ),
    ) ]

layout = dict(
    title = '',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
url = py.plot(fig, filename='d3-world-map')










