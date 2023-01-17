
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
athlete.info()
athlete.Medal = athlete.Medal.fillna(0)
athlete.head()
summ = athlete[athlete.Season == 'Summer'] 
wint = athlete[athlete.Season == 'Winter']

fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(x="Sex", hue="Medal", data=athlete,ax=ax)
fig, ax = plt.subplots(figsize=(8,6))
a = athlete[~athlete['Medal'].isnull()].Sex
sns.countplot(x=a, hue="Season", data=athlete,ax=ax)
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
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="Team", y="gold", data=g_data.head(10),ax=ax)
import plotly.plotly as py
import plotly.graph_objs as go
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

sdata.head()
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
# prepare data frames
dfd = data1.head(20)
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = dfd.Team,
                y = dfd.gold,
                name = "gold",
                marker = dict(color = 'rgba(255, 223, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace2 
trace2 = go.Bar(
                x = dfd.Team,
                y = dfd.silver,
                name = "silver",
                marker = dict(color = 'rgba(192, 192, 192, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace3
trace3 = go.Bar(
                x = dfd.Team,
                y = dfd.bronze,
                name = "bronze",
                marker = dict(color = 'rgba(205, 127, 50, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
data = [trace1, trace2, trace3]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# Group data according to gold for summer
counts_gold = {}
for key,values in summ.iterrows():
    if(values['Medal'] == 'Gold'):
        if(counts_gold.get(values.Team) == None):
            counts_gold[values.Team] = 1
        else:
            counts_gold[values.Team] +=1  
gdata = pd.concat({k:pd.Series(v) for k, v in counts_gold.items()}).unstack().astype(float).reset_index()
gdata.columns = ['Team', 'gold']
g_data = gdata.sort_values(by='gold',ascending=False)
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="Team", y="gold", data=g_data.head(10),ax=ax).set_title('Top 10 countries in summer oylmpics')
import plotly.plotly as py
import plotly.graph_objs as go

counts_silver = {}
for key,values in summ.iterrows():
    if(values['Medal'] == 'Silver'):
        if(counts_silver.get(values.Team) == None):
            counts_silver[values.Team] = 1
        else:
            counts_silver[values.Team] +=1
            
sdata = pd.concat({k:pd.Series(v) for k, v in counts_silver.items()}).unstack().astype(float).reset_index()
sdata.columns = ['Team', 'silver']
s_data = sdata.sort_values(by='silver',ascending=False)

counts_bronze = {}
for key,values in summ.iterrows():
    if(values['Medal'] == 'Bronze'):
        if(counts_bronze.get(values.Team) == None):
            counts_bronze[values.Team] = 1
        else:
            counts_bronze[values.Team] +=1
            
bdata = pd.concat({k:pd.Series(v) for k, v in counts_bronze.items()}).unstack().astype(float).reset_index()
bdata.columns = ['Team', 'bronze']
b_data = bdata.sort_values(by='bronze',ascending=False)

data1 = {}
data1['Team'] = summ.Team.unique()

data1=pd.DataFrame(data1)

data1 = pd.merge(data1, gdata, on='Team')
data1 =pd.merge(data1, sdata, on='Team')
data1 =pd.merge(data1, bdata, on='Team')

data1 = data1.sort_values(by=['gold','silver','bronze'],ascending=False)


# prepare data frames
dfd = data1.head(20)
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = dfd.Team,
                y = dfd.gold,
                name = "gold",
                marker = dict(color = 'rgba(255, 223, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace2 
trace2 = go.Bar(
                x = dfd.Team,
                y = dfd.silver,
                name = "silver",
                marker = dict(color = 'rgba(192, 192, 192, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace3
trace3 = go.Bar(
                x = dfd.Team,
                y = dfd.bronze,
                name = "bronze",
                marker = dict(color = 'rgba(205, 127, 50, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
data = [trace1, trace2, trace3]
layout = go.Layout(title='Top countries in summer oylmpics',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# Group data according to gold
counts_gold = {}
for key,values in wint.iterrows():
    if(values['Medal'] == 'Gold'):
        if(counts_gold.get(values.Team) == None):
            counts_gold[values.Team] = 1
        else:
            counts_gold[values.Team] +=1  
gdata = pd.concat({k:pd.Series(v) for k, v in counts_gold.items()}).unstack().astype(float).reset_index()
gdata.columns = ['Team', 'gold']
g_data = gdata.sort_values(by='gold',ascending=False)
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.barplot(x="Team", y="gold", data=g_data.head(10),ax=ax).set_title('Top 10 countries in winter oylmpics')
import plotly.plotly as py
import plotly.graph_objs as go

counts_silver = {}
for key,values in wint.iterrows():
    if(values['Medal'] == 'Silver'):
        if(counts_silver.get(values.Team) == None):
            counts_silver[values.Team] = 1
        else:
            counts_silver[values.Team] +=1
            
sdata = pd.concat({k:pd.Series(v) for k, v in counts_silver.items()}).unstack().astype(float).reset_index()
sdata.columns = ['Team', 'silver']
s_data = sdata.sort_values(by='silver',ascending=False)

counts_bronze = {}
for key,values in wint.iterrows():
    if(values['Medal'] == 'Bronze'):
        if(counts_bronze.get(values.Team) == None):
            counts_bronze[values.Team] = 1
        else:
            counts_bronze[values.Team] +=1
            
bdata = pd.concat({k:pd.Series(v) for k, v in counts_bronze.items()}).unstack().astype(float).reset_index()
bdata.columns = ['Team', 'bronze']
b_data = bdata.sort_values(by='bronze',ascending=False)

data1 = {}
data1['Team'] = wint.Team.unique()

data1=pd.DataFrame(data1)

data1 = pd.merge(data1, gdata, on='Team')
data1 =pd.merge(data1, sdata, on='Team')
data1 =pd.merge(data1, bdata, on='Team')

data1 = data1.sort_values(by=['gold','silver','bronze'],ascending=False)


# prepare data frames
dfd = data1.head(20)
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = dfd.Team,
                y = dfd.gold,
                name = "gold",
                marker = dict(color = 'rgba(255, 223, 0, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace2 
trace2 = go.Bar(
                x = dfd.Team,
                y = dfd.silver,
                name = "silver",
                marker = dict(color = 'rgba(192, 192, 192, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
# create trace3
trace3 = go.Bar(
                x = dfd.Team,
                y = dfd.bronze,
                name = "bronze",
                marker = dict(color = 'rgba(205, 127, 50, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dfd.Team)
data = [trace1, trace2, trace3]
layout = go.Layout(title='Top countries in Winter Oylmpics',barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
gusa = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'United States') and (values['Medal'] == 'Gold')):
        if(gusa.get(values.Year) == None):
            gusa[values.Year] = 1
        else:
            gusa[values.Year] +=1
            
gusa = pd.concat({k:pd.Series(v) for k, v in gusa.items()}).unstack().astype(float).reset_index()
gusa.columns = ['Year', 'gold']


#silver for usa
susa = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'United States') and (values['Medal'] == 'Silver')):
        if(susa.get(values.Year) == None):
            susa[values.Year] = 1
        else:
            susa[values.Year] +=1
            
susa = pd.concat({k:pd.Series(v) for k, v in susa.items()}).unstack().astype(float).reset_index()
susa.columns = ['Year', 'silver']  


# bronze for usa

busa = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'United States') and (values['Medal'] == 'Bronze')):
        if(busa.get(values.Year) == None):
            busa[values.Year] = 1
        else:
            busa[values.Year] +=1
            
busa = pd.concat({k:pd.Series(v) for k, v in busa.items()}).unstack().astype(float).reset_index()
busa.columns = ['Year', 'bronze']


#### for Soviet Union

# gold for Soviet Union

gsu = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'Soviet Union') and (values['Medal'] == 'Gold')):
        if(gsu.get(values.Year) == None):
            gsu[values.Year] = 1
        else:
            gsu[values.Year] +=1
            
gsu = pd.concat({k:pd.Series(v) for k, v in gsu.items()}).unstack().astype(float).reset_index()
gsu.columns = ['Year', 'gold']


ssu = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'Soviet Union') and (values['Medal'] == 'Silver')):
        if(ssu.get(values.Year) == None):
            ssu[values.Year] = 1
        else:
            ssu[values.Year] +=1
            
ssu = pd.concat({k:pd.Series(v) for k, v in ssu.items()}).unstack().astype(float).reset_index()
ssu.columns = ['Year', 'silver']

bsu = {}
for key,values in athlete.iterrows():
    if((values['Team'] == 'Soviet Union') and (values['Medal'] == 'Bronze')):
        if(bsu.get(values.Year) == None):
            bsu[values.Year] = 1
        else:
            bsu[values.Year] +=1
            
bsu = pd.concat({k:pd.Series(v) for k, v in bsu.items()}).unstack().astype(float).reset_index()
bsu.columns = ['Year', 'bronze']

# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1 USA Gold
trace1 = go.Scatter(
                    x = gusa.Year,
                    y = gusa.gold,
                    mode = "lines+markers",
                    name = "USA gold",
                    marker = dict(color = 'rgba(240,230,140 0.8)'),
                    )
# Creating trace2 USA silver
trace2 = go.Scatter(
                    x = susa.Year,
                    y = susa.silver,
                    mode = "lines+markers",
                    name = "USA silver",
                    marker = dict(color = 'rgba(211,211,211, 0.8)'),
                    )
# Creating trace3 USA bronze
trace3 = go.Scatter(
                    x = busa.Year,
                    y = busa.bronze,
                    mode = "lines+markers",
                    name = "USA bronze",
                    marker = dict(color = 'rgba(220,165,112)'),
                    )
# Creating trace4 Soviet gold
trace4 = go.Scatter(
                    x = gsu.Year,
                    y = gsu.gold,
                    mode = "lines+markers",
                    name = "Soviet Union gold",
                    marker = dict(color = 'rgba(218,165,32, 0.8)'),
                    )
# Creating trace5 Soviet silver
trace5 = go.Scatter(
                    x = ssu.Year,
                    y = ssu.silver,
                    mode = "lines+markers",
                    name = "Soviet Union Silver",
                    marker = dict(color = 'rgba(128,128,128, 0.8)'),
                    )
# Creating trace6 Soviet bronze
trace6 = go.Scatter(
                    x = bsu.Year,
                    y = bsu.bronze,
                    mode = "lines+markers",
                    name = "Soviet Union bronze",
                    marker = dict(color = 'rgba(144,89,35, 0.8)'),
                    )

data = [trace1, trace2,trace3,trace4,trace5,trace6]
layout = dict(title = 'Comparision between USA and Soviet union in years',
              xaxis= dict(title= 'Years',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
athlete = athlete[pd.notnull(athlete['Height'])]
athlete = athlete[pd.notnull(athlete['Weight'])]
# prepare data frames
athg = athlete[athlete.Medal == 'Gold']
aths = athlete[athlete.Medal == 'Silver']
athb = athlete[athlete.Medal == 'Bronze']

# creating trace1
trace1 =go.Scatter(
                    x = athg.Weight,
                    y = athg.Height,
                    mode = "markers",
                    name = "Gold",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= athg.Medal)
# creating trace2
trace2 =go.Scatter(
                    x = aths.Weight,
                    y = aths.Height,
                    mode = "markers",
                    name = "Silver",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= aths.Medal)
# creating trace3
trace3 =go.Scatter(
                    x = athb.Weight,
                    y = athb.Height,
                    mode = "markers",
                    name = "Bronze",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= athb.Medal)
data = [trace1, trace2, trace3]
layout = dict(title = 'Corelation between Height and Weight',
              xaxis= dict(title= 'Weight',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Height',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)




