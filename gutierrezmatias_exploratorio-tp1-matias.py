import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
df = pd.read_csv('../input/events.csv',low_memory=False)
df['timestamp']= df['timestamp'].astype('datetime64')
df.head(5)

dfa = df.loc[df['event']=='conversion'].groupby(['person','sku'])['event'].count()
dfa= dfa.reset_index()
dfb= df.loc[df['event']=='checkout'].groupby(['person','sku'])['event'].count()
dfb= dfb.reset_index()
result=pd.merge(dfa,dfb, on = ['person', 'sku'], how='inner')
result.rename(columns={"event_x": "conversion"}, inplace= True)
result.rename(columns={"event_y": "checkout"}, inplace= True)
result.sort_values(by= ['conversion','checkout'], ascending = False)
result.fillna(0)
result.sort_values('conversion', ascending = False).head(10)

visitspercountry = df.loc[(df['event']=='visited site') & (df['country']!='Unknown')].groupby(['country'])['event'].agg(['count']).sort_values('count',ascending = False).reset_index()
visitspercountry.head(10)
plotx = visitspercountry['count'].head(10)/len(df.loc[(df['event']=='visited site') ,'event'])*100
ploty = visitspercountry['country'].head(10)
plt.figure(figsize=(20, 5))
barlist = plt.bar(ploty,plotx,color = 'lightblue',edgecolor='blue')
barlist[0].set_color('g')
plt.xticks(rotation=45)
plt.title('% Visits per site')
plt.xlabel('Countries')
plt.show()

visitsperregion = df.loc[(df['event']=='visited site') & (df['region']!='Unknown')].groupby(['country','region'])['event'].value_counts()
visitsperregion.sort_values(ascending = False,inplace= True)
visitsperregion= visitsperregion.sort_values(ascending = False).head(10).reset_index(drop = True, level = 2).reset_index()
plt.figure(figsize=(20, 5))
plt.bar(visitsperregion['region'],visitsperregion['event'], color = 'rgbkymc',edgecolor='black')
visitscity= df.loc[(df['event']=='visited site') & (df['region']!='Unknown')].groupby(['region','city'])['event'].value_counts().reset_index(drop = True , level = 2)
visitscity.sort_values(ascending = False,inplace= True)
visitscity.to_frame().reset_index()
visitscity.head(20).plot(kind = 'bar')
visitsplatform = df.groupby(['device_type'])['event'].value_counts().reset_index(drop= True, level = 1)
visitsplatform.sort_values(ascending = False,inplace= True)
visitsplatform = visitsplatform.to_frame().reset_index()
visitsplatform
plt.figure(figsize=(10, 2))
plt.barh(visitsplatform['device_type'],visitsplatform['event'], color = 'rgb',edgecolor='black')
visitsos= df.groupby(['operating_system_version'])['event'].value_counts()
visitsos.sort_values(ascending = False,inplace= True)
visitsos.to_frame().reset_index(drop= True, level = 1).reset_index().head(20)
visitsos.head(10).plot(kind = 'bar')
visitsbrowser= df.groupby(['browser_version'])['event'].value_counts()
visitsbrowser.sort_values(ascending = False,inplace= True)
visitsbrowser.to_frame().reset_index(drop=True,level = 1).reset_index().head(10)
visitsbrowser.head(20).plot(kind = 'barh')
visitsscreen= df.groupby(['screen_resolution','device_type' ])['event'].value_counts()
visitsscreen.sort_values(ascending = False,inplace= True)
visitsscreen.to_frame().reset_index(drop=True, level = 2).reset_index().head(10)
visitsscreen.head(20).plot(kind = 'barh')
urls= df.groupby(['url' ])['event'].value_counts()
urls.sort_values(ascending = False,inplace= True)
urls.head(10)
data = [ dict(
        type = 'choropleth',
        showscale = True,
        locations = visitspercountry['country'],
        z = visitspercountry['count'],
        locationmode = 'country names',
        marker = dict(
            line = dict (
                color = 'rgb(200, 200, 200)',
                width = 1
            ) ),
        ) ]
 
layout = dict(
        title = 'Visits per country',
        geo = dict(
            scope = 'world',
            projection = dict( type='natural earth' ),
            countrycolor = 'rgb(255, 255, 255)')
             )

figure = dict(data=data, layout=layout)
iplot(figure)
incrementovisitas = df.loc[(df['event']=='visited site') & (df['country']=='Brazil')].groupby(['country','timestamp'])['event'].agg(['count']).sort_values('count',ascending = False).reset_index()
incrementovisitas['just_date'] = incrementovisitas['timestamp'].dt.date
increm = incrementovisitas[['country','just_date','count']]
increm = increm.groupby(['country','just_date'])['count'].agg(['sum']).reset_index()
plt.figure(figsize=(20, 2))
plt.plot(increm['just_date'],increm['sum'])
incrementovisitas = df.loc[(df['event']=='visited site') & (df['country']=='Brazil') & (df['region']=='Sao Paulo')].groupby(['country','timestamp'])['event'].agg(['count']).sort_values('count',ascending = False).reset_index()
incrementovisitas['just_date'] = incrementovisitas['timestamp'].dt.date
increm = incrementovisitas[['country','just_date','count']]
increm = increm.groupby(['country','just_date'])['count'].agg(['sum']).reset_index()
plt.figure(figsize=(20, 2))
plt.plot(increm['just_date'],increm['sum'])
incrementovisitas = df.loc[(df['event']=='visited site') & (df['country']=='Brazil') & (df['region']=='Rio de Janeiro')].groupby(['country','timestamp'])['event'].agg(['count']).sort_values('count',ascending = False).reset_index()
incrementovisitas['just_date'] = incrementovisitas['timestamp'].dt.date
increm = incrementovisitas[['country','just_date','count']]
increm = increm.groupby(['country','just_date'])['count'].agg(['sum']).reset_index()
plt.figure(figsize=(20, 2))
plt.plot(increm['just_date'],increm['sum'])
incrementovisitas = df.loc[(df['event']=='visited site') ].groupby(['country','timestamp'])['event'].agg(['count']).sort_values('count',ascending = False).reset_index()
incrementovisitas['just_date'] = incrementovisitas['timestamp'].dt.date
increm = incrementovisitas[['country','just_date','count']]
increm = increm.groupby(['country','just_date'])['count'].agg(['sum']).reset_index()
increm
#plt.plot(increm['coutry',increm['sum']])