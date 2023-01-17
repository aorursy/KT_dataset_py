import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.core.display import display, HTML
plt.style.use('default')
display(HTML("<style>.container { width:90% !important; }</style>"))
'''plotly :'''
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
df = pd.read_csv('../input/events.csv',low_memory=False)
#convieto las fechas a formato fecha 
df['timestamp']= df['timestamp'].astype('datetime64')
visitspercountry = df.loc[(df['event']=='visited site') & (df['country']!='Unknown')].groupby(['country'])['event'].agg(['count']).sort_values('count',ascending = False).reset_index()
visitspercountry.head(10)
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
        title = 'Numero de Visitas por Pais',
        geo = dict(
            scope = 'world',
            projection = dict( type='natural earth' ),
            countrycolor = 'rgb(255, 255, 255)')
             )

figure = dict(data=data, layout=layout)
iplot(figure)
plotx = visitspercountry['count'].head(5)/len(df.loc[(df['event']=='visited site') ,'event'])*100
ploty = visitspercountry['country'].head(5)
#calculo lo mismo sacando a Brasil
visitspercountry = df.loc[(df['event']=='visited site') & (df['country']!='Brazil')  & (df['country']!='Unknown')].groupby(['country'])['event'].agg(['count']).sort_values('count',ascending = False).reset_index()
plotxb = visitspercountry['count'].head(5)/len(df.loc[(df['event']=='visited site') ,'event'])*100
plotyb = visitspercountry['country'].head(5)
fig, ax = plt.subplots(2,1, figsize=(10, 6),)
barlist1 = ax[0].bar(ploty, plotx,)
barlist1[0].set_color('g')
ax[0].text((barlist1[0].get_x()+(barlist1[0].get_width()/2))-0.1,barlist1[0].get_height()-10,'%d'%int(barlist1[0].get_height()) + '%', color = 'w')
labels = ax[0].get_xticklabels()
ax[0].set( xlabel='Pais', ylabel='%',
       title='Visitas por pais')
barlist2 = ax[1].bar(plotyb, plotxb,)
barlist2[0].set_color('g')
labels = ax[1].get_xticklabels()
ax[1].set( xlabel='Pais', ylabel='%',
       title='Visitas por pais fuera de Brasil')
plt.tight_layout()


visitsperregion = df.loc[(df['event']=='visited site') & (df['region']!='Unknown') & (df['country']!='Brazil')].groupby(['country','region'])['event'].value_counts()
visitsperregion.sort_values(ascending = False,inplace= True)
visitsperregion= visitsperregion.sort_values(ascending = False).reset_index(drop = True, level = 2).reset_index()
fig, ax = plt.subplots(1,2, figsize=(10, 12),)
plt.axvline(visitsperregion.max().event,color = 'r',ls='--',)
plt.text(visitsperregion.max().event,-3,'Maximo fuera de Brasil: ' + '%d'%int(visitsperregion.max().event),rotation=90,color ='r')
ax[0].barh(visitsperregion['region'], visitsperregion['event'])
labels = ax[0].get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax[0].set(xlim=[0, visitsperregion.max().event+50], xlabel='Total de visitas', ylabel='Region',
       title='Visitas por region fuera de Brasil')
visitsperregion = df.loc[(df['event']=='visited site') & (df['region']!='Unknown') & (df['country']=='Brazil')].groupby(['country','region'])['event'].value_counts()
visitsperregion.sort_values(ascending = False,inplace= True)
visitsperregion= visitsperregion.sort_values(ascending = False).reset_index(drop = True, level = 2).reset_index()
ax[1].barh(visitsperregion['region'], visitsperregion['event'])
labels = ax[1].get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax[1].set(xlim=[0, visitsperregion.max().event+10000], xlabel='Total de visitas', ylabel='Region',
       title='Visitas por region en Brasil')
plt.tight_layout()


visitscity= df.loc[(df['event']=='visited site') & (df['region']!='Unknown')].groupby(['region','city'])['event'].value_counts().reset_index(drop = True , level = 2)
visitscity.sort_values(ascending = False,inplace= True)
visitscity.to_frame().reset_index().head(10)
visitsperregion = df.loc[(df['event']=='visited site') & (df['region']!='Unknown') & (df['country']=='Brazil')].groupby(['region','city'])['event'].value_counts()
visitsperregion.sort_values(ascending = False,inplace= True)
visitsperregion= visitsperregion.sort_values(ascending = False).head(10).reset_index(drop = True, level = 2).reset_index()
fig, ax = plt.subplots(figsize=(4, 3))
ax.barh(visitsperregion['city'],visitsperregion['event'],edgecolor='black' )
ax.set(xlim=[0, visitsperregion.max().event+1000], xlabel='Total de visitas', ylabel='Ciudad',
       title='Visitas por ciudad en Brasil')
plt.tight_layout()
incrementovisitas = df.loc[(df['event']=='visited site')].groupby(['region','timestamp'])['event'].agg(['count']).sort_values('count',ascending = False).reset_index()
incrementovisitas['just_date'] = incrementovisitas['timestamp'].dt.date
increm = incrementovisitas[['region','just_date','count']]
increm1 = increm.loc[increm['region']=='Sao Paulo'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
increm2 = increm.loc[increm['region']=='Minas Gerais'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
increm3 = increm.loc[increm['region']=='Rio de Janeiro'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
increm4 = increm.loc[increm['region']=='Bahia'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
increm5 = increm.loc[increm['region']=='Pernambuco'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
increm6 = increm.loc[increm['region']=='Ceara'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
increm7 = increm.loc[increm['region']=='Parana'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
increm8 = increm.loc[increm['region']=='Rio Grande do Sul'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
increm9 = increm.loc[increm['region']=='Federal District'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
increm10 = increm.loc[increm['region']=='Goias'].groupby(['region','just_date'])['count'].agg(['sum']).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.set( xlabel='Fecha de visita', ylabel='Cantidad de visitas',
       title='Visitas por Region en el periodo')
plt.tight_layout()
plt.plot(increm1['just_date'],increm1['sum'],label='Sao Paulo')
plt.plot(increm2['just_date'],increm2['sum'],label ='Minas Gerais')
plt.plot(increm3['just_date'],increm3['sum'], label ='Rio de Janeiro')
plt.plot(increm4['just_date'],increm4['sum'], label ='Bahia')
plt.plot(increm5['just_date'],increm5['sum'], label ='Pernambuco')
plt.plot(increm6['just_date'],increm6['sum'], label ='Ceara')
plt.plot(increm7['just_date'],increm7['sum'], label ='Parana')
plt.plot(increm8['just_date'],increm8['sum'], label ='Rio Grande do Sul')
plt.plot(increm9['just_date'],increm9['sum'], label ='Federal District')
plt.plot(increm10['just_date'],increm10['sum'], label ='Goias')
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.legend()