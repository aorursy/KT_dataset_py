# Alumnos:
#CALVO, MATEO IVÁN - 98290
#GUTIERREZ, MATÍAS - 92172
#PENNA, SEBASTIAN IGNACIO - 98752
#Link de GitHub: https://github.com/mateoicalvo/OrganizacionDeDatos
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.core.display import display, HTML
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.style.use('default')
display(HTML("<style>.container { width:90% !important; }</style>"))
import seaborn as sns
sns.set(style="whitegrid")
#plotly :
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
#Importamos los datos a un Data Frame
df = pd.read_csv('../input/events.csv',low_memory=False)
FIG_X, FIG_Y = 5, 5
#Verificamos si hay algún elemento nulo
#Hay elementos nulos
df.info()
#Realizamos la convesion de las fechas
df['timestamp']= df['timestamp'].astype('datetime64')
df['sku'] = pd.to_numeric(df['sku'], errors = 'coerce')
fechas_nulas = df['timestamp'].isnull().any()
print('Hay fechas nulas: {}'.format(fechas_nulas))
print('Fecha mínima: {}'.format(df['timestamp'].min()))
print('Fecha máxima: {}'.format(df['timestamp'].max()))
df['mes'] = [x for x in df['timestamp'].dt.month]
df['quincena'] = [(1 if x <= 15 else 2) for x in df['timestamp'].dt.day]
df['hora'] = [x for x in df['timestamp'].dt.hour]
df['dia'] = [x for x in df['timestamp'].dt.day]
df['semana'] = [x for x in df['timestamp'].dt.week]
df['dia_del_anio'] = [x for x in df['timestamp'].dt.dayofyear]
# Eventos por mes
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Eventos por mes')

df.groupby('mes').count()['event'].plot(kind="bar")

ax.set(xlabel='Mes', ylabel='Cantidad de eventos')
# Eventos por día del mes
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Eventos por dia del mes')

df.groupby('dia').count()['event'].plot(kind="bar")

ax.set(xlabel='Día del mes', ylabel='Cantidad de eventos')
# Eventos por hora
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Eventos por hora del dia')

df.groupby('hora').count()['event'].plot(kind="bar")

ax.set(xlabel='Hora', ylabel='Cantidad de eventos')
# Eventos por mes y quincena
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Eventos por quincena')

df.groupby(['mes', 'quincena']).count()['event'].plot(kind="bar")

ax.set(xlabel='(mes, quincena)', ylabel='Cantidad de eventos')
# Producto visto por quincena
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Productos vistos por quincena')

ax.set(xlabel='Quincena', ylabel='Cantidad de productos vistos')

productos_vistos = df.loc[df['event'] == 'viewed product',['mes','quincena','event']]
productos_vistos.groupby(['mes', 'quincena']).count()['event'].plot(kind="bar")
# Producto visto por quincena
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Brand listing por quincena')

ax.set(xlabel='Quincena', ylabel='Cantidad de listados')

listados = df.loc[df['event'] == 'brand listing',['mes','quincena','event']]
listados.groupby(['mes', 'quincena']).count()['event'].plot(kind="bar")
#Visitas por quincena
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Visitas por quincena')

ax.set(xlabel='Quincena', ylabel='Cantidad de visitas')

visitas = df.loc[df['event'] == 'visited site',['mes','quincena','event']]
visitas.groupby(['mes', 'quincena']).count()['event'].plot(kind="bar")
# Hits a capaña de marketing por quincena
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Ad campaign hit por quincena')

ax.set(xlabel='Quincena', ylabel='Cantidad de hits')

ad_hits = df.loc[df['event'] == 'ad campaign hit',['mes','quincena','event']]
ad_hits.groupby(['mes', 'quincena']).count()['event'].plot(kind="bar")

# Producto buscado por quincena
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Busquedas por quincena')

ax.set(xlabel='Quincena', ylabel='Cantidad de búsquedas')

busquedas = df.loc[df['event'] == 'searched products',['mes','quincena','event']]
busquedas.groupby(['mes', 'quincena']).count()['event'].plot(kind="bar")
# Checkouts por quincena
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Checkouts por quincena')

ax.set(xlabel='Quincena', ylabel='Cantidad de checkouts')

checkouts = df.loc[df['event'] == 'checkout',['mes','quincena','event']]
checkouts.groupby(['mes', 'quincena']).count()['event'].plot(kind="bar")
# Conversiones por quincena
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Conversiones por quincena')

ax.set(xlabel='Quincena', ylabel='Cantidad de conversiones')

conversiones = df.loc[df['event'] == 'conversion',['mes','quincena','event']]
conversiones.groupby(['mes', 'quincena']).count()['event'].plot(kind="bar")

# Leads por quincena
fig, ax = plt.subplots(figsize=(FIG_X,FIG_Y))

ax.set_title('Leads por quincena')

ax.set(xlabel='Quincena', ylabel='Cantidad de leads')

leads = df.loc[df['event'] == 'lead',['mes','quincena','event']]
leads.groupby(['mes', 'quincena']).count()['event'].plot(kind="bar")
x = df.loc[df['event'] == 'ad campaign hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'viewed product',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'searched products',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'viewed product',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'search engine hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'viewed product',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'ad campaign hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'brand listing',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'search engine hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'brand listing',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'ad campaign hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'visited site',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'ad campaign hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'visited site',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'ad campaign hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'generic listing',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'search engine hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'generic listing',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'visited site',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'searched products',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

plt.scatter(x,y)
x = df.loc[df['event'] == 'ad campaign hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'search engine hit',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)
x = df.loc[df['event'] == 'viewed product',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'checkout',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)
x = df.loc[df['event'] == 'visited site',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'checkout',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)
x = df.loc[df['event'] == 'ad campaign hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'checkout',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)
x = df.loc[df['event'] == 'viewed product',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'conversion',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)
x = df.loc[df['event'] == 'ad campaign hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'conversion',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)
x = df.loc[df['event'] == 'checkout',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'conversion',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)
x = df.loc[df['event'] == 'lead',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'conversion',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)
x = df.loc[df['event'] == 'visited site',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'lead',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)

x = df.loc[df['event'] == 'ad campaign hit',['dia_del_anio','event']]
x = x.groupby('dia_del_anio').count()['event']

y = df.loc[df['event'] == 'lead',['dia_del_anio','event']]
y = y.groupby('dia_del_anio').count()['event']


for j in x.index:
    if j not in y.index:
        y = y.append(pd.Series([0], index=[j]))

for k in y.index:
    if k not in x.index:
        x = x.append(pd.Series([0], index=[k]))
        
plt.scatter(x,y)

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
fig, ax = plt.subplots(figsize=(10, 3))
barlist1 = ax.bar(ploty, plotx,)
barlist1[0].set_color('g')
ax.text((barlist1[0].get_x()+(barlist1[0].get_width()/2))-0.1,barlist1[0].get_height()-10,'%d'%int(barlist1[0].get_height()) + '%', color = 'w')
labels = ax.get_xticklabels()
ax.set( xlabel='Pais', ylabel='%',
       title='Visitas por pais')
plt.tight_layout()
visitsperregion = df.loc[(df['event']=='visited site') & (df['region']!='Unknown') & (df['country']!='Brazil')].groupby(['country','region'])['event'].value_counts()
visitsperregion.sort_values(ascending = False,inplace= True)
visitsperregion= visitsperregion.sort_values(ascending = False).reset_index(drop = True, level = 2).reset_index()
fig, ax = plt.subplots( figsize=(10, 12),)
plt.axvline(visitsperregion.max().event,color = 'r',ls='--',)
plt.text(visitsperregion.max().event,-3,'Maximo fuera de Brasil: ' + '%d'%int(visitsperregion.max().event),rotation=90,color ='r')
#ax.barh(visitsperregion['region'], visitsperregion['event'])
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.set(xlim=[0, visitsperregion.max().event+50], xlabel='Total de visitas', ylabel='Region',
       title='Visitas por region fuera de Brasil')
visitsperregion = df.loc[(df['event']=='visited site') & (df['region']!='Unknown') & (df['country']=='Brazil')].groupby(['country','region'])['event'].value_counts()
visitsperregion.sort_values(ascending = False,inplace= True)
visitsperregion= visitsperregion.sort_values(ascending = False).reset_index(drop = True, level = 2).reset_index()
ax.barh(visitsperregion['region'], visitsperregion['event'])
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
ax.set(xlim=[0, visitsperregion.max().event+10000], xlabel='Total de visitas', ylabel='Region',
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
canalincrementovisitas = df.loc[(df['event']=='visited site')].groupby(['channel','timestamp'])['channel'].agg(['count']).sort_values('count',ascending = False).reset_index()
canalincrementovisitas['just_date'] = incrementovisitas['timestamp'].dt.date
increm = canalincrementovisitas[['channel','just_date','count']]
increm1 = increm.loc[increm['channel']=='Direct'].groupby(['channel','just_date'])['count'].agg(['sum']).reset_index()
increm2 = increm.loc[increm['channel']=='Paid'].groupby(['channel','just_date'])['count'].agg(['sum']).reset_index()
increm3 = increm.loc[increm['channel']=='Email'].groupby(['channel','just_date'])['count'].agg(['sum']).reset_index()
increm4 = increm.loc[increm['channel']=='Organic'].groupby(['channel','just_date'])['count'].agg(['sum']).reset_index()
increm5 = increm.loc[increm['channel']=='Referral'].groupby(['channel','just_date'])['count'].agg(['sum']).reset_index()
increm6 = increm.loc[increm['channel']=='Social'].groupby(['channel','just_date'])['count'].agg(['sum']).reset_index()
increm7 = increm.loc[increm['channel']=='Unknown'].groupby(['channel','just_date'])['count'].agg(['sum']).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.set( xlabel='Fecha de visita', ylabel='Cantidad de visitas',
       title='Visitas por Canal en el periodo')
plt.tight_layout()
plt.plot(increm1['just_date'],increm1['sum'],label='Direct')
plt.plot(increm2['just_date'],increm2['sum'],label ='Paid')
plt.plot(increm3['just_date'],increm3['sum'], label ='Email')
plt.plot(increm4['just_date'],increm4['sum'], label ='Organic')
plt.plot(increm5['just_date'],increm5['sum'], label ='Referral')
plt.plot(increm6['just_date'],increm6['sum'], label ='Social')
plt.plot(increm7['just_date'],increm7['sum'], label ='Unknown')
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.legend()
visitsplatform = df.groupby(['device_type'])['event'].value_counts().reset_index(drop= True, level = 1)
visitsplatform.sort_values(ascending = False,inplace= True)
visitsplatform = visitsplatform.to_frame().reset_index()
visitsplatform
fig, ax = plt.subplots(1,2,figsize=(10, 2))
ax[0].barh(visitsplatform['device_type'],visitsplatform['event'], color = 'rgb',edgecolor='black')
ax[0].set(xlabel='Total de visitas', ylabel='Dispositivo', title='Dispositivos usados para ingresar al sitio')
ax[1].barh(visitsplatform['device_type'],visitsplatform['event']/visitsplatform['event'].sum()*100, color = 'rgb',edgecolor='black')
ax[1].set(xlabel='% Total de visitas', ylabel='Dispositivo', title='Dispositivos usados para ingresar al sitio')
plt.tight_layout()
visitsos= df.groupby(['device_type','operating_system_version'])['event'].value_counts()
visitsos = visitsos.reset_index(drop = True, level = 2).reset_index()
computeros = visitsos.loc[visitsos['device_type']=='Computer']
tabletos = visitsos.loc[visitsos['device_type']=='Tablet']
smartphoneos = visitsos.loc[visitsos['device_type']=='Smartphone']
OS = computeros.groupby(['operating_system_version'])['event'].sum()
tOS = tabletos.groupby(['operating_system_version'])['event'].sum()
sOS = smartphoneos.groupby(['operating_system_version'])['event'].sum()

Linux = OS.filter(like = 'Linux').sum()  + OS.filter(like = 'Ubuntu').sum()
Mac = OS.filter(like = 'Mac').sum()
Windows = OS.filter(like = 'Windows').sum()

oses = pd.DataFrame(data = {'cant' :[Linux, Mac, Windows]}, index = ['Linux', 'Mac', 'Windows'])

oses
fig, ax = plt.subplots(1,2,figsize=(10, 2))
ax[0].barh(oses.index,oses['cant'], color = 'rgb',edgecolor='black')
ax[0].set(xlabel='Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (computadora)')
ax[1].barh(oses.index,oses['cant']/oses['cant'].sum()*100, color = 'rgb',edgecolor='black')
ax[1].set(xlabel='% Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (computadora)')
plt.tight_layout()
iOS = tOS.filter(like = 'iOS').sum() 
Android = tOS.filter(like = 'Android').sum()
toses = pd.DataFrame(data = {'cant' :[iOS, Android]}, index = ['iOS', 'Android'])
toses
fig, ax = plt.subplots(1,2,figsize=(10, 2))
ax[0].barh(toses.index,toses['cant'], color = 'rgb',edgecolor='black')
ax[0].set(xlabel='Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (tablet)')
ax[1].barh(toses.index,toses['cant']/toses['cant'].sum()*100, color = 'rgb',edgecolor='black')
ax[1].set(xlabel='% Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (tablet)')
plt.tight_layout()
sOS.reset_index()['operating_system_version'].unique()
Android = sOS.filter(like = 'Android').sum()  + OS.filter(like = 'Ubuntu').sum()
iOS = sOS.filter(like = 'iOS').sum()
Windows = sOS.filter(like = 'Windows').sum()
Other = sOS.filter(like = 'Other').sum()

soses = pd.DataFrame(data = {'cant' :[Android, iOS, Windows, Other]}, index = ['Android', 'iOS', 'Windows','Other'])

soses
fig, ax = plt.subplots(1,2,figsize=(10, 2))
ax[0].barh(soses.index,soses['cant'], color = 'rgb',edgecolor='black')
ax[0].set(xlabel='Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (smartphone)')
ax[1].barh(soses.index,soses['cant']/soses['cant'].sum()*100, color = 'rgb',edgecolor='black')
ax[1].set(xlabel='% Total de visitas', ylabel='Sistema Operativo', title='Sistemas Operativo usados (smartphone)')
plt.tight_layout()
visitsbrowser= df.groupby(['browser_version'])['event'].value_counts()
visitsbrowser.sort_values(ascending = False,inplace= True)
visitsbrowser.to_frame().reset_index(drop=True,level = 1).reset_index().head(10)
visitsscreen= df.groupby(['screen_resolution','device_type' ])['event'].value_counts()
visitsscreen.sort_values(ascending = False,inplace= True)
screens = visitsscreen.to_frame().reset_index(drop=True, level = 2).reset_index()
screens.loc[screens['device_type']=='Computer'].head(10).reset_index(drop = True)
screens.loc[screens['device_type']=='Smartphone'].head(10).reset_index(drop = True)
screens.loc[screens['device_type']=='Tablet'].head(10).reset_index(drop = True)
# Agrupados por modelos vendidos y sus cantidades
modelSales = df.loc[(df['event'] == 'conversion')].groupby('model').agg('size').sort_values(ascending = False)
modelSales.head()
plt.rcParams['figure.figsize'] = (10, 6)
# Se consideran los modelos con más de 10 ventas
topSales = modelSales.iloc[0:36]
gTS = sns.barplot(x = topSales , y = topSales.index , orient='h',palette = (sns.color_palette("gist_stern", 20)))
gTS.set_title("Modelos más vendidos", fontsize = 20)
gTS.set_xlabel("Ventas", fontsize = 15)
gTS.set_ylabel(" Modelo", fontsize = 15)
# Filtro los datos sobre cada marca
iPhone = modelSales.filter(like = 'iPhone')
samsung = modelSales.filter(like = 'Samsung')
motorola = modelSales.filter(like = 'Motorola')
lg = modelSales.filter(like = 'LG')
asus = modelSales.filter(like = 'Asus')
sony = modelSales.filter(like = 'Sony')
lenovo = modelSales.filter(like = 'Lenovo')
# Creo DataFrame con los datos obtenidos
d = {'amount' : [samsung.sum(),iPhone.sum(), motorola.sum(), sony.sum(), lg.sum(), lenovo.sum(), asus.sum()]}
brands = pd.DataFrame(data = d, index = ['Samsung', 'iPhone', 'Motorola', 'Sony', 'LG', 'Lenovo', 'Asus'])
brands.head(10)
# Grafico sobre las ventas por marca
gMarcas = sns.barplot(x = brands.index , y = brands.amount , orient='v', palette = (sns.color_palette("Set1", 7)))
gMarcas.set_title("Ventas por marca", fontsize = 20)
gMarcas.set_xlabel("Marca", fontsize = 15)
gMarcas.set_ylabel("Ventas", fontsize = 15)
# Funcion que analiza un string pasado como parametro y determina la marca segun el nombre del producto
def brandName(model):
    if 'iPhone' in model:
        return 'iPhone'
    if 'Samsung' in model:
        return 'Samsung'
    if 'Sony' in model:
        return 'Sony'
    if 'Lenovo' in model:
        return 'Lenovo'
    if 'Asus' in model:
        return 'Asus'
    if 'Motorola' in model:
        return 'Motorola'
    if 'LG' in model:
        return 'LG'
# Informacion detallada de los productos vendidos
sales = df.loc[df.event=='conversion'].loc[:,('model','condition','storage','color')]
sales['brand'] = sales.model.apply(brandName)
sales.head()
# Se organiza la cantidad de ventas segun la condicion para cada Marca
conditions = pd.pivot_table(sales.loc[:,('condition','brand')].set_index('brand'), index = ['brand'], columns = ['condition'], aggfunc = pd.Series.count).fillna('0')
conditions.columns = ['Bueno','Bueno - Sin Touch Id', 'Excelente', 'Muy Bueno', 'Nuevo']
conditions = conditions.loc[('Samsung','iPhone','Motorola','Sony','LG','Lenovo','Asus'),:]
for col in conditions.columns:
    conditions[col] = pd.to_numeric(conditions[col])
conditions
# Elimino las condicones Bueno - Sin Touch Id y Nuevo al no presentar un gran efecto en las ventas
conditions = conditions.loc[:,('Bueno','Excelente','Muy Bueno')]
conditions
# Sabiendo cuanto vendio cada marca se analiza la condicion de los productos en dichas ventas
conditions.plot(kind = 'bar', stacked = False, colormap = 'tab10', width = 0.8, grid = False)
plt.legend(loc = 'center right', bbox_to_anchor = (1.2, 0.5))
plt.xlabel('Marca', fontsize = 15)
plt.ylabel('Ventas', fontsize = 15)
plt.title('Condición de producto vendido según marca', fontsize=20)
plt.xticks(rotation = 40)
storageSales = sales.groupby('storage').size().to_frame(name='amount')
storageSales = storageSales.loc[('512MB','4GB','8GB','16GB','32GB','64GB','128GB','256GB'),:]
storageSales.amount = pd.to_numeric(storageSales.amount)
storageSales.head(10)
gStS = sns.barplot(x = storageSales.index, y = storageSales['amount'], data = storageSales)
gStS.set_title("Capacidad del producto según ventas", fontsize = 20)
gStS.set_xlabel("Capacidad", fontsize = 15)
gStS.set_ylabel("Ventas", fontsize = 15)
# Funcion que convierte el valor string del almacenaje en numerico (medido en GB)
def absMemValue(x):
    switcher = {
        '4GB': 4,
        '512MB' : 0.512,
        '8GB' : 8,
        '16GB' : 16,
        '32GB' : 32,
        '64GB' : 64,
        '128GB' : 128,
        '256GB' : 256
    }
    return switcher.get(x)
# Cada produto vendido con su marca y capacidad
storageBrand = sales.loc[:,('brand','storage')]
storageBrand.storage = storageBrand.storage.apply(absMemValue)
storageBrand.head(5)
sns.boxplot(x=storageBrand.brand, y=storageBrand.storage, data=storageBrand, width=0.9, fliersize = 8)
plt.title('Capacidad de los productos vendidos según marca', fontsize = 20)
plt.xlabel('Marca', fontsize = 15)
plt.ylabel('Capacidad (GB)', fontsize = 15)
# Filtro el iPhone de 256 gb para analizar mejor cada box
sns.boxplot(x=storageBrand.loc[storageBrand.storage<200].brand, y=storageBrand.loc[storageBrand.storage<200].storage, data=storageBrand, width=0.9, fliersize = 8)
plt.title('Capacidad de los productos vendidos según marca', fontsize = 20)
plt.xlabel('Marca', fontsize = 15)
plt.ylabel('Capacidad', fontsize = 15)
# Relacion entre cantidad visitas a pagina producto (event=viewed product) y cantidad conversiones
cantViewedProduct = df.loc[df.event=='viewed product'].groupby('person').size().apply('to_frame').reset_index()
cantViewedProduct.columns = ['person','viewed products']
cantConversion =  df.loc[df.event=='conversion'].groupby('person').size().apply('to_frame').reset_index()
cantConversion.columns = ['person','conversions']
cantConversion['vp'] = cantViewedProduct['viewed products']
cantConversion.head()
plt.scatter(x=cantConversion.conversions,y=cantConversion.vp)
plt.title('Relacion conversiones y visitas a productos por usuario', fontsize = 20)
plt.xlabel('Conversiones', fontsize = 15)
plt.ylabel('Visitas a productos', fontsize = 15)
# Me quedo sólo con los términos de búsqueda
busquedas = df.loc[df['event'] == 'searched products']['search_term']
#busquedas.str.split(expand=True).stack().value_counts()

# Todo a mayúscula
busquedas = busquedas.str.upper()
frecuencias = busquedas.str.split(expand=True).stack().value_counts().reset_index().values

#Solo se consideran las de longitud mayor que 4
diccionario = dict([tuple(x) for x in frecuencias if (len(x[0]) > 4)])

wordcloud = WordCloud(background_color="white", width=500, height=500, prefer_horizontal=0.9).generate_from_frequencies(frequencies=diccionario)

def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(230,100%%, %d%%)" % np.random.randint(49,51))


wordcloud.recolor(color_func = grey_color_func)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("first_review.png")
# Me quedo sólo con los términos de búsqueda
busquedas = df.loc[df['event'] == 'searched products']['search_term']
busquedas.str.split(expand=True).stack().value_counts()

# Todo a mayúscula
busquedas = busquedas.str.upper()
frecuencias = busquedas.str.split(expand=True).stack().value_counts().reset_index()
frecuencias.columns = ['palabra','cantidad']
frecuencias['longitud_palabra'] = frecuencias.agg({'palabra' : len})
frecuencias = frecuencias.set_index('palabra')
frecuencias.loc[frecuencias['longitud_palabra'] > 4,:].head(15)['cantidad'].plot(kind='bar')

#frecuencias.head(15).plot(kind='bar')
#Solo se consideran las de longitud mayor que 4
# AHORA Sólo por termino de busqueda
# Me quedo sólo con los términos de búsqueda
busquedas = df.loc[df['event'] == 'searched products']['search_term']
#busquedas.str.split(expand=True).stack().value_counts()

# Todo a mayúscula
busquedas = busquedas.str.upper()
frecuencias = busquedas.value_counts().reset_index().values

diccionario = dict([tuple(x) for x in frecuencias if (len(x[0]) > 4)])

wordcloud = WordCloud(background_color="white", width=500, height=500).generate_from_frequencies(frequencies=diccionario)

#def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
#    return("hsl(230,100%%, %d%%)" % np.random.randint(49,51))


#wordcloud.recolor(color_func = grey_color_func)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("first_review.png")
# Me quedo sólo con los términos de búsqueda
busquedas = df.loc[df['event'] == 'searched products']['search_term']
# Todo a mayúscula
busquedas = busquedas.str.upper()
frecuencias = busquedas.str.split(expand=True).stack().value_counts().reset_index()
frecuencias.columns = ['palabra','cantidad']
frecuencias['longitud_palabra'] = frecuencias.agg({'palabra' : len})
frecuencias = frecuencias.set_index('palabra')
frecuencias.loc[frecuencias['longitud_palabra'] > 3,:].head(20)['cantidad'].plot(kind='bar')
# Me quedo sólo con los términos de búsqueda
busquedas = df.loc[df['event'] == 'searched products']['search_term']
# Todo a mayúscula
busquedas = busquedas.str.upper()
frecuencias = busquedas.str.split(expand=True).stack().value_counts().reset_index()
frecuencias.columns = ['palabra','cantidad']
frecuencias['longitud_palabra'] = frecuencias.agg({'palabra' : len})
frecuencias = frecuencias.set_index('palabra')
frecuencias.loc[frecuencias['longitud_palabra'] < 4,:].head(20)['cantidad'].plot(kind='bar')
# Me quedo sólo con los términos de búsqueda
busquedas = df.loc[df['event'] == 'searched products']['search_term']
# Todo a mayúscula
busquedas = busquedas.str.upper()
busquedas = busquedas.str.split(expand=True).stack().value_counts().reset_index()
busquedas.columns = ['palabra', 'cantidad']
busquedas = busquedas.loc[busquedas['palabra'].str.contains('GB'),:]
busquedas['longitud_palabra'] = busquedas.agg({'palabra' : len})
busquedas = busquedas.loc[busquedas['longitud_palabra'] > 2,:]
busquedas.set_index('palabra')['cantidad'].nlargest(10).plot(kind='bar')
# Me quedo sólo con los términos de búsqueda
busquedas = df.loc[df['event'] == 'searched products']['search_term']
# Todo a mayúscula
busquedas = busquedas.str.upper()
busquedas.value_counts().nlargest(20).plot(kind='bar')
len(df['person'].value_counts())

df['channel'].value_counts().plot(kind='bar')
df[['event', 'channel']].dropna()['event'].value_counts()
df[['person','channel']].groupby('channel').agg('count').plot(kind='bar')
nulos = len(df) - df['channel'].count()
print('Hay {} eventos que no tienen informacion de channel'.format(nulos))
df.set_index('timestamp').groupby(['person', pd.Grouper(freq='240Min')])
# Funcion que convierte el valor numerico del dia de la semana en su valor string
def toDay (x):
        switch = {
            0 : 'Lunes',
            1 : 'Martes',
            2 : 'Miércoles',
            3 : 'Jueves',
            4 : 'Viernes',
            5 : 'Sábado',
            6 : 'Domingo'
        }
        return switch.get(x)
datesEvent = pd.DataFrame(data = {'day' : df['timestamp'].dt.dayofweek, 'event' : df['event']})
datesEvent = datesEvent.sort_values('day').set_index('day')
datesEvent = pd.pivot_table(datesEvent, index = ['day'], columns = ['event'], aggfunc = pd.Series.count)
datesEvent.index = datesEvent.index.map(toDay)
datesEvent
plt.rcParams['figure.figsize'] = (10, 6)
gDE = sns.heatmap(datesEvent, cmap='Spectral', annot = True, fmt= 'd')
gDE.set_title("Eventos por día", fontsize = 20)
gDE.set_xlabel("Tipo de evento", fontsize = 15)
gDE.set_ylabel("Día", fontsize = 15)
plt.yticks(rotation=0)
plt.xticks(rotation=45)
origenes = df.loc[df['new_vs_returning'] == 'New',['person','channel']].set_index('person')
origenes.columns = ['canal_origen']
por_persona = df.set_index('person')
por_persona = por_persona.merge(origenes, on='person', how='left')
por_persona.head()
#por_persona['event'].value_counts().plot(kind='bar', hue='canal_origen')
#sns.barplot(x="event", hue="canal_origen", data=por_persona);
#sns.catplot(x="event", kind="count", hue="canal_origen", data=por_persona);
f, ax = plt.subplots(figsize=(15, 15))
sns.countplot(y="event", data=por_persona, hue="canal_origen");
f, ax = plt.subplots(figsize=(15, 15))
sns.countplot(y="canal_origen", data=por_persona, hue="event");
f, ax = plt.subplots(figsize=(10, 10))
sns.countplot(y="canal_origen", data=por_persona.loc[por_persona['event'] == 'conversion']);
srchTrms = df.groupby('search_term').size().sort_values(ascending=False).to_frame('hits')
# Se convierten todos los caracteres a minusculas de manera de poder agruparlos 
# independientemente del formato en que lo haya escrito el usuario en su motor de busqueda
srchTrms.index = srchTrms.index.str.lower()
srchTrms = srchTrms.reset_index()
srchTrms = srchTrms.groupby('search_term').agg({'hits':'sum'}).sort_values('hits',ascending=False)
srchTrms.head()
# 200 o más búsquedas
topSrchTrms = srchTrms.iloc[0:41]
gTST = sns.barplot(x = topSrchTrms.hits , y = topSrchTrms.index , orient='h',palette = (sns.color_palette("winter",20)))
gTST.set_title("Términos de búsqueda más utilizados", fontsize = 20)
gTST.set_xlabel("Búsquedas", fontsize = 15)
gTST.set_ylabel("Término", fontsize = 15)