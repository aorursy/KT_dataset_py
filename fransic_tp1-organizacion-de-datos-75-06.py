#Import libraries
import pandas as pd
import numpy as np
#Plots
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Load the Data and take a quick look
trocafone = pd.read_csv('../input/events.csv', low_memory=False)
trocafone.tail()
# Information about the dataset
trocafone.info()
# Some stats about the numeric columns in our dataset
trocafone.describe()
events_vc=trocafone['event'].value_counts()
events_vc
# Convert Date
import calendar
trocafone['timestamp'] = pd.to_datetime(trocafone['timestamp'])
trocafone['Year']=trocafone['timestamp'].map(lambda x:x.year)
trocafone['Month'] = trocafone['timestamp'].dt.month_name()
trocafone['day_of_week'] = trocafone['timestamp'].dt.weekday_name
trocafone['day'] = trocafone['timestamp'].map(lambda x: x.day)
trocafone['Hour'] = pd.to_datetime(trocafone['timestamp'], format='%H:%M',errors='coerce').dt.hour
eventos_por_hora = trocafone.groupby(['Hour','event']).size().unstack()

eventos_por_hora.plot(kind='bar',stacked=True,figsize=(24,10),rot=0).legend(bbox_to_anchor=(0.4, 1.00),prop={'size': 15})

plt.title("Tipos de eventos por hora",fontsize='22')
plt.xlabel('Hora',fontsize='22')
plt.ylabel('Eventos',fontsize='22')
plt.xticks(size = 15)
plt.yticks(size = 15)
eventos_por_hora
filter_viewed_product = trocafone.loc[trocafone['event'] != 'viewed product']\
                                .groupby(['Hour','event']).size().unstack()

#Defino las filas cuyo evento es viewed product 
fig, ax = plt.subplots(figsize=(18,6))
labels = filter_viewed_product.columns
ax.stackplot(filter_viewed_product.index , [filter_viewed_product[column] for column in filter_viewed_product.columns], labels=labels)
plt.legend(loc= (0.25,0.3) , prop={'size': 12})
ax.xaxis.set_ticks(filter_viewed_product.index)
plt.xlabel('Hora del día',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 12)
plt.yticks(size = 12)

plt.show()
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(eventos_por_hora['visited site'] , 'lightblue' , linestyle = 'dashed' , linewidth = 3)
plt.plot(eventos_por_hora['brand listing'] , 'orange' ,linestyle = '-.' , linewidth = 3)
plt.plot(eventos_por_hora['searched products'] , 'go')
plt.plot(eventos_por_hora['generic listing'] , 'darkviolet',linestyle='-.',linewidth = 3)
plt.legend(prop={'size': 12})
ax.xaxis.set_ticks(eventos_por_hora.index)
plt.xlabel('Hora del día',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.grid()
trocafone["day"].value_counts().sort_index().plot(kind='bar',figsize=(24,10),rot=0 )

plt.title("Cantidad de eventos a lo largo del mes",fontsize='22')
plt.xlabel('Día del mes',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
trocafone['nonNaN'] = trocafone['event'].map(lambda x : 1) #Defino una columna para poder sumar

# llevamos a una representacion de ese tipo usando una tabla pivot.
for_heatmap = trocafone.pivot_table(index='Hour', columns='day' , values = 'nonNaN' , aggfunc = 'sum')
dims = (11, 9)
fig, ax = plt.subplots(figsize=dims)
g = sns.heatmap(for_heatmap.T , cmap="YlGnBu")
g.set_title("Cantidad de eventos (Primer semestre 2018)", fontsize=20)
g.set_xlabel("Hora del día",fontsize=13)
g.set_ylabel("Día del mes", fontsize=13)
#Create a column with days of the week
trocafone['day_of_week'] = pd.Categorical(trocafone['day_of_week'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'], ordered=True)
events_by_day = trocafone['day_of_week'].value_counts().sort_index().plot(kind='bar',figsize=(24,10),rot=0)

plt.title("Cantidad de eventos por día de la semana",fontsize='22')
plt.xlabel('Día de la semana',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15)
df_week=trocafone['day_of_week'].value_counts().to_frame()
promedio_semana=df_week.iloc[0:5].mean()
promedio_fin_semana=df_week.iloc[6:8].mean()
diferencia=promedio_semana-promedio_fin_semana
print("El promedio de visitas de lunes a viernes es",promedio_semana.values[0],", mientras que los fines de semana es",promedio_fin_semana.values[0])
trocafone.loc[trocafone['event']=='conversion']['day_of_week'].value_counts().sort_index().plot(kind='bar',figsize=(24,10),rot=0)

plt.title("Cantidad de compras por día de la semana",fontsize='22')
plt.xlabel('Día de la semana',fontsize=20)
plt.ylabel('Compras',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
trocafone.sort_values(by='timestamp')['timestamp'].hist(figsize=(24,10),bins=166)

plt.title("Cantidad de eventos a lo largo del año",fontsize='22')
plt.xlabel('Fecha',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
trocafone.loc[trocafone["event"] == "conversion"].sort_values(by='timestamp')['timestamp']\
                                                 .hist(figsize=(24,10),bins=166,color='green')

plt.title("Cantidad de compras a lo largo del año",fontsize='22')
plt.xlabel('Fecha',fontsize=20)
plt.ylabel('Compras',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
trocafone['Month'] = pd.Categorical(trocafone['Month'], categories=['January','February','March','April','May','June'], ordered=True)
df=trocafone.groupby('Month').size().to_frame()
df['compras']=trocafone.loc[trocafone['event']=='conversion'].groupby('Month').size()
df.columns = ['total', 'compras']
df['relacion']=df['compras']/df['total']

df['relacion'].plot(kind='bar',figsize=(24,10),rot=0)
plt.title("Relacion entre eventos y compras",fontsize='22')
plt.xlabel('Mes',fontsize=20)
plt.ylabel('Compras/Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
trocafone.loc[(trocafone['event']=='conversion') | (trocafone['event']=='lead')]\
        .groupby(['Month','event']).size().unstack()\
        .plot(kind='bar',figsize=(24,10),rot=0).legend(prop={'size': 22})

plt.title("Relacion compras/subscripciones por mes",fontsize='22')
plt.xlabel('Mes',fontsize='22')
plt.ylabel('Compras/Subscripciones',fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.grid(axis='y')
primeros_ingresos = trocafone.groupby(['person'])['timestamp'].min().to_frame()
primeros_ingresos['timestamp'] = primeros_ingresos['timestamp'].map(lambda x: x.date())
primeros_ingresos_por_dia = primeros_ingresos.groupby('timestamp').agg('size')
primeros_ingresos_por_dia.plot(figsize=(24,10) , linewidth = 4)
plt.title("Ingreso de usuarios nuevos",fontsize='22')
plt.xlabel('Tiempo',fontsize=20)
plt.ylabel('Cantidad de usuarios',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
plt.grid()
eventos_por_persona = trocafone.groupby('person')['event'].count()
max = eventos_por_persona.max()
min = eventos_por_persona.min()
promedio = eventos_por_persona.mean()
media = eventos_por_persona.median()
std = eventos_por_persona.std()
d = {"promedio":promedio, "media":media, "std":std, "max":max, "min":min}
pd.DataFrame(data=d, index=["eventos por persona"])
f = (lambda n : eventos_por_persona[eventos_por_persona < n]) 
f(200).value_counts().sort_index()\
        .plot(color='r', figsize=(20,8) ,linewidth = 4,  label ='Usuarios por cantidad de eventos')

plt.title("Eventos realizados por usuarios",fontsize='22')
plt.ylabel('Cantidad de usuarios',fontsize=18)
plt.xlabel('Cantidad de eventos realizados',fontsize=18)
plt.legend(prop={'size': 15})
plt.xticks(size = 15)
plt.yticks(size = 15) 
plt.grid()
f = (lambda n : eventos_por_persona[eventos_por_persona < n])
s1 , s2 = f(75) , f(30)
fig, axs = plt.subplots(nrows=2 , figsize=(6,6))
plt.tight_layout(h_pad=3.0)
g1 =sns.violinplot(s1.values , orient = 'v' , ax=axs[0] , palette = 'Greens')
g2 =sns.violinplot(s2.values , orient = 'v' , ax=axs[1] , palette = 'Greens')

g1.set_title("Personas con menos de 75 eventos ", fontsize=15)
g1.set_ylabel("Eventos",fontsize=13)
g1.set_xlabel("Cantidad de personas", fontsize=13)

g2.set_title("Personas con menos de 30 eventos ", fontsize=15)
g2.set_ylabel("Eventos",fontsize=13)
g2.set_xlabel("Cantidad de personas", fontsize=13)
fig, ax = plt.subplots(figsize=(18, 14))
ax = sns.barplot(x=events_vc.values, y=events_vc.index , palette = 'rocket')
ax.set_title("Eventos generados por usuarios", fontsize=22)
ax.set_xlabel("Frecuencia absoluta", fontsize=20)
ax.set_ylabel("Evento", fontsize=20)
ax.tick_params(labelsize=16)
fig.tight_layout()


rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # Use X value as label and format number with one decimal place
    label = x_value.astype(int)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)         
#Top Models 
grouped = trocafone.groupby('event')
compras = grouped.get_group('conversion')
top_compras = compras['model'].value_counts().head(20)
top_compras
fig, ax = plt.subplots(figsize=(18, 14))
ax = sns.barplot(x = top_compras.values , y = top_compras.index )
ax.set_title("Compras por Modelo", fontsize=25)
ax.set_xlabel("Cantidad de Compras", fontsize=22)
ax.set_ylabel("Modelo", fontsize=22)
fig.tight_layout()
ax.tick_params(labelsize=14)
rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'    

    # Use X value as label and format number with one decimal place
    label = x_value.astype(int)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)  

celulares=trocafone.loc[trocafone['event']=='conversion']
celulares=celulares['model'].dropna().str.split().str.get(0)
celulares=celulares.value_counts()

fig, ax = plt.subplots(figsize=(24, 10))
ax = sns.barplot(x = celulares.values , y = celulares.index , palette = "RdBu")
ax.set_title("Compras por Marca", fontsize=25)
ax.set_xlabel("Cantidad de compras", fontsize=22)
ax.set_ylabel("Marca", fontsize=22)
fig.tight_layout()
ax.tick_params(labelsize=14)

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'   

    # Use X value as label and format number with one decimal place
    label = x_value.astype(int)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)
celulares=trocafone['model'].dropna().str.split().str.get(0)
celulares=celulares.value_counts()

fig, ax = plt.subplots(figsize=(24, 10))
ax = sns.barplot(x = celulares.values , y = celulares.index , palette = 'cubehelix')
ax.set_title("Visitas por marca", fontsize=25)
ax.set_xlabel("Cantidad de visitas", fontsize=22)
ax.set_ylabel("Marca", fontsize=22)
fig.tight_layout()
ax.tick_params(labelsize=14)

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = x_value.astype(int)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)
trocafone['marcaCel'] =trocafone['model'].dropna().str.split().str.get(0)

df=trocafone.loc[(trocafone['event']=='conversion')].groupby(['Month','marcaCel']).size().unstack()
df.drop([col for col, val in df.sum().iteritems() if val < 50], axis=1, inplace=True)
df.plot(kind='bar',figsize=(24,10),rot=0).legend(prop={'size': 22})

plt.title("Marcas más compradas por mes",fontsize='22')
plt.xlabel('Mes',fontsize='22')
plt.ylabel('Compras',fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.grid(axis='y')
capacidad= trocafone.loc[trocafone['storage'].notnull()]
cantidad_por_capacidad = capacidad['storage'].value_counts()
cantidad_por_capacidad = cantidad_por_capacidad.reindex(['512MB', '4GB', '8GB', '16GB', '32GB', '64GB', '128GB', '256GB', '512GB'])
cantidad_por_capacidad.plot(kind="bar", rot=0, figsize=(24,10) )
plt.xlabel('Capacidad',fontsize='22')
plt.ylabel('Visitas',fontsize='22')
plt.title("Celulares vistos por capacidad",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
capacidad= trocafone.loc[(trocafone['storage'].notnull()) & (trocafone['event']=='conversion')]
cantidad_por_capacidad = capacidad['storage'].value_counts()
cantidad_por_capacidad = cantidad_por_capacidad.reindex(['512MB', '4GB', '8GB', '16GB', '32GB', '64GB', '128GB', '256GB', '512GB'])
cantidad_por_capacidad.plot(kind="bar", rot=0,figsize=(24,10))

plt.xlabel('Capacidad',fontsize='22')
plt.ylabel('Compras',fontsize='22')
plt.title("Celulares comprados por capacidad",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
estado=trocafone.groupby('condition')
estado=estado.size().sort_values(ascending=False)
estado.rename(index={"Bom": 'Bueno',"Muito Bom":"Muy bueno","Bom - Sem Touch ID":"Bueno-Sin touch ID","Novo":"Nuevo"}).plot(kind='bar',rot=0,figsize=(24,8) ) 

plt.xlabel('Condición',fontsize='22')
plt.ylabel('Visitas',fontsize='22')
plt.title("Celulares vistos por condicion",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
estado=trocafone.loc[trocafone['event']=='conversion']
estado=estado.groupby('condition')
estado=estado.size().sort_values(ascending=False)
estado.rename(index={"Bom": 'Bueno',"Muito Bom":"Muy bueno","Bom - Sem Touch ID":"Bueno-Sin touch ID","Novo":"Nuevo"}).plot(kind='bar',rot=0,figsize=(24,8)) 

plt.xlabel('Condición',fontsize='22')
plt.ylabel('Compras',fontsize='22')
plt.title("Celulares comprados por condición",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
campaña=trocafone.groupby(['Month','person','event'])
campaña=campaña.size().unstack()
campaña=campaña.loc[(campaña['ad campaign hit'].notnull()) & (campaña['conversion'].notnull())]
campaña.groupby('Month').size().plot(kind='bar',rot=0,figsize=(24,8)) 

plt.xlabel('Mes',fontsize='22')
plt.ylabel('Usuarios',fontsize='22')
plt.title("Usuarios que compraron a partir de una campaña",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
device_types=trocafone[(trocafone['device_type']!='Unknown') & (trocafone['device_type'].notnull())]
device_types['device_type'].value_counts().plot(kind='bar',rot=0,figsize=(24,8)) 

plt.xlabel('Dispositivo',fontsize='22')
plt.ylabel('Eventos',fontsize='22')
plt.title("Tipos de dispositivos utilizados",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
device_types['event'].nunique()
#No hay datos sobre tipos de dispositivos con otros eventos
browser=trocafone['browser_version'].dropna().str.replace('\d+', '').str.replace('.', '')
browser_df=browser.value_counts().to_frame()
browser_df['percentages']=browser_df['browser_version']/browser_df['browser_version'].sum()*100
browser_df.head(10)
browser_df['percentages'].head(5).plot(kind='bar',rot=0,figsize=(24,8)) 

plt.title("Tipos de navegadores utilizados",fontsize='22')
plt.xlabel('Navegador',fontsize='22')
plt.ylabel('Eventos(%)',fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
buscadores = trocafone.loc[trocafone['event'] == 'search engine hit']
buscadores.index.size / trocafone.index.size * 100
frec_buscadores = buscadores['search_engine'].value_counts().to_frame('frecuencia')
frec_buscadores.index.title = 'Buscador'
frec_buscadores['porcentaje'] = buscadores['search_engine'].value_counts(normalize=True)*100
frec_buscadores
countries = trocafone.loc[trocafone["country"] != "Unknown"]
countries = countries.drop_duplicates(subset=['person', 'country'])
countries = countries["country"].value_counts().to_frame("cantidad")
countries.index.title = "pais"
countries = countries.assign(porcentaje=countries/countries.sum()*100)
countries.head(10)
cities = trocafone.loc[trocafone["city"] != "Unknown"]
cities = cities.drop_duplicates(subset=['person', 'city'])
cities = cities["city"].value_counts().to_frame("cantidad")
cities.index.title = "ciudad"
cities = cities.assign(porcentaje=cities/cities.sum()*100)
cities.head(10)
cities2 = trocafone.loc[trocafone["city"] != "Unknown"]
cities2 = cities2["city"].value_counts().to_frame("frecuencia")
cities2["porcentaje"] = (cities2["frecuencia"]/ cities2.frecuencia.sum())*100
cities2.index.title = "ciudad"
cities2.head(10)
fig, axs = plt.subplots(nrows=2 , figsize=(15,12))
plt.tight_layout(h_pad=9.0)

ax = sns.barplot(x=cities.head(10).index, y=cities.head(10)["cantidad"] ,  ax=axs[0] , palette = "Reds_d")
ax.set_title("Ciudades con más usuarios", fontsize=25)
ax.set_xlabel("Ciudad", fontsize=22)
ax.set_ylabel("Usuarios", fontsize=22)
ax.tick_params(labelsize=14)

ax = sns.barplot(x=cities2.head(10).index, y=cities2.head(10)["frecuencia"] ,  ax=axs[1] , palette = 'Reds_d')
ax.set_title("Ciudades con más eventos", fontsize=25)
ax.set_xlabel("Ciudad", fontsize=22)
ax.set_ylabel("Eventos", fontsize=22)
ax.tick_params(labelsize=14)
countries = trocafone.loc[trocafone["country"] != "Unknown"]
countries = countries["country"].value_counts().to_frame("cantidad")
countries.index.title = "pais"
countries = countries.assign(porcentaje=countries/countries.sum()*100)
countries.head(10)
cities_loc = pd.read_csv("../input/coordinates.csv")
cities_loc = cities_loc.dropna()
import folium

cm = plt.get_cmap("winter")

folium_map = folium.Map(tiles="Mapbox Bright", location=(0,0), zoom_start=2.47)

for city in cities_loc.values:
    marker = folium.CircleMarker(location=[city[2], city[3]], radius=1, color='red', opacity=0.5)
    marker.add_to(folium_map)
folium_map.zoom_control = False
folium_map
folium_map.zoom_start = 4
folium_map.location = (-15, -47)
folium_map
trocafone.loc[trocafone['event']=='conversion']['person'].nunique()
compras=trocafone.loc[trocafone['event']=='conversion'].groupby('person')
compras.size().mean()
compras.size().max()
compras.size().hist(figsize=(24,15),bins=716,width=1)

plt.title("Compras por usuario",fontsize='22')
plt.xlabel('Compras',fontsize=20)
plt.ylabel('Usuarios',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
(compras.size()==1).sum()/len(compras)*100
checkouts=trocafone.loc[trocafone['event']=='checkout'].groupby('person')
len(checkouts)
checkouts.size().mean()
len(compras)/len(checkouts)*100
total_checkouts=trocafone.loc[trocafone['event']=='checkout']
len(total_checkouts)
total_compras=trocafone.loc[trocafone['event']=='conversion']
len(total_compras)
len(total_compras)/len(total_checkouts)*100
vistos_por_persona = trocafone.loc[trocafone['event'] == 'viewed product'].groupby('person')['sku'].nunique()
max = vistos_por_persona.max()
min = vistos_por_persona.min()
promedio = vistos_por_persona.mean()
media = vistos_por_persona.median()
std = vistos_por_persona.std()
d = {"promedio":promedio, "media":media, "std":std, "max":max, "min":min}
pd.DataFrame(data=d, index=["productos vistos por persona"])
checkout = trocafone.loc[trocafone['event'] == 'checkout'].groupby('person')['event'].agg('count')
conversion = trocafone.loc[trocafone['event'] == 'conversion'].groupby('person')['event'].agg('count')
lead = trocafone.loc[trocafone['event'] == 'lead'].groupby('person')['event'].agg('count')
d = {'checkout': checkout[checkout < 15], 'conversion': conversion, 'lead': lead}
fig, ax = plt.subplots(figsize=(24, 10))
ax = sns.boxplot(data=pd.DataFrame(d))
ax = sns.stripplot(data=pd.DataFrame(d), jitter=True, dodge=True, alpha=0.5)
ax.set_title("Eventos por persona", fontsize=25)
ax.set_xlabel("Evento", fontsize=22)
ax.set_ylabel("Cantidad", fontsize=22)
publicidad = trocafone.loc[trocafone['event'] == 'ad campaign hit']
publicidad.index.size / trocafone.index.size * 100
frec_publicidad = publicidad['campaign_source'].value_counts(normalize=True).to_frame('porcentaje')
frec_publicidad.index.title = 'Buscador'
frec_publicidad['porcentaje'] = frec_publicidad['porcentaje']*100
a = frec_publicidad.head(10).plot(kind='bar', figsize=(20, 15),rot=0 )
plt.title("Ingreso de usuarios por publicidad",fontsize='22')
plt.xlabel('Campaña',fontsize=20)
plt.ylabel('Porcentaje',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
plt.grid(axis='y')