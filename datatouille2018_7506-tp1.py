import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify # pip install squarify
import geopandas as gpd #  conda install -c conda-forge geopandas
from wordcloud import WordCloud # conda install -c conda-forge wordcloud
from pySankey import sankey# pip install pySankey
from shapely.geometry import Point
from time import strptime
from math import pi
from PIL import Image
import calendar

import plotly
import plotly.plotly as py # conda install -c conda-forge plotly
import plotly.graph_objs as go
from __future__ import division
plotly.tools.set_credentials_file(username='datatouille', api_key='GJ7Foc8nFWf23VZfzFSK')

%matplotlib inline

df = pd.read_csv('../input/events.csv', low_memory=False)
sns.set(style="darkgrid")
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.figsize'] = (25,15)
sns.set(font_scale=2)
with pd.option_context('display.max_column',0):
  display(df.sample(n=5))
bytes_used = df.memory_usage().sum()
print('Memoria usada: {:.2f}MB'.format(bytes_used/1000000))
print('{} atributos y {} registros en el dataframe.\n'.format(df.shape[1],df.shape[0]))
print('Primer registro: {} \nÚltimo registro: {}.'.format(df['timestamp'].min(),df['timestamp'].max()))
describe = df.describe().T
descripcion = pd.read_csv('../input/columns-desc.csv',index_col='column')
data = pd.merge(descripcion,describe,left_index=True,right_index=True)
data['null count'] = df.isnull().sum()
data['dtype'] = df.dtypes
with pd.option_context('display.max_colwidth',-1):
    display(data)
print('Cantidad de usuarios: {}'.format(df['person'].nunique()))

by_person = df[['event','person']].groupby('person')
con_checkouts = by_person.agg({'event':lambda x: any(y == 'checkout' for y in x)}).sum()['event']

print('Cantidad de usuarios con checkouts: {}'.format(con_checkouts))
# Los atributos con pocos valores posibles se pasan a variables categoricas para ahorrar memoria
df['event'] = df['event'].astype('category')
df['condition'] = df['condition'].astype('category')
df['storage'] = df['storage'].astype('category')
df['search_engine'] = df['search_engine'].astype('category')
df['channel'] = df['channel'].astype('category')
df['device_type'] = df['device_type'].astype('category')

# Se pasan los sku a números, para evitar conflictos entre skus iguales pero registrados como 1001 vs 1001.0
df['sku'] = df['sku'].replace({np.nan:0.0, 'undefined':0.0})
df['sku'] = df['sku'].astype('float64')

# El tiempo es mejor manejarlo como tal
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Chequeo
df.info()
ahorro = (bytes_used - df.memory_usage().sum())
porcentaje = (ahorro/bytes_used) * 100
print("Memoria ahorrada: {:.4f}MB ({:.2f}%)".format(ahorro/1000000,porcentaje))
df_sessions = pd.read_csv('../input/sessions.csv')
df = df.merge(df_sessions, how='left', left_index=True, right_index=True)
session_cols = ['person', 'timestamp', 'time_diff_min', \
        'session_id', 'session_total_events', \
        'session_cumno', 'session_first', 'session_last', \
        'session_conversion', 'session_checkout', 'session_ad']
df[session_cols].head(15)
df_brands = pd.read_csv('../input/brands.csv')
df = df.merge(df_brands, how='left', on='model')
df['brand'] = df['brand'].astype('category')
sample = df[df['model'].notnull()]
sample[['model','brand']].head()
df_os = pd.read_csv('../input/os.csv')
df = df.merge(df_os, how='left', on='operating_system_version')
df['operating_system'] = df['operating_system'].astype('category')
sample = df[df['operating_system_version'].notnull()]
sample[['operating_system_version', 'operating_system']].head()
df_browsers = pd.read_csv('../input/browsers.csv')
df = df.merge(df_browsers, how='left', on='browser_version')
df['browser'] = df['browser'].astype('category')
sample = df[df['browser_version'].notnull()]
sample[['browser_version','browser']].head()
df['month_number'] = df['timestamp'].dt.month
df['month_name'] = df['month_number'].apply(lambda x: calendar.month_abbr[x])
df['week_day'] = df['timestamp'].dt.weekday
df['week_number'] = df['timestamp'].dt.week
df['week_day_name'] = df['timestamp'].dt.weekday_name
df['day_date'] = df['timestamp'].dt.to_period('D')
df['day_dom'] = df['timestamp'].dt.day
df['hour_count'] = df['timestamp'].dt.hour
df['day_doy'] = df['timestamp'].dt.dayofyear
df['sku_name'] = df['model'] + ' ' + df['storage'].astype(str) + ' ' + df['color'] + ' (' + df['condition'].astype(str) + ')'
df[['sku','sku_name']].head()
df2 = df[df['event'] == 'conversion'][['day_date','person','event','model']]
df2.groupby('person').head()
double_tracking_rows = df2.duplicated(keep='last')
double_tracking_rows = double_tracking_rows[double_tracking_rows]
print('Se limpian {} registros producto de double tracking.'.format(double_tracking_rows.count()))
display(df2[df2.index.isin(double_tracking_rows.index)].sort_values('person').head())

df = df[~df.index.isin(double_tracking_rows.index)]
df2 = df[(df['session_total_events'] == 1) & (df['event'] == 'conversion')]
print('Se limpian {} registros de conversión directa.'.format(df2['event'].count()))
df = df[~df.index.isin(df2.index)]
event = df['event']
descripcion = pd.read_csv('../input/events-desc.csv',index_col='event')
descripcion['value_counts'] = event.value_counts()
with pd.option_context('display.max_colwidth',0):
    display(descripcion)
conversion_rate_total = (df.loc[df['event']=='conversion'].shape[0] / df.shape[0] )*100
print('Overall conversion rate, from 2018-01-01 to 2018-06-15: {:.3f}%'.format(conversion_rate_total))
data = df[['event','week_number']]
data = data.groupby(['week_number','event']).agg({'event':'count'})
data = data.rename(columns={'event':'count'})
data = data.reset_index()
data = data.pivot_table(index='week_number', values='count', columns='event')
data = pd.DataFrame(data.to_records())
data['conversion rate'] = ( data['conversion'] / data.sum(axis=1) ) * 100

plt.plot(data['conversion rate'])
plt.ylabel('Conversion rate')
plt.xlabel('Semana del año')
plt.xticks(data['week_number'])
plt.title('Conversion rate por semana')

#plt.savefig('informe/figures/010-conversion_rate_semana-lineplot.png')
data = df[['event','month_number']]
data = data.groupby(['month_number','event']).agg({'event':'count'})
data = data.rename(columns={'event':'count'})
data = data.reset_index()
data = data.pivot_table(index='month_number', values='count', columns='event')
data = pd.DataFrame(data.to_records())
data['conversion rate'] = ( data['conversion'] / data.sum(axis=1) ) * 100

visu = sns.barplot(x=data['month_number'],y=data['conversion rate'])
visu.set_title('Conversion rate según mes')
visu.set_ylabel('Conversion Rate')
visu.set_xlabel('Mes')
visu.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', '(First half) Jun'])

#plt.savefig('informe/figures/011-conversion_rate_mes-barplot.png')
orden = event.value_counts().head(7).index
visu = sns.countplot(x='event',data=df,order=orden)
visu.axes.set_title('Frecuencia de eventos')
visu.set_xlabel("Evento")
visu.set_ylabel("Cantidad")

#plt.savefig('informe/figures/02-eventos-barplot.png')
data = df.pivot_table(index='day_dom',columns='month_number', values='event', aggfunc='count')
data = ((data-data.min()) / (data.max() - data.min()))

visu = sns.heatmap(data.T,  cmap="OrRd")
visu.set_title("Tráfico (normalizado) en sitio según mes y día")
visu.set_xlabel("Día")
visu.set_ylabel("Mes")

#plt.savefig('informe/figures/030-eventos_segun_mes-heatmap.png')
data = df.pivot_table(index='week_day',columns='month_number', values='event', aggfunc='count')
data.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data = ((data-data.min()) / (data.max() - data.min()))

visu = sns.heatmap(data.T,  cmap="OrRd")
visu.set_title('Tráfico (normalizado) en sitio según mes y día de la semana')
visu.set_xlabel('Día de la semana')
visu.set_ylabel('Mes')

#plt.savefig('informe/figures/031-eventos_segun_dow-heatmap.png')
month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
month_counts = df.groupby('month_name').count()
month_counts = month_counts.loc[month_name]
visu = month_counts['event'].plot(kind='bar')
visu.axes.set_title('Cantidad de eventos segun mes')
visu.set_xlabel('Mes')
visu.set_ylabel("Cantidad")

#plt.savefig('informe/figures/032-eventos_segun_mes-barplot.png')
df_top = df.loc[(df['month_name'] == 'May') | (df['month_name'] == 'Jun')]

df_temporal = df_top[['event', 'day_dom']]
df_temporal = df_temporal.loc[(df_temporal['event'] == 'conversion') | (df_temporal['event'] == 'checkout') | (df_temporal['event'] == 'viewed product')]
df_temporal = df_temporal.groupby('day_dom')['event'].value_counts().unstack('event')

visu = plt.plot(np.log(df_temporal))
plt.legend(iter(visu), ('checkout', 'conversion', 'viewed_products'))
plt.title("Cantidad de eventos según día de los meses de junio y mayo")
plt.xlabel("Día")
plt.ylabel("Cantidad")
#plt.savefig('informe/figures/033-eventos_mayo_junio_lineplot.png')
df_conversion = df.loc[df['event'] == 'conversion']['hour_count'].value_counts().to_frame().sort_index()
df_checkout = df.loc[df['event'] == 'checkout']['hour_count'].value_counts().to_frame().sort_index()

labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
angles = [i / float(24) * 2 * pi for i in range(24)]
angles += angles[:1]  #cerrar el círculo

gray = '#999999'
black = '#000000'
orange = '#FD7120'
blue = '#00BFFF'
fig=plt.figure(figsize=(7,7))
series = plt.subplot(1, 1, 1, polar=True)
    
series.set_theta_offset(pi / 2)
series.set_theta_direction(-1)

plt.xticks(angles, labels, color=black, size=20)

plt.yticks([20, 40, 60, 80], ['20', '40', '60', '80'], color=gray, size=2)

plt.ylim(0,100),

series_values = df_conversion.values.flatten().tolist()
series_values += series_values[:1]  

series.set_rlabel_position(0)
series.plot(angles, series_values, color=orange, linestyle='solid', linewidth=1)
series.fill(angles, series_values, color=orange, alpha=0.5)
series.set_title('Cantidad de conversiones por hora', y=1.08)
#plt.savefig('informe/figures/040-hours-conversion-radarchart.png')
fig=plt.figure(figsize=(7,7))
series = plt.subplot(1, 1, 1, polar=True)
    
series.set_theta_offset(pi / 2)
series.set_theta_direction(-1)

plt.xticks(angles, labels, color=black, size=20)

#plt.yticks([1000, 1250, 1500, 1750, 2000], ['1000', '1250', '1500', '1750', '2000'], color=gray, size=7)
    
plt.ylim(0,2250)

series_values = df_checkout.values.flatten().tolist()
series_values += series_values[:1]  

series.set_rlabel_position(0)
series.plot(angles, series_values, color=blue, linestyle='solid', linewidth=1)
series.fill(angles, series_values, color=blue, alpha=0.5)
series.set_title('Cantidad de checkouts por hora', y=1.08)
#plt.savefig('informe/figures/041-hours-checkout-radarchart.png')
usuarios = df.groupby(['person', 'event'])
usuarios = usuarios.size().unstack(level = 'event')
usuarios = usuarios.fillna(usuarios.mean())
p = sns.boxplot(data=usuarios)
sns.set(font_scale=1.5)
plt.xticks(rotation=45)

#plt.savefig('informe/figures/190-usuarios_eventos-boxplot.png')
usuarios = df.groupby(['person', 'event'])
usuarios = usuarios.size().unstack(level = 'event')
usuarios = usuarios.fillna(usuarios.mean())
p = sns.boxplot(data=usuarios)
p.axes.set_ylim((0,25))
plt.xticks(rotation=45)

#plt.savefig('informe/figures/191-usuarios_eventos_truncado-boxplot.png')
countries = df['country'].value_counts()
countries = countries.drop('Unknown')
data = countries.head(3)

visu = squarify.plot(data, label=data.index, alpha=.5, color=['green','red','cyan'])
visu.set_title('Países con más visitas')

#plt.savefig('informe/figures/050-paises_visitas-treemap.png')
countries = df['country'].value_counts()
countries = countries.drop('Unknown')
countries = countries.drop('Brazil')
data = countries.head(7)

visu = squarify.plot(data, label=data.index, alpha=.5, color=['red','cyan','yellow','grey','purple','orange','blue'])
visu.set_title('Países con más visitas, exceptuando Brazil')

#plt.savefig('informe/figures/051-paises_visitas_sin_brazil-treemap.png')
regiones_brasil = df.loc[(df['country'] == 'Brazil')]['region']
data = regiones_brasil.value_counts()
data = data.drop('Unknown')

fig = data.head(7).plot(kind='bar')
fig.axes.set_title('Regiones de Brazil con más visitas')
fig.axes.set_ylabel('Visitas')
fig.axes.set_xlabel('Región')

#plt.savefig('informe/figures/060-regiones_brazil-barplot.png')
BR = pd.read_csv('../input/BR.csv', low_memory=False, sep='\t')
BR = BR[['name','latitude','longitude']]

# Para que geopandas pueda leer bien las latitudes y longitudes, deben ser de la clase Point
BR['coordinates'] = list(zip(BR['longitude'],BR['latitude']))
BR['coordinates'] = BR['coordinates'].apply(Point)
BR.head()
# Se debe hacer un join de los datos que se tienen (nombre de ciudad, cantidad de eventos) y los datos de geonames (nombre de ciudad, punto en el mapa), y esto plotearlo sobre los datos de geopandas (nombre de pais, punto en el mapa mundial)

# Se preparan los datos para el join (inner join de nombre de ciudad (columna name))
ciudades_brazil = df.loc[(df['country'] == 'Brazil')]
data = ciudades_brazil['city'].value_counts()
data = data.drop('Unknown')
data = data.to_frame()
data.reset_index(inplace=True)
data = data.rename(columns={'index':'name','city':'count'})

# Se pasa de un dataframe normal de pandas a uno de geopandas
BRA = gpd.GeoDataFrame(BR, geometry='coordinates')

# Se hace el inner join de ambos sets. Siendo que geonames daba mucha más información de la necesaria, duplicando valores por ciudades, se borran los duplicados
data = BRA.merge(data, on='name')
data = data.drop_duplicates('name')

# Se prepara el 'fondo' del gráfico, siendo este nomás el país. Para esto se usan los datos por defecto de geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world[world.name=='Brazil'].plot(color='white',edgecolor='black')

visu = data.plot(ax=ax,cmap='OrRd', legend=True)
visu.axes.set_title('Ciudades de Brazil con más visitas')

#plt.savefig('informe/figures/061-ciudades_brazil-choropleth.png')
# Buscamos las palabras más buscadas, con un mínimo de 300 busquedas

search_terms = df['search_term'].dropna()
search_terms = search_terms.apply(lambda x: x.lower())
search_terms = search_terms.value_counts()
search_terms = search_terms[search_terms >= 300]

# Para que funcione correctamente el módulo de wordcloud, hay que juntar todas las palabras en el mismo texto.
text = ''
for w,q in zip(search_terms.index,search_terms):
    text += ' '.join([w for x in range(q)])

text = ' '.join([s for s in text.split() if len(s)>2])    

wordcloud = WordCloud(width=2000, height=800, margin=0,collocations=False).generate(text)
 
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Keywords más buscadas')
plt.show()

#wordcloud.to_image().save('informe/figures/07-search_terms-wordcloud.png')
# Sabemos que el evento searched products refiere a varios SKUs. Spearemos y busquemos especificamente cuales son los más buscados.
searched = df[df['event']=='searched products']
skus_codiciados = searched['skus'].str.split(',').to_frame()
skus_codiciados = skus_codiciados['skus'].apply(pd.Series).unstack().reset_index()
skus_codiciados = skus_codiciados[['level_1',0]]
skus_codiciados.rename(columns={'level_1':'id',0:'sku'}, inplace=True)
skus_codiciados.sample(5)
top_5 = skus_codiciados['sku'].value_counts().head(5)
sku_names = df[df['sku'].isin(top_5.index)][['sku','sku_name']].drop_duplicates()

visu = top_5.plot(kind='bar', title ='5 productos más buscados por interfaz')
visu.set_title('5 productos más buscados por interfaz')
visu.set_xlabel('SKU')
visu.set_ylabel('Busquedas')

display(sku_names)
#plt.savefig('informe/figures/08-skus_buscados-barplot.png')
cronologia = df.loc[df['model'].notnull(), ['model', 'event','condition']]
cronologia['event'].value_counts()
# Se descartan los eventos 'lead' porque no influyen en el análisis
cronologia = cronologia.loc[cronologia['event'] != 'lead'] 

# Comparemos los modelos por los cuales más se opera segun el evento
vistos = cronologia.loc[cronologia['event']=='viewed product']['model'].value_counts().head()
checkouts = cronologia.loc[cronologia['event']=='checkout']['model'].value_counts().head()
comprados = cronologia.loc[cronologia['event']=='conversion']['model'].value_counts().head()

print("Más vistas") ; display(vistos)
print("Más checkouts") ; display(checkouts)
print("Más compras") ; display(comprados)
# Definimos un set de X modelos como los más prominentes, haciendo una combinación de los 7 más vistos, los 5 más por comprar  y los 5 más comprados
modelos_prominentes = set(vistos.index)
modelos_prominentes.update(checkouts.index)
modelos_prominentes.update(comprados.index)

eventos = cronologia.groupby('model')['event'].value_counts().unstack('event') 
eventos = eventos[['viewed product', 'checkout', 'conversion']]
eventos = eventos.loc[eventos.index.isin(modelos_prominentes)]
eventos
# Paso a porcentajes las columnas
eventos['total'] = eventos.sum(axis=1)
for c in eventos:
    eventos[c+'%'] = ( eventos[c] / eventos['total'] ) * 100

vistos = eventos['viewed product%']
checkouts = eventos['checkout%']
comprados = eventos['conversion%']

plt.bar(vistos.index, vistos, color=sns.xkcd_rgb["muted blue"],log=True)
plt.bar(checkouts.index, checkouts, bottom=vistos, color=sns.xkcd_rgb["muted green"],log=True)
plt.bar(comprados.index, comprados, bottom=vistos+checkouts, color=sns.xkcd_rgb["muted pink"],log=True)
plt.legend(['viewed product','checkout', 'conversion'])
plt.ylabel('Cantidad de eventos')
plt.xlabel('Modelos de celular')
plt.xticks(rotation=45)
plt.title('Cantidad de eventos en función de modelo (escala logarítmica)')
#plt.savefig('informe/figures/090-modelos_eventos-stackedbarplot.png')
for c in eventos[['viewed product','checkout','conversion']]:
    eventos[c+' ranking'] = eventos[c].rank('index', ascending=False)

rankings = (eventos.filter(regex='ranking')).T
#display(eventos[['viewed product','viewed product ranking','checkout','checkout ranking','conversion','conversion ranking']])
orden = (eventos.sort_values('viewed product ranking', ascending=True)).index
orden = [(str(i+1) + ' - ' + x) for i,x in enumerate(orden)]

orden2 = (eventos.sort_values('conversion ranking')).index
orden2 = [(str(i+1) + ' - ' + x ) for i,x in enumerate(orden2)]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(8)
ax.plot(rankings)
ax.set_yticklabels(orden)

ax2 = ax.twinx()
ax2.set_ylim(8)
ax2.set_yticklabels(orden2)

plt.title('Rankings de modelos para distintos eventos')
#plt.savefig('informe/figures/091-rankings_celulares-rank.png')
condiciones_analizadas = ['Bom', 'Muito Bom', 'Excelente']

condicion = cronologia.groupby('model')['condition'].value_counts().unstack('condition')
condicion = condicion[condiciones_analizadas]
condicion = condicion.loc[condicion.index.isin(modelos_prominentes)]

condicion['total'] = condicion.sum(axis=1)
for c in condicion:
    condicion[c] = ( condicion[c] / condicion['total'] ) * 100

bom = condicion['Bom']
muito_bom = condicion['Muito Bom']
excelente = condicion['Excelente']

plt.bar(bom.index, bom, color=sns.xkcd_rgb["muted blue"],log=True)
plt.bar(muito_bom.index, muito_bom, bottom=bom, color=sns.xkcd_rgb["muted green"],log=True)
plt.bar(excelente.index, excelente, bottom=bom+muito_bom, color=sns.xkcd_rgb["muted pink"],log=True)
plt.legend(condiciones_analizadas)
plt.ylabel('Cantidad')
plt.xlabel('Modelo')
plt.xticks(rotation=45)
plt.title('Cantidad de vistas y compras segun condición de modelo')
ax.set_yticklabels(orden)


#plt.savefig('informe/figures/10-condicion-stackedbarplot.png')
sns_colors = {'Preto':'black', 'Dourado':'gold', 'Branco':'white', 'Prata':'silver', 'Rosa':'pink', 'Azul':'blue',
       'Black Piano':'midnightblue', 'Olympic Edition':'dodgerblue', 'Cinza espacial':'grey', 'Prateado':'darkgray',
       'Ouro Rosa':'hotpink', 'Preto Matte':'darkslategray', 'Preto Brilhante':'k', 'Vermelho':'red', 'Branco Vermelho':'salmon', 'Bambu':'maroon', 'Preto Vermelho':'crimson'} 

data = df.loc[df['model'].isin(modelos_prominentes)][['model','color']].groupby('model')['color'].value_counts().to_frame()
data = data.rename(columns={'color':'count'})
data = data.reset_index()
visu = plt.scatter(data['model'], data['count'], s=3000, alpha=0.5,c=data['color'].apply(lambda x:sns_colors[x]))
visu.axes.set_title("Visitas a modelos prominentes segun color")
visu.axes.set_xlabel("Modelo")
visu.axes.set_ylabel("Visitas")

#plt.savefig('informe/figures/11-colores-bubbleplot.png')
df['staticpage'].value_counts()
faq_and_service = df[(df['staticpage'] == 'CustomerService') | (df['staticpage'] == 'FaqEcommerce')]['staticpage']
visu = sns.countplot(faq_and_service)
visu.set(xlabel='Page', ylabel='Visits')
visu.axes.set_title('Cantidad de eventos según staticpage')

#plt.savefig('informe/figures/12-static_pages-barplot.png')
order = df['new_vs_returning'].value_counts().index
visu = sns.countplot(df['new_vs_returning'].dropna(), order=order)
visu.set(xlabel='Tipo de usario', ylabel='Visitas al home')
visu.axes.set_title('Cantidad de eventos según marca')

#plt.savefig('informe/figures/130-eventos_new_returning-barplot.png')
gb = df.groupby(['person', 'new_vs_returning']).agg({'new_vs_returning' : 'size'})
gb = gb.unstack(level='new_vs_returning')
gb = gb['new_vs_returning']
returners_count = gb['Returning'].count()
only_once_count = gb['New'].count() - returners_count

objects = ('Regresan', 'Ingresan sólo una vez')
y_pos = np.arange(len(objects))
type_users_count = pd.Series([returners_count, only_once_count])
 
plt.bar(y_pos, type_users_count, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Cantidad')
plt.title('Tipos de usuarios')
 
#plt.savefig('informe/figures/131-tipos_usuarios-barplot.png')
plt.show()
order = df['brand'].value_counts().index
visu = sns.countplot(df['brand'].dropna(), order=order)
visu.axes.set_title('Cantidad de eventos segun marca')
visu.axes.set_xlabel('Cantidad')
visu.axes.set_ylabel('Marca')
sns.set(font_scale=2)

#plt.savefig('informe/figures/141-eventos_marca-barplot.png')
df_conv_by_brand = df.loc[(df['event'] == 'conversion') | (df['event'] == 'checkout')][['event', 'brand']]
# Es necesario para que no me muestre las otras categorías que ya fueron filtradas
df_conv_by_brand['event'] = df_conv_by_brand['event'].astype('object').astype('category')
ax = sns.countplot(x='brand', hue='event', data=df_conv_by_brand)
ax.set_yscale('log')

ax.legend(loc='upper left',bbox_to_anchor=(0, 1.1))
ax.set_xlabel('Cantidad (log)');
ax.set_ylabel('Marca');
ax.set_title('Relación de conversiones y checkouts en escala logarítmica');
sns.set(font_scale=3)

#plt.savefig('informe/figures/142-conversiones_checkouts_marca-barplot.png')
order = df['device_type'].value_counts().index
visu = sns.countplot(df['device_type'].dropna(), order=order)
visu.set(xlabel='Device Type', ylabel='Visits')
visu.axes.set_title('Cantidad de eventos segun tipo de dispositivo')

#plt.savefig('informe/figures/150-eventos_tipo-barplot.png')
data = df[['operating_system','brand','person']].groupby('person').first()
data = data.dropna()

data['operating_system'].replace({'mac':'iOS/mac', 'ios':'iOS/mac', 'ubuntu':'linux'},inplace=True)
oss = ['android','windows','iOS/mac','linux']
brands = ['samsung','motorola','iphone']

data = data[(data['brand'].isin(brands)) & (data['operating_system'].isin(oss))]
data.head()
display('Diagrama de flujo de fidelidad de usuarios')
sankey.sankey(data['operating_system'],data['brand'])
publicitados = df.loc[df['campaign_source'].notnull()][['campaign_source','month_number']] 

month_counts = publicitados.groupby('month_number').count()
month_counts = month_counts['campaign_source']
visu = sns.barplot(x=month_counts.index, y=month_counts)
visu.set_title('Cantidad de visitas por mes por una publicidad')
visu.set_ylabel('Visitas')
visu.set_xlabel('Mes')
visu.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', '(First half) Jun'])

#plt.savefig('informe/figures/16-presupuesto-barplot.png')
ranking = df['campaign_source'].value_counts()
ranking_visu = ranking.head(5)

visu = squarify.plot(ranking_visu, label=ranking_visu.index, alpha=.5, color=['red','cyan','yellow','grey','purple','orange','blue'])
visu.axes.set_title('5 metodos de publicidad más clikeados')

#plt.savefig('informe/figures/170-publicidad_clickeada-barplot.png')
ranking = ranking.drop('google')
ranking_visu2 = ranking.head(10)

ranking_visu2
visu = squarify.plot(ranking_visu2, label=ranking.index, alpha=.5, color=['red','cyan','yellow','grey','purple','orange','blue'])
visu.axes.set_title('10 metodos de publicidad más clikeados sin Google')

#plt.savefig('informe/figures/171-publicidad_sin_google-barplot.png')
channels = df[(df['channel'].notnull()) & (df['channel'] != 'Unknown')][['channel','event','person']].drop_duplicates()
channels['total persons'] = channels.groupby('channel')['person'].transform('count')

users = channels[['channel','person']]
total_persons = channels[['channel','total persons']].drop_duplicates()
conversions = df[df['event']=='conversion'][['person']]
total_conversions = users.merge(conversions, how='right').groupby('channel').agg({'person':'count'}).rename(columns={'person':'total conversions'})

channels_conversions = channels.merge(total_conversions,left_on='channel',right_index=True)
channels_conversions = channels_conversions[['channel','total persons','total conversions']].drop_duplicates()
channels_conversions['revenue'] = ( channels_conversions['total conversions'] / channels_conversions['total persons'] ) * 100
channels_conversions
revenue_label = list("{:.2f}%".format(x) for x in channels_conversions['revenue'])

barra_total = sns.barplot(x='channel', y='total persons', data=channels_conversions, color=sns.xkcd_rgb["muted blue"])
barra_conversiones = sns.barplot(x='channel', y='total conversions', data=channels_conversions, color=sns.xkcd_rgb["muted green"])
barra_total.set_yscale('log')
barra_conversiones.set_yscale('log')

for i,barra in enumerate(barra_total.patches[:6]):
    height = barra.get_height()
    barra_conversiones.text(barra.get_x() + barra.get_width()/2., height + 10, revenue_label[i], ha='center')

barra_total.set_title('Revenue by Traffic Source (escala logarítmica)')
barra_total.set_ylabel('Cantidad de eventos (log)')
barra_total.set_xlabel('Canal de tráfico')
barra_total.legend(['conversions','total'])

#plt.savefig('informe/figures/180-revenue_traffic-boxplot.png')
dfunnel = df[\
    (df['event'] == 'generic listing') | \
    (df['event'] == 'brand listing') | \
    (df['event'] == 'searched products') | \
    (df['event'] == 'viewed product') | \
    (df['event'] == 'checkout') | \
    (df['event'] == 'conversion') \
]

# color for each segment
colors = ['rgb(63,92,128)', 'rgb(90,131,182)', 'rgb(255,255,255)', 'rgb(127,127,127)', 'rgb(84,73,75)']
colors = ['rgb(63,92,128)', 'rgb(84,73,75)']
dfunnel = dfunnel.groupby(['event', 'session_ad']).size().unstack(level='session_ad')
dfunnel.head()
# Se cambió de nombre a las columnas, el orden de las filas y se pasó los valores a escala logarítmica

dfunnel.columns = ['Other', 'Ad']
dfunnel = dfunnel.reindex([\
                 'viewed product', 'checkout', 'conversion'])
dfunnel = np.log(dfunnel)
dfunnel
total = [sum(row[1]) for row in dfunnel.iterrows()]
n_phase, n_seg = dfunnel.shape
plot_width = 500
unit_width = plot_width / total[0]
 
phase_w = [int(value * unit_width) for value in total]
 
# height of a section and difference between sections 
section_h = 100
section_d = 10

# shapes of the plot
shapes = []
 
# plot traces data
data = []
 
# height of the phase labels
label_y = []
height = section_h * n_phase + section_d * (n_phase-1)

# rows of the DataFrame
df_rows = list(dfunnel.iterrows())

# iteration over all the phases
for i in range(n_phase):
    # phase name
    row_name = dfunnel.index[i]
    
    # width of each segment (smaller rectangles) will be calculated
    # according to their contribution in the total users of phase
    seg_unit_width = phase_w[i] / total[i]
    seg_w = [int(df_rows[i][1][j] * seg_unit_width) for j in range(n_seg)]
    
    # starting point of segment (the rectangle shape) on the X-axis
    xl = -1 * (phase_w[i] / 2)
    
    # iteration over all the segments
    for j in range(n_seg):
        # name of the segment
        seg_name = dfunnel.columns[j]
        
        # corner points of a segment used in the SVG path
        points = [xl, height, xl + seg_w[j], height, xl + seg_w[j], height - section_h, xl, height - section_h]
        path = 'M {0} {1} L {2} {3} L {4} {5} L {6} {7} Z'.format(*points)
        
        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': colors[j],
                'line': {
                    'width': 1,
                    'color': colors[j]
                }
        }
        shapes.append(shape)
        
        # to support hover on shapes
        hover_trace = go.Scatter(
            x=[xl + (seg_w[j] / 2)],
            y=[height - (section_h / 2)],
            mode='markers',
            marker=dict(
                size=min(seg_w[j]/2, (section_h / 2)),
                color='rgba(255,255,255,1)'
            ),
            text="Segment : %s" % (seg_name),
            name="Value : %d" % (dfunnel[seg_name][row_name])
        )
        data.append(hover_trace)
        
        xl = xl + seg_w[j]

    label_y.append(height - (section_h / 2))

    height = height - (section_h + section_d)
# For phase names
label_trace = go.Scatter(
    x=[-350]*n_phase,
    y=label_y,
    mode='text',
    text=dfunnel.index.tolist(),
    textfont=dict(
        color='rgb(200,200,200)',
        size=9
    )
)

data.append(label_trace)
 
# For phase values (total)
value_trace = go.Scatter(
    x=[350]*n_phase,
    y=label_y,
    mode='text',
    text=total,
    textfont=dict(
        color='rgb(200,200,200)',
        size=9
    )
)
layout = go.Layout(
    title="<b>Funnel (log) [Otro:Azul | Ad:Marrón]</b>",
    titlefont=dict(
        size=20,
        color='rgb(230,230,230)'
    ),
    hovermode='closest',
    shapes=shapes,
    showlegend=False,
    paper_bgcolor='rgba(44,58,71,1)',
    plot_bgcolor='rgba(44,58,71,1)',
    xaxis=dict(
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
#plotly.io.write_image(fig,'informe/figures/200-advertisement-funnel.png')