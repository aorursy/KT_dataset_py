# Alumnos:
#CALVO, MATEO IVÁN - 98290
#GUTIERREZ, MATÍAS - 92172
#PENNA, SEBASTIAN IGNACIO - 98752
#Link de GitHub: https://github.com/mateoicalvo/tp2datos
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
%matplotlib inline
labels  = pd.read_csv(r"../input/labels_training_set.csv", low_memory = False)
labels.set_index('person',inplace = True)
print(labels.shape)
labels.head()
events = pd.read_csv(r"../input/events_up_to_01062018.csv", low_memory = False)
events.shape
test_final  = pd.read_csv(r"../input/trocafone_kaggle_test.csv", low_memory = False)
test_final.set_index('person',inplace = True)
print(test_final.shape)
test_final.head()
#Formateo y creación de columnas
events['sku'] = pd.to_numeric(events['sku'], errors = 'coerce')
events['timestamp'] = pd.to_datetime(events['timestamp'])
events['mes'] = events['timestamp'].dt.month
events['hora'] = events['timestamp'].dt.hour
events['dia'] = events['timestamp'].dt.day
events['dia_de_la_semana'] = events['timestamp'].dt.dayofweek
events['dia_del_anio'] = events['timestamp'].dt.dayofyear
events['semana'] = events['timestamp'].dt.week
cantusuarios= events.loc[:,['person']].drop_duplicates()
print(cantusuarios.count())
months = events.sort_values('timestamp')
months = months.loc[:,{'timestamp','person'}]
months['month'] = months.timestamp.dt.month
months.sort_values('person').head(8)
features = events['person'].drop_duplicates().to_frame().set_index('person')
tmp1 = events.groupby('person').agg({'person' : 'count'})
tmp1.columns = ['cantidad_eventos']
features = features.join(tmp1, how='outer')
 #Actividad mes a mes y hasta junio
actividad = months.groupby(['person','month']).size().to_frame('month_activity').reset_index()
actividad['january_activity'] = np.where(actividad['month']==1, actividad['month_activity'], 0)
actividad['february_activity'] = np.where(actividad['month']==2, actividad['month_activity'], 0)
actividad['march_activity'] = np.where(actividad['month']==3, actividad['month_activity'], 0)
actividad['april_activity'] = np.where(actividad['month']==4, actividad['month_activity'], 0)
actividad['may_activity'] = np.where(actividad['month']==5, actividad['month_activity'], 0)
actividad = actividad.drop({'month','month_activity'},1)
actividad['total_activity'] = actividad.sum(1)
actividad = actividad.groupby('person').sum().reset_index()
actividad.head()
# Cantidad de productos que se mostraron como resultado de evento en skus
skusSeen = events.set_index('person').loc[:,'skus'].dropna().str.split(',').apply(len).reset_index().groupby('person').sum()
skusSeen = skusSeen.rename(columns={'skus':'skus_seen'})
skusSeen.head()
features = actividad.set_index('person').join(skusSeen)
features = features.fillna(0)
features.head()
dvc_type = events.sort_values('person').loc[:,{'person','device_type'}]
dvc_type = dvc_type.groupby(['person','device_type']).size().to_frame('device').reset_index()
# Filtro el dispositivo mas utilizado por cada usuario (para los casos donde mas de uno)
dvc_most_used = dvc_type.groupby('person').max().loc[:,'device_type'].to_frame()
one_hot_dvc = dvc_type.set_index('person').drop('device',1)
one_hot_dvc = pd.get_dummies(one_hot_dvc,prefix='device').groupby('person').sum()
dvc_features = one_hot_dvc.join(dvc_most_used)
dvc_features= dvc_features.drop(['device_type'], axis=1)
features = features.join(dvc_features, how='outer')
# Checkouts realizados en mayo, posibles ventas de junio
mayCheckouts = events.loc[(events.timestamp.dt.month==5)&(events.event=='checkout')].groupby('person').size().reset_index()
mayCheckouts = mayCheckouts.rename(columns={0:'may_checkouts'}).set_index('person')
mayCheckouts.head()
features = features.join(mayCheckouts)
features = features.fillna(0)
features.head()
# Cuantas conversiones realizo hasta junio
previousConversions = events.loc[events.event=='conversion'].groupby('person').size().reset_index()
previousConversions = previousConversions.rename(columns={0:'previous_conversions'}).set_index('person')
previousConversions.head(10)
features = features.join(previousConversions)
features = features.fillna(0)
features.head()
lastWeekCheckouts = events.set_index('timestamp').loc['2018-05-20':'2018-06-01']
lastWeekCheckouts = lastWeekCheckouts.loc[lastWeekCheckouts.event == 'checkout'].groupby('person').size().reset_index()
lastWeekCheckouts = lastWeekCheckouts.rename(columns={0:'last_2_weeks_chk'}).set_index('person')
lastWeekCheckouts = lastWeekCheckouts.fillna(0)
features = features.join(lastWeekCheckouts)
features.head()
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
def columnValue(x):
    switcher = {
        '4GB_hits': 4,
        '512MB_hits' : 0.512,
        '8GB_hits' : 8,
        '16GB_hits' : 16,
        '32GB_hits' : 32,
        '64GB_hits' : 64,
        '128GB_hits' : 128,
        '256GB_hits' : 256
    }
    return switcher.get(x)
# Features sobre el storage encontrado
strg = events.groupby(['person','storage']).size().to_frame('hits')
strg = strg.reset_index()
strg['storage'] = strg.storage.apply(absMemValue)
strg['512MB_hits'] = np.where(strg['storage']==0.512, strg['hits'], 0)
strg['4GB_hits'] = np.where(strg['storage']==4, strg['hits'], 0)
strg['8GB_hits'] = np.where(strg['storage']==8, strg['hits'], 0)
strg['16GB_hits'] = np.where(strg['storage']==16, strg['hits'], 0)
strg['32GB_hits'] = np.where(strg['storage']==32, strg['hits'], 0)
strg['64GB_hits'] = np.where(strg['storage']==64, strg['hits'], 0)
strg['128GB_hits'] = np.where(strg['storage']==128, strg['hits'], 0)
strg['256GB_hits'] = np.where(strg['storage']==256, strg['hits'], 0)
strg = strg.drop({'storage','hits'},1)
strg = strg.groupby('person').sum()
strg['storage_mode_GB'] = strg.idxmax(1)
strg['storage_mode_GB'] = strg['storage_mode_GB'].apply(columnValue)
strg.head()
features = features.join(strg)
features = features.fillna(0)
features.head()
lastWeekViewed = events.set_index('timestamp').loc['2018-05-20':'2018-06-01']
lastWeekViewed = lastWeekViewed.loc[lastWeekViewed.event == 'viewed product'].groupby('person').size().reset_index()
lastWeekViewed = lastWeekViewed.rename(columns={0:'last_10_days_viewed'}).set_index('person')
lastWeekViewed = lastWeekViewed.fillna(0)
features = features.join(lastWeekViewed)
features = features.fillna(0)
# lead de los ultimos 2 meses
lastLeads = events.set_index('timestamp').loc['2018-04-01':'2018-06-01']
lastLeads = lastLeads.loc[lastLeads.event=='lead'].groupby('person').size().reset_index()
lastLeads = lastLeads.rename(columns={0:'leads_done'}).set_index('person')
lastLeads = lastLeads.fillna(0)
lastLeads.head()
features = features.join(lastLeads)
features = features.fillna(0)
cantidad_eventos = events.groupby('person').agg({'person' : 'count'})
cantidad_eventos.columns = ['n_eventos']
features = features.join(cantidad_eventos, how='inner')
cantidad_eventos_mes = events.groupby(['person','mes']).agg({'person' : 'count'})
cantidad_eventos_mes = cantidad_eventos_mes.unstack().fillna(0)
cantidad_eventos_mes.columns = ['n_eventos_ENE',
                               'n_eventos_FEB',
                               'n_eventos_MAR',
                               'n_eventos_ABR',
                               'n_eventos_MAY']
features = features.join(cantidad_eventos_mes, how='outer')
cantidad_eventos_semana = events.loc[(events['mes'] == 5) | (events['mes'] == 4)].groupby(['person','semana']).agg({'person' : 'count'})
cantidad_eventos_semana = cantidad_eventos_semana.unstack().fillna(0)
cantidad_eventos_semana.columns = cantidad_eventos_semana.columns.droplevel()
nombres = {}
for semana_del_anio in cantidad_eventos_semana.columns:
    nombres[semana_del_anio] = 'n_eventos_semana_{}'.format(str(semana_del_anio))

cantidad_eventos_semana.rename(columns=nombres, inplace=True)
features = features.join(cantidad_eventos_semana, how='outer')
cantidad_eventos_dia = events.loc[events['mes'] == 5].groupby(['person','dia_del_anio']).agg({'person' : 'count'})
cantidad_eventos_dia = cantidad_eventos_dia.unstack().fillna(0)
cantidad_eventos_dia.columns = cantidad_eventos_dia.columns.droplevel()

nombres = {}
for dia_del_anio in cantidad_eventos_dia.columns:
    nombres[dia_del_anio] = 'n_eventos_dia_{}'.format(str(dia_del_anio))
cantidad_eventos_dia.rename(columns=nombres, inplace=True)

features = features.join(cantidad_eventos_dia, how='outer')
cantidad_eventos_dia_semana = events.groupby(['person','dia_de_la_semana']).agg({'person' : 'count'})
cantidad_eventos_dia_semana = cantidad_eventos_dia_semana.unstack().fillna(0)
cantidad_eventos_dia_semana.columns = cantidad_eventos_dia_semana.columns.droplevel()
nombres = {}
for dia_de_la_semana in cantidad_eventos_dia_semana.columns:
    nombres[dia_de_la_semana] = 'n_eventos_dia_de_la_semana_{}'.format(str(dia_de_la_semana))

cantidad_eventos_dia_semana.rename(columns=nombres, inplace=True)
features = features.join(cantidad_eventos_dia_semana, how='outer')
eventos_por_tipo = events.groupby(['person','event']).agg({'person' : 'count'}).unstack().fillna(0)
columnas = ['n_ad_campaign_hit', 'n_brand_listing', 'n_checkout', 'n_conversion', 'n_generic_listing', 'n_lead', 'n_search_engine_hit', 'n_searched_products', 'n_static_page', 'n_viewed_product', 'n_visited_site']
eventos_por_tipo.columns = columnas
features = features.join(eventos_por_tipo, how='outer')
cantidad_eventos_tipo_mes = events.loc[(events['mes'] == 5) | (events['mes'] == 4)].groupby(['person','event','mes']).agg({'person' : 'count'}).unstack().unstack().fillna(0)

cantidad_eventos_tipo_mes.columns = cantidad_eventos_tipo_mes.columns.droplevel()
columnas = []
for columna in cantidad_eventos_tipo_mes.columns:
    evento = columna[1]
    mes = columna[0]    
    columnas.append('n_{}_mes_{}'.format(evento,mes))

cantidad_eventos_tipo_mes.columns = columnas

features = features.join(cantidad_eventos_tipo_mes, how='outer') #
cantidad_eventos_tipo_semana = events.loc[(events['mes'] == 5) | (events['mes'] == 4)].groupby(['person','event','semana']).agg({'person' : 'count'}).unstack().unstack().fillna(0)
cantidad_eventos_tipo_semana.columns = cantidad_eventos_tipo_semana.columns.droplevel()

columnas = []
for columna in cantidad_eventos_tipo_semana.columns:
    evento = columna[1]
    semana = columna[0]    
    columnas.append('n_{}_semana_{}'.format(evento,semana))

cantidad_eventos_tipo_semana.columns = columnas

features = features.join(cantidad_eventos_tipo_semana, how='outer')
#Se toma visited site como el primer evento en el sitio, se le cambia el nombre para que se ordene
events['event'] = events['event'].replace(to_replace='visited site', value='aavisited site' )
events = events.sort_values(by=['person','timestamp','event'])
events['es_inicio_sesion'] = (events['event'] == 'aavisited site')
events['es_inicio_sesion'] = events['es_inicio_sesion'].astype(np.int8)
events['nro_sesion'] = events.groupby('person')[['person','es_inicio_sesion']].cumsum()
# duracion de sesion media, con desvio y cantidad de sesiones totales
tiempos_sesiones = events.groupby(['person','nro_sesion']) \
                       .agg({'timestamp' : ['min', 'max']})

tiempos_sesiones.columns = tiempos_sesiones.columns.droplevel()
tiempos_sesiones['duracion'] = (tiempos_sesiones['max'] - tiempos_sesiones['min']).dt.seconds

tiempos_sesiones = tiempos_sesiones.reset_index().groupby('person').agg({'duracion' : ['mean','std','min','max'], 'nro_sesion' : 'max'})
tiempos_sesiones.columns = tiempos_sesiones.columns.droplevel()
tiempos_sesiones.columns = ['duracion_sesion_media', 'duracion_sesion_desvio', 'duracion_sesion_minima', 'duracion_sesion_maxima', 'cantidad_sesiones']

features = features.join(tiempos_sesiones, how='outer')
# Canales de proveniencia para el total de los eventos
canales = events.groupby(['person','channel']).agg({'event' : 'count'}).unstack()
canales.columns = ['n_direct',
                  'n_email',
                  'n_organic',
                  'n_paid',
                  'n_referral',
                  'n_social',
                  'n_unknown']
features = features.join(canales, how='outer')
# Canales de proveniencia en las semanas de mayo
canales_por_semana_mayo = events.loc[events['mes'] == 5].groupby(['person','channel', 'semana']).agg({'event' : 'count'}).unstack().unstack()
canales_por_semana_mayo.columns = canales_por_semana_mayo.columns.droplevel()
columnas = []
for columna in canales_por_semana_mayo.columns:
    semana = columna[0]
    canal = columna[1]    
    columnas.append('n_{}_semana_{}'.format(canal,semana))
canales_por_semana_mayo.columns = columnas
features = features.join(canales_por_semana_mayo, how='outer')
# Se calculan por mes
events['es_conversion'] = (events['event'] == 'conversion')
events['n_eventos_hasta_convertir'] = events.groupby([events['person'],events['es_conversion'].cumsum()])['es_conversion'].apply(lambda x : (~x).cumsum())
eventos_hasta_convertir_por_mes = events.groupby(['person', 'mes']).agg({'n_eventos_hasta_convertir' : ['mean', 'std', 'max']}).unstack()
eventos_hasta_convertir_por_mes.columns = eventos_hasta_convertir_por_mes.columns.droplevel()
columnas = []
for columna in eventos_hasta_convertir_por_mes.columns:
    estadistica = columna[0]
    mes = columna[1]    
    columnas.append('{}_hasta_convertir_mes_{}'.format(estadistica,mes))
eventos_hasta_convertir_por_mes.columns = columnas
features = features.join(eventos_hasta_convertir_por_mes, how='outer')
features = features.fillna(0)
paisdeusuario = events.sort_values('person').loc[:,{'person','country'}]
paisdeusuario = paisdeusuario.groupby(['person','country']).size()
paisdeusuario= paisdeusuario.to_frame('country_count').reset_index()
paisdeusuario = paisdeusuario.groupby('person').max().loc[:,'country'].to_frame()
paisdeusuariolist=paisdeusuario['country'].value_counts().index
print(paisdeusuario['country'].value_counts().head())
paisdeusuario['country']=paisdeusuario['country'].replace(paisdeusuariolist, range(len(paisdeusuariolist)))
print(paisdeusuario.shape)
paisdeusuario.info()
print(paisdeusuariolist[1]) # unknown se asigno al 1
features = features.join(paisdeusuario,how='left')
features['country']=features['country'].fillna(1)
ciudadusuario = events.sort_values('person').loc[:,{'person','city'}]
ciudadusuario = ciudadusuario.groupby(['person','city']).size()
ciudadusuario= ciudadusuario.to_frame('city_count').reset_index()
ciudadusuario = ciudadusuario.groupby('person').max().loc[:,'city'].to_frame()
ciudadusuariolist=ciudadusuario['city'].value_counts().index
print(ciudadusuario['city'].value_counts().head())
ciudadusuario['city']=ciudadusuario['city'].replace(ciudadusuariolist, range(len(ciudadusuariolist)))
print(ciudadusuario.shape)
ciudadusuario.info()
print(ciudadusuariolist[0]) # unknown se asigno al 0
features = features.join(ciudadusuario,how='left')
features['city']= features['city'].fillna(0)
regionusuario = events.sort_values('person').loc[:,{'person','region'}]
regionusuario = regionusuario.groupby(['person','region']).size()
regionusuario= regionusuario.to_frame('region_count').reset_index()
regionusuario = regionusuario.groupby('person').max().loc[:,'region'].to_frame()
regionusuariolist=regionusuario['region'].value_counts().index
print(regionusuario['region'].value_counts().head())
regionusuario['region']=regionusuario['region'].replace(regionusuariolist, range(len(regionusuariolist)))
print(regionusuario.shape)
regionusuario.info()
print(regionusuariolist[1]) # unknown se asigno al 1
features = features.join(regionusuario,how='left')
features['region']= features['region'].fillna(1)
visitos = events.loc[events['device_type']=='Computer'].sort_values('person').loc[:,{'person','operating_system_version'}]
visitos = visitos.groupby(['person','operating_system_version']).size()
visitos= visitos.to_frame('operating_system_version_count').reset_index()
visitos = visitos.groupby('person').max().loc[:,'operating_system_version'].to_frame()
visitos= visitos.reset_index()
visitos.loc[visitos['operating_system_version'].str.contains("Mac"),['operating_system_version']] = 'Mac'
visitos.loc[visitos['operating_system_version'].str.contains("Windows"),['operating_system_version']] = 'Windows'
visitos.loc[(visitos['operating_system_version'].str.contains("Linux")) | (visitos['operating_system_version'].str.contains("Ubuntu")) | (visitos['operating_system_version'].str.contains("Fedora")),['operating_system_version']] = 'Linux'
visitos=visitos.rename(columns={'operating_system_version': 'comp_operating_system_version'})
SO_comp = visitos.set_index('person')
visitoslist=visitos['comp_operating_system_version'].value_counts().index
print(visitos['comp_operating_system_version'].value_counts().head())
visitos['comp_operating_system_version']=visitos['comp_operating_system_version'].replace(visitoslist, range(len(visitoslist)))
SO_comp = visitos.set_index('person')
features = features.join(SO_comp,how='left')
features['comp_operating_system_version']= features['comp_operating_system_version'].fillna(3)
visitos = events.loc[events['device_type']=='Tablet'].sort_values('person').loc[:,{'person','operating_system_version'}]
visitos = visitos.groupby(['person','operating_system_version']).size()
visitos= visitos.to_frame('operating_system_version_count').reset_index()
visitos = visitos.groupby('person').max().loc[:,'operating_system_version'].to_frame()
visitos= visitos.reset_index()
visitos.loc[visitos['operating_system_version'].str.contains("iOS"),['operating_system_version']] = 'iOS'
visitos.loc[visitos['operating_system_version'].str.contains("Android"),['operating_system_version']] = 'Android'
visitos=visitos.rename(columns={'operating_system_version': 'tab_operating_system_version'})
SO_tablet = visitos.set_index('person')
visitoslist=visitos['tab_operating_system_version'].value_counts().index
print(visitos['tab_operating_system_version'].value_counts().head())
visitos['tab_operating_system_version']=visitos['tab_operating_system_version'].replace(visitoslist, range(len(visitoslist)))
SO_tablet = visitos.set_index('person')
features = features.join(SO_tablet,how='left')
features['tab_operating_system_version']= features['tab_operating_system_version'].fillna(2)
visitos = events.loc[events['device_type']=='Smartphone'].sort_values('person').loc[:,{'person','operating_system_version'}]
visitos = visitos.groupby(['person','operating_system_version']).size()
visitos= visitos.to_frame('operating_system_version_count').reset_index()
visitos = visitos.groupby('person').max().loc[:,'operating_system_version'].to_frame()
visitos= visitos.reset_index()
visitos.loc[visitos['operating_system_version'].str.contains("iOS"),['operating_system_version']] = 'iOS'
visitos.loc[(visitos['operating_system_version'].str.contains("Android")) | (visitos['operating_system_version'].str.contains("Ubuntu")),['operating_system_version']] = 'Linux'
visitos.loc[visitos['operating_system_version'].str.contains("Windows"),['operating_system_version']] = 'Windows'
visitos.loc[visitos['operating_system_version'].str.contains("Other") | (visitos['operating_system_version'].str.contains("BlackBerry")) | (visitos['operating_system_version'].str.contains("Symbian")),['operating_system_version']] = 'Other'
visitos=visitos.rename(columns={'operating_system_version': 'SP_operating_system_version'})
SO_SP = visitos.set_index('person')
visitoslist=visitos['SP_operating_system_version'].value_counts().index
print(visitos['SP_operating_system_version'].value_counts().head())
visitos['SP_operating_system_version']=visitos['SP_operating_system_version'].replace(visitoslist, range(len(visitoslist)))
SO_SP = visitos.set_index('person')
features = features.join(SO_SP,how='left')
features['SP_operating_system_version']= features['SP_operating_system_version'].fillna(3)
mascomprada = events.loc[events.event=='conversion'].sort_values('person').loc[:,{'person','model'}]
mascomprada.loc[mascomprada['model'].str.contains("iPhone"),['model']] = 'iPhone'
mascomprada.loc[mascomprada['model'].str.contains("Samsung"),['model']] = 'Samsung'
mascomprada.loc[mascomprada['model'].str.contains("Motorola"),['model']] = 'Motorola'
mascomprada.loc[mascomprada['model'].str.contains("LG"),['model']] = 'LG'
mascomprada.loc[mascomprada['model'].str.contains("Asus"),['model']] = 'Asus'
mascomprada.loc[mascomprada['model'].str.contains("Sony"),['model']] = 'Sony'
mascomprada.loc[mascomprada['model'].str.contains("Lenovo"),['model']] = 'Lenovo'
mascomprada.loc[mascomprada['model'].str.contains("iPad"),['model']] = 'iPad'
mascomprada.loc[mascomprada['model'].str.contains("Quantum"),['model']] = 'Quantum'
mascomprada = mascomprada.groupby(['person','model']).size()
mascomprada= mascomprada.to_frame('model_count').reset_index()
mascomprada = mascomprada.groupby('person').max().loc[:,'model'].to_frame()
mascomprada= mascomprada.reset_index()
mascomprada=mascomprada.rename(columns={'model': 'modelo_mas_comprado'})
mascomprada['modelo_mas_comprado'].unique()
mascompradalist=mascomprada['modelo_mas_comprado'].value_counts().index
print(mascomprada['modelo_mas_comprado'].value_counts().head())
mascomprada['modelo_mas_comprado']=mascomprada['modelo_mas_comprado'].replace(mascompradalist, range(len(mascompradalist)))
modelo_mas_comprado = mascomprada.set_index('person')
features = features.join(modelo_mas_comprado,how='left')
features['modelo_mas_comprado']= features['modelo_mas_comprado'].fillna(9)
masvista = events.loc[events.event=='viewed product'].sort_values('person').loc[:,{'person','model'}]
masvista.loc[masvista['model'].str.contains("iPhone"),['model']] = 'iPhone'
masvista.loc[masvista['model'].str.contains("Samsung"),['model']] = 'Samsung'
masvista.loc[masvista['model'].str.contains("Motorola"),['model']] = 'Motorola'
masvista.loc[masvista['model'].str.contains("LG"),['model']] = 'LG'
masvista.loc[masvista['model'].str.contains("Asus"),['model']] = 'Asus'
masvista.loc[masvista['model'].str.contains("Sony"),['model']] = 'Sony'
masvista.loc[masvista['model'].str.contains("Lenovo"),['model']] = 'Lenovo'
masvista.loc[masvista['model'].str.contains("iPad"),['model']] = 'iPad'
masvista.loc[masvista['model'].str.contains("Quantum"),['model']] = 'Quantum'
masvista = masvista.groupby(['person','model']).size()
masvista= masvista.to_frame('model_count').reset_index()
masvista = masvista.groupby('person').max().loc[:,'model'].to_frame()
masvista= masvista.reset_index()
masvista=masvista.rename(columns={'model': 'modelo_mas_visto'})
masvista['modelo_mas_visto'].unique()
masvistalist=masvista['modelo_mas_visto'].value_counts().index
print(masvista['modelo_mas_visto'].value_counts().head())
masvista['modelo_mas_visto']=masvista['modelo_mas_visto'].replace(masvistalist, range(len(masvistalist)))
modelo_mas_visto = masvista.set_index('person')
features = features.join(modelo_mas_visto,how='left')
features['modelo_mas_visto']= features['modelo_mas_visto'].fillna(9)
checkout_producto = events.loc[events['event']=='checkout',['person','sku']]
producto_visto = events.loc[events['event']=='viewed product',['person','sku']]
n_veces_vistoycheckout = checkout_producto.merge(producto_visto,how= 'inner', right_on = ['sku', 'person'],left_on = ['sku', 'person']).groupby(['person'])['sku'].agg(['count']) 
n_veces_vistoycheckout= n_veces_vistoycheckout.rename(columns={'count': 'n_veces_vistoycheckout'})
features = features.join(n_veces_vistoycheckout,how='left')
features['n_veces_vistoycheckout']= features['n_veces_vistoycheckout'].fillna(0)
model_lead = events.loc[events['event']=='lead',['person','model']]
checkout_model = events.loc[events['event']=='checkout',['person','model']]
n_veces_leadycheckout = checkout_model.merge(model_lead,how= 'inner', right_on = ['model', 'person'],left_on = ['model', 'person']).groupby(['person'])['model'].agg(['count']) 
n_veces_leadycheckout= n_veces_leadycheckout.rename(columns={'count': 'n_veces_leadycheckout'})
features = features.join(n_veces_leadycheckout,how='left')
features['n_veces_leadycheckout']= features['n_veces_leadycheckout'].fillna(0)
features.head()
featurelist= features.columns
test = test_final.join(features,how='left')
print(test.shape)
test.head()
train= labels.join(features,how='left')
print(train.shape)
train.head()
#columnasdupli= train.columns
columnasdupli= train.columns.duplicated()
columnasdupli
X= train[featurelist]
X.info()
X_test= test[featurelist]
X_test.info()
y=train[['label']].values.ravel()
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.4, random_state=4)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=300)
scores = cross_val_score(knn, X, y, cv=10, scoring='roc_auc')
print(scores)
print(scores.mean())
knn.fit(X,y)
y_test= knn.predict_proba(X_test)[:,1]
finalpredknn = X_test.drop(featurelist,axis=1)
finalpredknn['label']= y_test
print(finalpredknn.shape)
finalpredknn.to_csv("prediccionknn.csv",index=True)
from xgboost import XGBClassifier
XGB = XGBClassifier(nthread=6, max_depth=3, n_estimators=20, seed=6, scale_pos_weight=18)
scores = cross_val_score(XGB, X, y, cv=10, scoring='roc_auc')
print(scores)
print(scores.mean())
XGB.fit(X,y)
y_test= XGB.predict_proba(X_test)[:,1]
finalpredXGB = X_test.drop(featurelist,axis=1)
finalpredXGB['label']= y_test
print(finalpredXGB.shape)
finalpredXGB.to_csv("prediccionXGB.csv",index=True)
featureimportance = pd.DataFrame(XGB.feature_importances_.tolist())
featureimportance['indice'] = features.columns
featureimportance.loc[featureimportance[0]>0].sort_values(by =  0, ascending = False)
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=100,min_samples_split=100,n_jobs=-1)
scores = cross_val_score(RF, X, y, cv=10, scoring='roc_auc')
print(scores)
print(scores.mean())
RF.fit(X,y)
y_test= RF.predict_proba(X_test)[:,1]
finalpredRF = X_test.drop(featurelist,axis=1)
finalpredRF['label']= y_test
print(finalpredRF.shape)
finalpredRF.to_csv("prediccionRF.csv",index=True)
featureimportance = pd.DataFrame(RF.feature_importances_.tolist())
featureimportance['indice'] = features.columns
featureimportance.loc[featureimportance[0]>0].sort_values(by =  0, ascending = False)
import lightgbm as lgb
LGBM=lgb.LGBMClassifier(learning_rate=0.01,objective='binary',num_leaves=4096,max_depth=13,n_estimators=250,colsample_bytree=0.8,n_jobs=-1,random_state=0,max_features=None)
scores = cross_val_score(LGBM, X, y, cv=10, scoring='roc_auc')
print(scores)
print(scores.mean())
LGBM.fit(X,y)
y_test= LGBM.predict_proba(X_test)[:,1]
finalpredLGBM = X_test.drop(featurelist,axis=1)
finalpredLGBM['label']= y_test
print(finalpredLGBM.shape)
finalpredLGBM.to_csv("prediccionLGBM.csv",index=True)
featureimportance = pd.DataFrame(LGBM.feature_importances_.tolist())
featureimportance['indice'] = features.columns
featureimportance.loc[featureimportance[0]>0].sort_values(by =  0, ascending = False)
todo = finalpredRF.join(finalpredXGB, how= 'left', lsuffix='rf', rsuffix='xgb')
#Promedio
todo['label'] = (0.6*todo['labelxgb'] + 0.4*todo['labelrf']) /2
todo[['label']].to_csv('predicts_merge_rf_xgb.csv', index = True)
todo = finalpredLGBM.join(finalpredXGB, how= 'left', lsuffix='lgbm', rsuffix='xgb')
#Promedio
todo['label'] = (0.6*todo['labelxgb'] + 0.4*todo['labellgbm']) /2
todo[['label']].to_csv('predicts_merge_lgbm_xgb.csv', index = True)