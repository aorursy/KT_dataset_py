import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pickle
import folium
from folium.plugins import HeatMap
from folium.features import CustomIcon
#from bokeh.plotting import figure, show, output_file
#from bokeh.tile_providers import CARTODBPOSITRON
#Para evitar algunas advertencias
pd.options.mode.chained_assignment = None
# Seteamos los datos para Seaborn
sns.set(font_scale=1.5, rc={'figure.figsize':(14,10)})
df_events = pd.read_csv('../input/eventos/events.csv', low_memory=False)
#En primer lugar vemos por arriba la información del dataset.
df_events.info()
df_events.shape
df_events['event'].value_counts()
# Graficamos para ver rapidamente los eventos generados.
eventos = df_events['event'].value_counts()
grafico1 = sns.barplot(eventos.index, eventos.values)
grafico1.grid(True)
grafico1.set_title("Eventos generados")
grafico1.set_ylabel("Cantidad de Eventos Generados")
grafico1.set_xlabel("Eventos Generados")
grafico1.set_xticklabels(grafico1.get_xticklabels(), rotation=90)
for i, v in enumerate(eventos.items()):        
    grafico1.text(i ,v[1], v[1], color='black', va ='bottom',  size=18)
plt.tight_layout()
#plt.savefig("./graficos/grafico1.png", bbox_inches='tight')
plt.show()
df_events.loc[df_events['event'] == 'visited site', 'timestamp'].count()
df_events.loc[df_events['event'] == 'visited site'].head(8)
#Asignamos el tipo datetime a la columna timestamp.
df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
#filtramos los campos con los que queremos trabajar para hacer la comprobaciones.
df_work_group = df_events.filter(items=['timestamp', 'person', 'event', 'new_vs_returning'])
#creamos la columna periodo y le asinamos una clave unica para diferenciar los días...
df_work_group['periodo'] = df_events['timestamp'].apply(lambda x: str(x.month) + str(x.day))
#Hacemos la agrupación 
df_work_group = df_work_group.groupby(['person', 'periodo','event'])
users, rows = df_events['person'].nunique(), df_work_group.ngroups
print("La cantidad de usuarios son: {} y la cantidad de registros generados {}, El promedio de actividad por persona es: {:.2f}".format(users, rows, (rows/users)))
# Hacemos apply y un unstack para acomodar los valores que deseamos chequear.
df_work_group = df_work_group.agg({'event':'count'}).unstack()
#cambiamos los nombres de las columnas para luego poder acceder sin problemas con los graficos.
level1 = df_work_group.columns.get_level_values(1).tolist()
i = 0
for e in level1:
    level1[i] = e.replace(" ", "_")
    i += 1
df_work_group.columns = level1
df_work_group
eventos_all = df_work_group.count()
grafico2 = sns.barplot(eventos_all.index, eventos_all.values)
grafico2.set_title("Contabilización de Eventos agrupados por usuario y día. Periodo: 01-01-18 / 15-06-18")
grafico2.grid(True)
grafico2.set_ylabel("Cantidad de apariciones de Eventos")
grafico2.set_xlabel("Eventos")
grafico2.set_xticklabels(grafico2.get_xticklabels(), rotation=90)
for i, v in enumerate(eventos_all.items()):        
    grafico2.text(i ,v[1], v[1], color='black', va ='bottom',  size=18)
plt.tight_layout()
#plt.savefig("./graficos/grafico2.png", bbox_inches='tight')
plt.show()
eventos_nan = df_work_group.query('visited_site=="NaN"').count()
grafico3 = sns.barplot(eventos_nan.index, eventos_nan.values)
grafico3.set_title("Contabilización de Eventos agrupados por usuario y día y que no aparezca el evento: 'visited site'. Periodo: 01-01-18 / 15-06-18")
grafico3.grid(True)
grafico3.set_ylabel("Cantidad de apariciones de Eventos")
grafico3.set_xlabel("Eventos")
grafico3.set_xticklabels(grafico3.get_xticklabels(), rotation=90)
for i, v in enumerate(eventos_nan.items()):        
    grafico3.text(i ,v[1], v[1], color='black', va ='bottom',  size=18)
#plt.savefig("./graficos/grafico3.png", bbox_inches='tight')
plt.show()
eventos_vs_not_nan =  df_work_group.query('visited_site!="NaN"').count()
grafico4 = sns.barplot(eventos_vs_not_nan.index, eventos_vs_not_nan.values)
grafico4.set_title("Contabilización de Eventos agrupados por usuario y día y siempre que aparezca el evento: 'visited site'. Periodo: 01-01-18 / 15-06-18")
grafico4.grid(True)
grafico4.set_ylabel("Cantidad de apariciones de Eventos")
grafico4.set_xlabel("Eventos")
grafico4.set_xticklabels(grafico4.get_xticklabels(), rotation=90)
for i, v in enumerate(eventos_vs_not_nan.items()):        
    grafico4.text(i ,v[1], v[1], color='black', va ='bottom',  size=18)
#plt.savefig("./graficos/grafico4.png", bbox_inches='tight')
plt.show()
df_filter_sesion1 =  df_work_group.filter(items=['visited_site', 'checkout', 'conversion'])
conversion_with_vs = df_filter_sesion1.query('visited_site!="NaN"').sum().drop(['visited_site'], axis=0)
conversion_without_vs = df_filter_sesion1.query('visited_site=="NaN"').sum().drop(['visited_site'], axis=0)
grafico5 = sns.barplot(conversion_with_vs.index, conversion_with_vs.values)
grafico5.set_title("Cantidad de Checkout y conversiones con sesiones iniciadas con Visite Site.")
grafico5.grid(True)
grafico5.set_ylabel("Suma de los eventos generados")
grafico5.set_xlabel("Eventos")
grafico5.set_xticklabels(grafico5.get_xticklabels(), rotation=90)
for i, v in enumerate(conversion_with_vs.items()):        
    grafico5.text(i ,v[1], v[1], color='black', va ='bottom',  size=18)
#plt.savefig("./graficos/grafico5.png", bbox_inches='tight')
plt.show()
grafico6 = sns.barplot(conversion_without_vs.index, conversion_without_vs.values)
grafico6.set_title("Cantidad de Checkout y conversiones con sesiones iniciadas directamente sin Visited Site.")
grafico6.grid(True)
grafico6.set_ylabel("Suma de los eventos generados")
grafico6.set_xlabel("Eventos")
grafico6.set_xticklabels(grafico6.get_xticklabels(), rotation=90)
for i, v in enumerate(conversion_without_vs.items()):
    grafico6.text(i ,v[1], v[1], color='black', va ='bottom',  size=18)
#plt.savefig("./graficos/grafico6.png", bbox_inches='tight')
plt.show()
#Obtenemos los usuarios que hayan realizado conversiones directamente sin haber iniciado sesión.
users_conversion_list = df_work_group.loc[(df_work_group['conversion']>0) & (df_work_group['visited_site'].isnull()==True)]
users_conversion_list = users_conversion_list.index.droplevel(1).tolist()
users_conversion_list = df_work_group.loc[users_conversion_list]
#Controlamos cuantos de esos usuarios han iniciado sesion
users_conversion_list = users_conversion_list.loc[users_conversion_list['visited_site']>0]
usuarios_ = len(users_conversion_list.groupby(['person']).count())
print("Cantidad de usuarios que han iniciado sesión {}".format(usuarios_))
def compEvent_User(listaEventos):
    comp_checkout = []
    for i in listaEventos:
        otro = df_events.loc[df_events['event'] == i ].groupby(['person', 'event'])[['event']].count()
        comp_checkout.append(int(otro.size))
    return comp_checkout
listaEventos = df_events['event'].value_counts().index.tolist()
cantidadUsuarios = compEvent_User(listaEventos)
grafico7 = sns.barplot(listaEventos, cantidadUsuarios)
grafico7.set_title("Cantidad de usuarios que han realizado cada evento al menos una vez.")
grafico7.grid(True)
grafico7.set_ylabel("Cantidad de Usuarios")
grafico7.set_xlabel("Eventos")
grafico7.set_xticklabels(grafico7.get_xticklabels(), rotation=90)
for i in range(len(listaEventos)):
    grafico7.text(i, cantidadUsuarios[i], cantidadUsuarios[i], color='black', va ='bottom',  size=18)
#plt.savefig("./graficos/grafico7.png", bbox_inches='tight')
plt.show()
ini = df_events.nsmallest(1, 'timestamp')['timestamp']
fin = df_events.nlargest(1, 'timestamp')['timestamp']
print("El primer evento registrado es en la fecha: {} y el ultimo registro es en la fecha: {}".format(str(ini.values)[2:12],
                                                                                                     str(ini.values)[2:12]))
#Generamos campos con los datos que necesitamos para trabajar comodamente
df_events['hora'] = df_events['timestamp'].apply(lambda x: x.hour)
df_events['dia'] = df_events['timestamp'].apply(lambda x: x.day)
df_events['mes'] = df_events['timestamp'].apply(lambda x: x.month)
listaMeses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
df_events['mes'] = df_events['mes'].apply(lambda x: listaMeses[x-1])
df_events['mes'] = df_events['mes'].astype('category')
df_events['mes'] = pd.Categorical(df_events['mes'], listaMeses)
df_events['mes'].value_counts()
#Analisis de cantidad de Eventos según Hora.
grafico8 = sns.distplot(df_events['hora'],  kde=False )
grafico8.set_title("Cantidad de eventos generado según el horario.")
grafico8.grid(True)
grafico8.set_ylabel("Cantidad de Eventos")
grafico8.set_xlabel("Hora del día")
plt.xlim(0,23)
#plt.savefig("./graficos/grafico8.png", bbox_inches='tight')
plt.show()
#Ahora veremos la cantidad de eventos generados según el día del mes para ver si se encuentra una relación
#Para esto sacaremos Junio para que no incremente los valores de los primeros 15 días.
sinJunio = df_events.loc[df_events['mes'] != 'Junio']
grafico9 = sns.distplot(sinJunio['dia'], bins='auto', kde=False )
grafico9.set_title("Cantidad de eventos generados según el día del mes. Periodo: 01-01-2018 al 31-05-2018.")
grafico9.grid(True)
grafico9.set_ylabel("Cantidad de Evnetos")
grafico9.set_xlabel("Día del Mes")
plt.xlim(1,31)
#plt.savefig("./graficos/grafico9.png", bbox_inches='tight')
plt.show()
#Seguimos trabajando sin junio.
eventos_mensuales = sinJunio['mes'].value_counts()
eventos_mensuales.index = pd.Categorical(eventos_mensuales.index, listaMeses[:-1])
grafico10 = sns.barplot(eventos_mensuales.index, eventos_mensuales.values)
grafico10.set_title(".")
grafico10.grid(True)
grafico10.set_ylabel("Eventos")
grafico10.set_xlabel("Mes")
grafico10.set_xticklabels(grafico10.get_xticklabels(), rotation=45)
#plt.savefig("./graficos/grafico10.png", bbox_inches='tight')
plt.show()
### Analisis Particulares de Eventos según en el tiempo.
# Compararemos el KDE de los distintos eventos por horario.
plt.subplot(4,3,1)
graf_eventos_hora = sns.distplot(df_events['hora'], bins='auto', kde=True, label="small" )
plt.ylabel("KDE Eventos")
plt.xlabel("Hora")
plt.title("Eventos generados por hora.")
plt.xlim(0,23)


number_graph = 2
#Recorremos la variable listaEventos que fue creada anteriormente.
for i in listaEventos:
    plt.subplot(4,3,number_graph)
    sns.distplot(df_events.loc[df_events['event'] == i, 'hora'], bins='auto', kde=True , label="small")
    plt.ylabel("KDE Eventos")
    plt.xlabel("Hora")
    plt.title("KDE del evento " + i)
    plt.xlim(0,23)
    number_graph += 1
    
plt.tight_layout()
#plt.savefig("./graficos/grafico11.png", bbox_inches='tight')
plt.show()
### Analisis Particulares de Eventos según en el tiempo.

plt.subplot(4,3,1)
sns.barplot(eventos_mensuales.index, eventos_mensuales.values)
plt.grid(True)
plt.title("Apareciones de Eventos")
plt.xlabel("Mes")
plt.ylabel("Apariciones")
plt.xticks(rotation=45)
number_graph = 2

for i in listaEventos:
    plt.subplot(4,3,number_graph)
    eventos_men = sinJunio.loc[sinJunio['event'] == i, 'mes'].value_counts()
    eventos_men.index = pd.Categorical(eventos_men.index, listaMeses[:-1])
    sns.barplot(eventos_men.index, eventos_men.values) 
    plt.title("Apariciones evento: " + i)
    plt.xlabel("Mes")
    plt.ylabel("Apariciones")
    plt.xticks(rotation=45)
    plt.grid(True)
    number_graph += 1

plt.tight_layout()
#plt.savefig("./graficos/grafico12.png", bbox_inches='tight')
plt.show()
# Compararemos el KDE de los distintos eventos por  día del mes.
plt.subplot(4,3,1)
graf_eventos_dia = sns.distplot(sinJunio['dia'], bins='auto', kde=True, label="small" )
plt.ylabel("KDE Eventos")
plt.xlabel("dia")
plt.title("Eventos generados por día.")
plt.xlim(0,23)


number_graph = 2
#Recorremos la variable listaEventos que fue creada anteriormente.
for i in listaEventos:
    plt.subplot(4,3,number_graph)
    sns.distplot(sinJunio.loc[sinJunio['event'] == i, 'dia'], bins='auto', kde=True , label="small")
    plt.ylabel("KDE Eventos")
    plt.xlabel("día")
    plt.title("KDE del evento " + i)
    plt.xlim(1,31)
    number_graph += 1
    
plt.tight_layout()
#plt.savefig("./graficos/grafico13.png", bbox_inches='tight')
plt.show()
#generamos los días de la semana
df_events['diaSem']=df_events['timestamp'].dt.weekday
eventos_by_day = df_events['diaSem'].value_counts(sort=False)
new_index = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
eventos_by_day.index = new_index
grafico14 = sns.barplot(eventos_by_day.index, eventos_by_day.values)
grafico14.set_title("Eventos por día de la semana. Periodo: 01-01-18 / 15-06-18")
grafico14.grid(True)
grafico14.set_ylabel("Cantidad de eventos")
grafico14.set_xlabel("Días")
grafico14.set_xticklabels(grafico14.get_xticklabels(), rotation=90)
for i, v in enumerate(eventos_by_day.items()):        
    grafico14.text(i ,v[1], v[1], color='black', va ='bottom',  size=18)
plt.tight_layout()
#plt.savefig("./graficos/grafico14.png", bbox_inches='tight')
plt.show()
df_Conversions = df_events['diaSem'][df_events['event'].str.strip()=='conversion']
conversions_by_day = df_Conversions.value_counts(sort=False)
conversions_by_day.index = new_index
grafico15 = sns.barplot(conversions_by_day.index, conversions_by_day.values)
grafico15.set_title("Conversiones agrupadas por día. Periodo: 01-01-18 / 15-06-18")
grafico15.grid(True)
grafico15.set_ylabel("Cantidad de conversiones")
grafico15.set_xlabel("Días")
grafico15.set_xticklabels(grafico15.get_xticklabels(), rotation=90)
for i, v in enumerate(conversions_by_day.items()):        
    grafico15.text(i ,v[1], v[1], color='black', va ='bottom',  size=18)
plt.tight_layout()
#plt.savefig("./graficos/grafico15.png", bbox_inches='tight')
plt.show()
#new_index = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
#Generamos los indices y creamos una columna nueva con los nombres de la semana para agruparlos.
df_events['diaSem2'] = df_events['diaSem'].apply(lambda x: new_index[x])
df_events['diaSem2'] = df_events['diaSem2'].astype('category')
df_events['diaSem2'] = pd.Categorical(df_events['diaSem2'], new_index)
df_new = df_events.groupby(['diaSem2', 'hora'])['event'].count()
df_new = df_new.reset_index(name="event")
grafico16 = sns.heatmap(df_new.pivot("hora", "diaSem2", "event"), annot=False, cmap="viridis")
grafico16.set_xlabel('Día de la semana')
grafico16.set_ylabel('Hora')
grafico16.set_title("Eventos por día de la semana y hora. Periodo: 01-01-18 / 15-06-18")
#plt.savefig("./graficos/grafico16.png", bbox_inches='tight')
plt.show()
#Realizamos un groupby y el apply contabilizando los eventos.
df_new = df_events[df_events['event'].str.strip()=='conversion'].groupby(['diaSem2', 'hora'])['event'].count()
df_new = df_new.reset_index(name="event")
# Pivot the dataframe to create a [hour x date] matrix containing counts
grafico16 = sns.heatmap(df_new.pivot("hora", "diaSem2", "event"), annot=False, cmap="viridis")
grafico16.set_xlabel('Día de la semana')
grafico16.set_ylabel('Hora')
grafico16.set_title("Eventos por día de la semana y hora. Periodo: 01-01-18 / 15-06-18")
#plt.savefig("./graficos/grafico16.png", bbox_inches='tight')
plt.show()
#Veremos cuantos valores distintos y que cantidad de ocurrencias tiene en el Dataset.
df_events['country'].value_counts()
# Predomina Brasil. Vamos a controlar que tipo de eventos se generan desde otros 
# paises, para controlar que no afecte nuestros resultados.
# Filtramos los que no son Brazil y también eliminamos los que tiene el campo vacio.
ext_visit = df_events.loc[(df_events['country'] != "Brazil") & (df_events['country'].isnull() == False), 'timestamp'].count()
# Controlamos las veces que se genero el evento 'visited site'. Y lo medimos con timestamp que es un
# campo que nunca es nulo.
total_visit = df_events.loc[(df_events['event'] == 'visited site'), 'timestamp'].count()
#Imprimimos los resultados.
print("Ingresos totales: {} | Ingresos Brazil {}  | Otros Ingresos {}".format(total_visit, 
                                                                              (total_visit-ext_visit),
                                                                             ext_visit))
df_events.reset_index(inplace =True)
# Creamos una lista con los usuarios que se conectaron desde otro lugar que no sea Brasil y controlamos que la cantidad sea 
# consistente con los resultados obtenidos anteriormente.
list_person = df_events.loc[(df_events['country'] != 'Brazil') & (df_events['event'] == 'visited site'), 'person'].tolist()
len(list_person)
# Cambiamos el indice para filtrar los usuarios que necesitamos.
df_events.set_index('person', inplace=True)
# Filtramos en un nuevo dataFrame los usuarios que se conectaron desde afuera.
df_person = df_events.loc[list_person]
# Agrupamos filtrando solo los inicio de Sesion que no hayan sido desde Brasil.
# Esto nos da por resultado usuarios que ingresaron solo desde afuera u origen desconocido.
person_ext = df_person.loc[(df_person['event'] == 'visited site') & (df_person['country'] != 'Brazil')].groupby(['person', 'country'])['country'].count().unstack()
# Rellanamos con cero, solo por cuestion visual.
person_ext.fillna(0, inplace=True)
# Revisarmos rapidamente los resultados.
person_ext.head(10)
#Tomamos un registro para realizar un control.
control = df_events.loc['00204059']
control.loc[control['event'] == 'visited site']
#Obtenemos los usuarios que hay realizados conversiones.
user_conversion = df_events.loc[df_events['event'] == 'conversion', :].groupby('person')['event'].count().index.tolist()
# Eliminamos de la lista de usuarios a eliminar los que hayan realizado conversiones.
list_person = person_ext.index.tolist()
for user in user_conversion:
    if user in list_person:
        list_person.remove(user)
print("Cantidad de usuarios que se conectaron desde otro lugar o desconocido: {}".format(len(list_person)))
df_user_drop = df_events.loc[list_person]
sesion_count =df_user_drop.loc[df_user_drop['event'] == 'visited site', 'timestamp'].count()
reg_count = len(df_user_drop)
print("La cantidad de inicios de sesión a eliminar: {} | registros a eliminar: {}.".format(sesion_count, reg_count))
# Eliminamos los registros y controlamos el nuevo tamaño de los registros.
df_events = df_events.drop(list_person, axis=0)
df_events.shape
# Vemos ahora los sesiones que se han iniciado desde donde sea...
# Asignamos las desconocidas a Brasil
df_events.loc[df_events['country'] == 'Unknown', 'country'] = 'Brazil'
df_events['country'].value_counts()
df_events['region'].value_counts()
print("Cantidad de Regiones: {}".format(len(df_events['region'].value_counts())))
df_events['city'].value_counts()
df_events.reset_index(inplace=True)
df_events.loc[(df_events['region'] == 'Unknown'), 'tmpRegion'] = 'Unknown'
lista = df_events.loc[df_events['tmpRegion'] == 'Unknown', :].groupby('person')['person'].count()
lista = lista.index.tolist()
df_events.set_index('person', inplace=True)
df_temp = df_events.loc[lista]
df_temp = df_temp.loc[(df_temp['event'] == 'visited site')].groupby(['region', 'device_type']).agg({'region' :  'count'})
df_temp.unstack()
df_temp2 = df_events.loc[df_events['event']=='visited site'].groupby(['city','device_type']).agg({'city' :  'count'})
df_temp2.unstack()
def cargar_datos(archivo):
    try:
        with open("../input/" + archivo, "rb") as f:
            return pickle.load(f)
    except (OSError, IOError) as e:
        return dict()
dicc_lat_long = cargar_datos('dic-final/dic_final.dat')
dicc_lat_longRegion = cargar_datos('dic-corto/dic2_regiones.dat')
df_events['concat_address'] = df_events['city'] + ", " + df_events['region'] + ", " + df_events['country']
df_events['concat_address_short'] = df_events['region'] + ", " + df_events['country']
df_events.loc[df_events['concat_address'].isnull()==True, 'concat_address'] = "N"
df_events.loc[df_events['concat_address_short'].isnull()==True, 'concat_address_short'] = "N"
#Agregamos la latitud y la longitud.
df_events['lat'] = df_events['concat_address'].apply(lambda x: dicc_lat_long[x]['lat'] if x!="N" else None)
df_events['lng'] = df_events['concat_address'].apply(lambda x: dicc_lat_long[x]['lng'] if x!="N" else None)
df_events['lat_reg'] = df_events['concat_address_short'].apply(lambda x: dicc_lat_longRegion[x]['lat'] if x!="N" else None)
df_events['lng_reg'] = df_events['concat_address_short'].apply(lambda x: dicc_lat_longRegion[x]['lng'] if x!="N" else None)
#Vamos a llevar a los cantidad de ingresos a la mitad, porque en el caso contrario cuando queremos hacer el mapa de calor se traba.
df_events_grouped_address = df_events.loc[(df_events['event']=='visited site') & (df_events['region'] != 'Unknown') & (df_events['city'] != 'Unknown')].groupby(['concat_address','lat','lng'])['concat_address'].count()
df_events_grouped_address = df_events_grouped_address.apply(lambda x: round(x/2,0))
#Lo que hacemos es contar la cantidad de veces que se ingreso en determinada region, ese numero se lo divide por 2
#Luego lo que hacemos es generar una lista donde se repiten la cantidad de apariciones para que puedan ser interpretadas
# por el plugins de Folium.
otro = []
for i, valores in enumerate(df_events_grouped_address.items()):
    latylng = [valores[0][1], valores[0][2]]
    for generar in range(int(valores[1])):
        otro.append(latylng)
mapF = folium.Map(location=[-13.778429999999958,-55.92864999999995], zoom_start = 4.5)
mapF.add_child(folium.plugins.HeatMap(otro, radius=15))
mapF
df_events_grouped_region = df_events \
                                       .loc[(df_events['event']=='visited site') & (df_events['region'] != 'Unknown') & (df_events['country'] == 'Brazil')] \
                                       .groupby(['concat_address_short', 'device_type'])['device_type'].count()

df_events_grouped_region = df_events_grouped_region.reset_index(name='total')
df_events_grouped_region.loc[df_events_grouped_region['concat_address_short']=='Colorado, United States']
grafico18 = sns.heatmap(df_events_grouped_region.pivot("concat_address_short", "device_type", "total"), annot=False, cmap="viridis")
grafico18.set_xlabel('Tipo de Dispositivo')
grafico18.set_ylabel('Región')
grafico18.set_title("Sesiones Iniciadas por Tipo de dispositivo por Región")
#plt.savefig("./graficos/grafico18.png", bbox_inches='tight')
plt.show()
df_events_grouped_region = df_events \
                                       .loc[(df_events['event']=='visited site') & (df_events['region'] != 'Unknown')] \
                                       .groupby(['concat_address_short','lat_reg','lng_reg', 'device_type'])['concat_address_short'].count()
df_events_grouped_region.unstack(3)
#Vamos a ver que direccion vamos a agregar
totalValores = df_events_grouped_region.values.sum()
mapFr = folium.Map(location=[-13.778429999999958,-55.92864999999995], zoom_start = 4.5)
def agregarMarcador(valores, mapa):
    dispositivo = valores[0][3]
    latitud = valores[0][1]
    longitud = valores[0][2]
    size = valores[1]
    if size < 500:
        size = 20 + (size/200*2)
    if size > 500 and size < 5000:
        size = 30+(size/3/100*2)
    if size > 10000:
        size = size/70000*400
    nombre = valores[0][0]
    cantidad = valores[1]
    
    factor = 0.2
    if dispositivo == 'Computer':
        url = 'https://icon-icons.com/icons2/1367/PNG/512/32officeicons-31_89708.png'
        latitud = latitud - factor
    if dispositivo == 'Smartphone':
        url = 'https://diariodeunsysadmin.files.wordpress.com/2014/05/smartphone-android-icon.png?w=600'
        longitud = longitud + factor
    if dispositivo == 'Tablet':
        url = 'https://images.vexels.com/media/users/3/128862/isolated/preview/5b021d17fb3643d144434b4cc6c3a74c-tablet-icono-plana-by-vexels.png'
        latitud = latitud + factor
    if dispositivo == 'Unknown':
        url = 'https://images.vexels.com/media/users/3/143554/isolated/lists/4891b5f6c604304b74f030ce8a13f762-icono-de-signo-de-interrogaci-n-3d-rojo.png'
        longitud = longitud - factor
    icon = CustomIcon(
    url,
    icon_size=(size, size*2)
    )
    marker = folium.Marker(
    location=[latitud, longitud],
    icon=icon,
    popup=nombre + " Cantidad Eventos:  " + str(cantidad)
    )
    mapa.add_child(marker)


totalValores = df_events_grouped_region.values.sum()
for i, valores in enumerate(df_events_grouped_region.items()):
    agregarMarcador(valores, mapFr)
mapFr
mapFrUn = folium.Map(location=[-13.778429999999958,-55.92864999999995], zoom_start = 4.5)

for i, valores in enumerate(df_events_grouped_region.items()):        
    if(valores[0][3] == 'Unknown'):
        agregarMarcador(valores, mapFrUn)
mapFrUn
mapFrUn = folium.Map(location=[-13.778429999999958,-55.92864999999995], zoom_start = 4.5)
for i, valores in enumerate(df_events_grouped_region.items()):
    if(valores[0][3] == 'Computer'):
        agregarMarcador(valores, mapFrUn)
mapFrUn
mapFrUn = folium.Map(location=[-13.778429999999958,-55.92864999999995], zoom_start = 4.5)
for i, valores in enumerate(df_events_grouped_region.items()):        
    if(valores[0][3] == 'Smartphone'):
        agregarMarcador(valores, mapFrUn)
mapFrUn
mapFrUn = folium.Map(location=[-13.778429999999958,-55.92864999999995], zoom_start = 4.5)
for i, valores in enumerate(df_events_grouped_region.items()):
    if(valores[0][3] == 'Tablet'):
        agregarMarcador(valores, mapFrUn)
mapFrUn
df_events['contador'] = 1
df_porDispositivo = df_events.loc[df_events['event']=='visited site'].groupby(['device_type']).agg({'contador' :  'sum'}).unstack(1)
reordenar = [0,2,1,3]
size = []
labels = []
for i in reordenar:
    size.append(df_porDispositivo.values[i])
    labels.append(df_porDispositivo.index.get_level_values(1)[i])
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} visited site)".format(pct, absolute)

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
fig1, ax1 = plt.subplots(figsize=(8,8))
plt.figure(figsize=(1,1))
wedges, texts, autotexts= ax1.pie(size,  colors=colors, labels=labels,  autopct=lambda pct: func(pct, size),
        shadow=True, startangle=160)
ax1.axis('equal') 

ax1.legend(wedges,labels,
          title="Tipo Dispotivo",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
ax1.set_title('Participación de conecciones por tipo de Dispositivo', fontsize=16)
plt.tight_layout()
#plt.savefig("./graficos/grafico13.png", bbox_inches='tight')
plt.show()
df_events.to_csv('events_2.csv', header=False, index=False)