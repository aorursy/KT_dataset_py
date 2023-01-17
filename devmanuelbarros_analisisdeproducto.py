import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange
from datetime import timedelta, datetime
import seaborn as sns
#Para evitar algunas advertencias
pd.options.mode.chained_assignment = None
# Seteamos los datos para Seaborn
sns.set(font_scale=1.5, rc={'figure.figsize':(14,10)})
#Simplemente cargamos el dataset, que exportamos terminado el notebook analisisPreliminar
df_events = pd.read_csv('../input/events_2.csv', low_memory=False)
#Observamos los registros
df_events.info()
#hacemos una vista rapido de lo que contiene el dataset
df_events.head()
#Vamos a ver los enventos para tener un numero general.
df_events['event'].value_counts()
df_events['url'].value_counts()
# Le asignamos el type a la columna 'timestamp
df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
#Vamos a necesitar tener en numero el campo mes
df_events['mes_numero'] = df_events['timestamp'].apply(lambda x: x.month)
#Vamos a empezar a mirar los modelos que contiene el dataset
list_models = df_events['model'].value_counts().head(25)
#Listamos los modelos y la cantidad que aparecen por cada evento.
list_models
#converimos los valores a un array.
lista = list_models.index.tolist()
lista
#reseteamos el indice, si es que tiene.
df_events.reset_index(inplace=True)
df_events.index
#asignamos un indice para luego filtrar los valores que le pasemos en la lista.
df_events.set_index('model', inplace=True)
df_events.index
#Vamos a hacer un groupby y contar la cantidad de eventos y los presentamos.
df = df_events.loc[lista, :].groupby(['model','event'])[['event']].count().unstack()
df.columns = ['checkout', 'conversion', 'lead', 'viewed product']
df.fillna(0, inplace=True)
#ordenamos por cantidad de ventas.
df.sort_values(by='conversion',ascending=False)
#sumamos los eventos y los agregamos y filtramos.
df['total_events'] = df['checkout'] +  df['conversion'] + df['lead'] + df['viewed product']
#filtramos por cantidad de eventos..
df.head(25).sort_values(by='total_events', ascending=False)
#vemos un pantallazo de los datos...
df.describe()
df2 = df['viewed product']

#vemos la cantidad de ventas realizadas entre los primeros 25 productos
print("Ventas totales: {} | Total de ventas 25 tipos productos: {} | participación: {:.2f}%".format(df_events['event'].value_counts()['conversion'],
                                                                                         df['conversion'].sum(),
                                                                                        df['conversion'].sum()/df_events['event'].value_counts()['conversion']*100))
#Empezamos a realizar un analisis basado en las marcas de los productos.
df_events.reset_index(inplace=True)
df_events.index
df_events['Marca'] = "NaN"
df_events['model'].fillna("-",inplace=True)
df_events.loc[df_events['model'].str.contains('iPhone'), 'Marca'] = "iPhone"
df_events.loc[df_events['model'].str.contains('Samsung'), 'Marca'] = "Samsung"
df_events.loc[df_events['model'].str.contains('Motorola'), 'Marca'] = 'Motorola'
df_events.loc[df_events['model'].str.contains('Nokia'), 'Marca'] = 'Nokia'
df_events.loc[df_events['model'].str.contains('iPad'), 'Marca'] = 'iPad'
df_events.loc[df_events['model'].str.contains('Sony'), 'Marca'] = 'Sony'
df_events.loc[df_events['model'].str.contains('Quantum'), 'Marca'] = 'Quantum'
df_events.loc[df_events['model'].str.contains('Blackberry'), 'Marca'] = 'Blackberry'
df_events.loc[df_events['model'].str.contains('LG'), 'Marca'] = 'LG'
df_events.loc[df_events['model'].str.contains('Asus'), 'Marca'] = 'Asus'
df_events.loc[df_events['model'].str.contains('Lenovo'), 'Marca'] = 'Lenovo'
#hay que darle formato category a este campo....

df_events.loc[df_events['event'] =='conversion']
df_events['model'].value_counts()
filt = df_events.loc[(df_events['Marca'] == "NaN")]
#checkout	conversion	lead	viewed product	
filt['event'].value_counts()
#Estos registros son erroneos.
filt.loc[filt['event'] == 'checkout']
df_marcas = df_events.loc[df_events['Marca'] != 'NaN'].groupby(['Marca','event'])[['event']].count().unstack()
df_marcas.columns = ['checkout', 'conversion', 'lead', 'viewed product']
df_marcas.fillna(0, inplace=True)
df_marcas.sort_values(by='conversion',ascending=False)
#vamos a hacer un grafico que muestre la evolucion de las ventas en el tiempo.
#df_events['mes'] = df_events['mes'].astype('category')

df_marcas = df_events.loc[(df_events['Marca'] != 'NaN') & (df_events['event']=='conversion') & (df_events['mes'] != 'Junio')].groupby(['Marca','mes_numero'])[['contador']].sum().unstack(0)
#df_marcas.columns = ['Asus', 'LG', 'Lenovo', 'Motorola', 'Samsung', 'Sony', 'iPhone']
#df_marcas.index = ['4 - Abril', '1 - Enero', '2 - Febrero', '6 - Junio', '3 - Marzo', '5 - Mayo']
#df_marcas.index = df_marcas.index.sort_values()
df_marcas.index = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo']
labels = df_marcas.columns.get_level_values(1)
plt.plot(df_marcas, linewidth=4.5, marker='X', markersize=12)
plt.grid(True)
plt.title("Evolución de la ventas por linea de producto")
plt.ylabel("Cantidad de Conversiones")
plt.xlabel("Meses")
plt.legend(labels, loc="upper right")
plt.show()
#Controlar que efectivamente los campos son correctos.
marcas_a_analizar = ['Samsung', 'iPhone', 'Motorola']
sns.set(font_scale=1 , rc={'figure.figsize':(14,14)})
for i in marcas_a_analizar:
    df_detalle_marca  = df_events \
                                .loc[(df_events['Marca'] == i) & (df_events['event']=='conversion') & (df_events['mes'] != 'Junio')] \
                                .groupby(['model','mes_numero'])['contador'].sum()
    df_detalle_marca = df_detalle_marca.reset_index(name='contador')
    listaMeses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
    df_detalle_marca['mes_numero'] = df_detalle_marca['mes_numero'].apply(lambda x: str(x) + ")" + listaMeses[x-1]  )
    grafico2 = sns.heatmap(df_detalle_marca.pivot("model", "mes_numero", "contador"),  cmap="viridis", annot_kws={"size": 10})
    grafico2.set_xlabel('Mes')
    grafico2.set_ylabel('Modelos {}'.format(i))
    grafico2.set_title("Ventas de Aparatos Marca {} por Mes. Periodo: 01-01-18 / 31-05-2018".format(i))
    #plt.savefig("./graficos/grafico16.png", bbox_inches='tight')
    plt.show()
sns.set(font_scale=1.5, rc={'figure.figsize':(14,10)})

def generarSesion(user):
    return user + "-" + str(randrange(9999))
#Vamos a crear Sesiones de Usuarios.
#Vamos a trabajar con series para no ocupar toda la memoria.
serie_person = df_events['person']
serie_timestamp = df_events['timestamp']
i = 1

script_init_time =  datetime.now() #Para controlar cuanto tiempo tarda el script.
#primera carga
sesion = generarSesion(serie_person[0])
array_sesiones = []
array_sesiones.append(sesion)

#primer chacheo de hora.
cmp_hora = str(serie_timestamp[0] + timedelta(hours=3))

#comenzamos a recorrer toda la de "person" desde el valor 1.
for person in serie_person[1:]:
    if(person == serie_person[i-1]): #si Es igual seguimos dentro del mismo campo, no hay que generar una nueva sesion
        if(cmp_hora > str(serie_timestamp[i])):
            array_sesiones.append(sesion) #agremos a nuestro array una nueva sesion
        else:
            sesion = generarSesion(person) #Si ingreso aqui es otro usuario, necesitamos:
            array_sesiones.append(sesion) #generar una nueva sesion y 
            cmp_hora = str(serie_timestamp[i] + timedelta(hours=3)) #cachaear una nueva fecha
    else:
        sesion = generarSesion(person) #Si es una persona distinta, tambien debemos generar una sesion
        array_sesiones.append(sesion) # agregarla
        cmp_hora = str(serie_timestamp[i] + timedelta(hours=3)) #Y cachear una nueva hora.
        
    i += 1
    
    
script_fin_time =  datetime.now()
print("tiempo del script: {}".format(script_fin_time - script_init_time))
df_events['sesiones'] = pd.Series(array_sesiones)
df_events['sesiones'].value_counts()
person_con_conversiones = df_events.loc[df_events['event']=="conversion", 'person']
#df_events.reset_index(inplace=True)
df_events.set_index('person', inplace=True)
person_con_conversiones = person_con_conversiones.values.tolist()
df_con_conversiones = df_events.loc[person_con_conversiones]
df_con_conversiones.head()
#Cuantas veces ingresa en promedio la persona:
df_con_conversiones.groupby('sesiones')[['sesiones']].count().describe()
df_group_actividad = df_con_conversiones.groupby(['sesiones','event'])[['event']].count().unstack()
df_group_actividad.reset_index(inplace=True)
df_group_actividad.columns = df_group_actividad.columns.get_level_values(1)
df_group_actividad = df_group_actividad.loc[df_group_actividad['conversion']>0]
df_group_actividad
df_group_actividad.describe()
