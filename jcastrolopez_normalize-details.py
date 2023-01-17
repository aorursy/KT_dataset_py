import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as datetime

import sys



VALOR_CAMBIO_A_DOLAR = 19.54
df = pd.read_csv('../input/mexican-zonaprop-datasets/train.csv',

        index_col='id',

        dtype={'gimnasio': bool,

                'usosmultiples': bool,

                'escuelascercanas': bool,

                'piscina': bool,

                'centroscomercialescercanos': bool,

                'tipodepropiedad': 'category',

                'provincia': 'category',

                'ciudad': 'category'

            },

        parse_dates=['fecha'])

pd.set_option('display.float_format', '{:.2f}'.format)
df['tipodepropiedad'].isin(['Apartamento'])
df.columns
# Me quedo solo con las columnas que son importantes para nuestro analisis.

# Quizas el titulo y la descripcion tampoco las use, pero primero habria que hacer  un analisis de ese campo

df.drop(['direccion','idzona','lat','lng'], axis=1, inplace=True)
df.columns
# 1. Ver cuantos datos (observaciones) tenemos en total

# 2. Ver cuantos atributos tiene cada observaci´on

# 3. Ver el nombre y tipo de dato de cada observaci´on

# 4. Ver cuantos valores faltantes existen y en que proporci´on se presentan para

# cada atributo

# Despues de parsear y filtrar los nulos, deberiamos poner cada columna en su tipo ideal.

# df['antiguedad'] = df['antiguedad'].astype(int)     algo de este estilo



print("Filas, columnas: ", df.shape)

print("Tipos: \n", df.dtypes)
# Cosas a observar de esto:

# Propiedades sin ciudad ni provincia.

# Muchas propiedades sin metroscubiertos y totales especificados. Estas columnas son claves para 

# el precio (falta demostrar)

# 46 propiedades sin clasificacion de tipo de propiedad. 

# Habitaciones y Banos, muchos datos nulos. Sera que no tienen? Una casa no pude no tener habitacio ni bano...

# Garages, piscina, salondeusosmultiples y gimnasio son extras. Supongo que el hecho de que no aparezcan 

# es que no los tienen.

# Estos 4 extras, podrian reunirse en una categoria todas juntas para evaluar como crece el precio en base a esos

# extras.

# 43000 propiedades sin saber su antiguedad. Esto si ni idea como podemos reemplazarlos.



df_nulos = df.isnull().sum().to_frame().reset_index().rename(columns={'index':'Columna',0:'Cantidad de nulos'})

df_nulos = df_nulos[df_nulos['Cantidad de nulos']>200].sort_values(by='Cantidad de nulos')



plt.figure(figsize=(15,10))

ax = sns.barplot(data=df_nulos, x='Columna', y='Cantidad de nulos')

#ax.set(xlabel='\n Columna ', ylabel='Cantidad \n')

ax.set_xlabel('Columna \n', fontsize=18)

ax.set_ylabel('\n Cantidad', fontsize=18)

ax.set_title('Cantidad de nulos por columna', fontdict={'fontsize':20})

df_nulos = df.isnull().sum().to_frame().reset_index().rename(columns={'index':'Columna',0:'Cantidad de nulos'})

df_nulos
# Dato importante y valioso: No hay ninguna casa que no tenga especificado ni metros cubiertos ni totales

# Esto me asegura que no tengo que limpiar datos, ya que si no hubiese tenido ninguno hubiese sido dato invalido

# Conociendo uno de los dos valores, puedo intentar asumir el otro con alguna decision.

df[(df['metroscubiertos'].isnull()) & (df['metrostotales'].isnull())].head()
# Tiene sentido que los terrenos no tengan metros cubiertos. 

# Pero como una casa, un edificio, y cualquier otra cosa que no sea un terreno no tiene metros cubiertos? 

# Y ya demostre en uno de los graficos que los metros cubiertos y totales son lo que mas influyen en el precio. 

# Esto, a mi criterio, convierte estos datos en invalidos.

df_nulos_metroscub_tipoprop = df[df.metroscubiertos.isnull()]['tipodepropiedad'].value_counts().to_frame().reset_index().rename(columns={'index':'Tipo de propiedad', 'tipodepropiedad': 'Cantidad de nulos'})

df_nulos_metroscub_tipoprop = df_nulos_metroscub_tipoprop[df_nulos_metroscub_tipoprop['Cantidad de nulos']>100]



df_nulos_metroscub_tipoprop['Tipo de propiedad'].cat.remove_unused_categories(inplace=True)



df_nulos_metroscub_tipoprop.sort_values(by='Cantidad de nulos', inplace=True)



df_nulos_metroscub_tipoprop.reset_index(inplace=True)



plt.figure(figsize=(15,10))

ax = sns.barplot(data=df_nulos_metroscub_tipoprop, x='Tipo de propiedad', y='Cantidad de nulos')

ax.set(xlabel='\n Tipo de propiedad ', ylabel='Cantidad \n')

ax.set_xlabel('\n Tipo de propiedad ', fontsize=18)

ax.set_ylabel('Cantidad \n', fontsize=18)

ax.set_title('Cantidad de metros cubiertos nulos por tipo de propiedad', fontdict={'fontsize':20})
# Extraigo todos aquellas propiedades que tienen metroscubiertos nulos y no son terrenos. 

# Quizas en vez de extraerlos, se podria asumir que los metros totales son todos cubiertos.

indices_invalidos = df[(df.metroscubiertos.isnull()) & (~df.tipodepropiedad.isin(['Terreno','Terreno comercial','Lote']))].index

df.drop(index=indices_invalidos, inplace=True)

df.shape
# No tiene sentido que una casa no tenga ciudad ni provincia, ya que son datos que influyen en el precio

# y no se pueden deducir de otro lado, ademas, son pocos

df.dropna(subset=['ciudad','provincia'], inplace=True)

df.shape
# Es imposible que cualquier inmueble tenga metroscubiertos, y no tenga metros totales.

# Para esos casos, le asigno la misma cantidad de metros cubiertos que totales, osea no tiene metros descubiertos.

df['metrostotales'].fillna(df['metroscubiertos'], inplace=True)
# Despues de estos pequeños arreglos, se reducen la cantidad de nulos. Si bien siguen habiendo, 

df.isnull().sum()
# El resto de filas con metros cubiertos, son terrenos. Deberian tener 0 metros cubiertos.

df['metroscubiertos'].fillna(0, inplace=True)
# Aquellas casas que no tienen el tipodepropiedad, son solo 19. Podria intentar sacar el tipo de prop

# desde la descripcion pero no estaria seguro al respecto. Siendo solo 19, las elimino.

df.dropna(subset=['tipodepropiedad'],inplace=True)
df.shape
# Para banos y habitaciones, completo los nulos con la moda agrupando por tipo de propiedad y ciudad.

# En caso de que no este en ninguno de esos, le asigno la moda por tipo de propiedad.



df['banos'] = df.groupby(['tipodepropiedad','ciudad'])['banos'].transform(lambda x: x.fillna(x.mode()))

df['habitaciones'] = df.groupby(['tipodepropiedad','ciudad'])['habitaciones'].transform(lambda x: x.fillna(x.mode()))

df['banos'] = df.groupby(['tipodepropiedad'])['banos'].transform(lambda x: x.fillna(x.mode()))

df['habitaciones'] = df.groupby(['tipodepropiedad'])['habitaciones'].transform(lambda x: x.fillna(x.mode()))
df.isnull().sum()
# 17964 registros sin banos ni habitaciones completados. Analizo eso:

# Hay alguno con ambos nulos?

df[(df.banos.isnull())&(df.habitaciones.isnull())].shape

# Esos 17964 coinciden.
# Registros sin baño agrupados por tipo de propiedad.

df[df.banos.isnull()].groupby('tipodepropiedad').size()
# Causalmente, todos esos nulos corresponden a inmuebles no habitables. Es raro suponer que no tienen baño, pero asumimos

# que ese None, es algo que no modifica el valor del inmueble. No cambia para un local su valor si tiene o no baño.

df[df.habitaciones.isnull()].groupby('tipodepropiedad').size().sum()
# La idea es ver si alguno que no sea habitable, tiene baños distinto de None para ver como reemplazar los otros nan.

# Por habitable me refiero a propiedades donde vive gente.



habitables = ['Casa','Apartamento','Casa en condominio','Casa uso de suelo','Quinta Vacacional','Villa','Duplex','Rancho','Departamento Compartido']



df[~df.tipodepropiedad.isin(habitables)]['habitaciones'].value_counts()
# Tomo la decision de llenar los banos y habitaciones nulos, ya demostrado que corresponden a inmuebles no habitables,

# con el valor 0. Al margen de llenar los nulos, clarmaente son datos que no van a interesar para el analisis de este tipo de 

# propiedades.

df.banos.fillna(0, inplace = True)

df.habitaciones.fillna(0, inplace = True)
df.isnull().sum()
# Ahora voy a analizar los garages nulos. 

df[df.garages.isnull()].groupby('tipodepropiedad').size()
# De aca se puede ver informacion importante: Segun la ciudad y el tipo de propiedad, varia bastante la cantidad de garages

# de una casa.

df.groupby(['tipodepropiedad','ciudad']).agg({'garages':'mean'})
# Tomo la decision de completar aquellos valores de garage nulos, con el promedio de garages para ese tipo de propiedad,

# en la ciudad en que se encuentre.

df['garages'] = df.groupby(['tipodepropiedad','ciudad'])['garages'].transform(lambda x: x.fillna(x.mode()))



df['banos'].value_counts()
# Para aquellos registros que registraban de promedio nan, osea ninguno tenia, los relleno con 0 como valor de garage.

df['garages'].fillna(0, inplace=True)
# Solo me queda revisar los nulos de antiguedad. Son muchisimos. 

df.isnull().sum()
# La decision que voy a tomar para esos nulos, es similar a la de los garages. Voy a rellenar los nulos con el promedio

# de antiguedad para ese tipo de propiedad y esa ciudad en la que esta ubicada la propiedad.

# No se que tanto sentido tenga asignarle valores asi a la antiguedad porque de verdad estoy modificando de una forma que

# no necesariamente sea correcta





# df.groupby(['tipodepropiedad','ciudad']).agg({'antiguedad':'mean'})
# df['antiguedad'] = df.groupby(['tipodepropiedad','ciudad'])['antiguedad'].transform(lambda x: x.fillna(x.mean()))
df.isnull().sum()
# Se puede ver mucha informacion importante aca. 

# Por ejemplo, la diferencia abismal que hay entre el tercer cuantil y el maximo de precio.

# Deberia seguir mirando un poco mas

df.describe()
# Antes de empezar a analizar los datos, deberia intentar detectar outliers en cuanto al precio.

print('El percentil 0.99: ',df.precio.quantile(0.99))

print('El percentil 0.01: ',df.precio.quantile(0.01))

print('El maximo: ', df.precio.max())

print('El minimo: ', df.precio.min())

print('Maximo / Percentil 0.99', df.precio.max()/df.precio.quantile(0.99))

print('Percentil 0.01 / Minimo', df.precio.quantile(0.01)/df.precio.min())

print('Cantidad de registros por debajo del percentil 0.01: ', df[df['precio']<df.precio.quantile(0.01)].shape)

print('Cantidad de registros por encima del percentil 0.99: ', df[df['precio']>df.precio.quantile(0.99)].shape)
# Un boxplot ayuda a poder detectar estos outliers.



df['precio_m2'] = df['precio']/df['metrostotales']



def plot_outliers(df):

    fig, ax = plt.subplots(figsize=(15,10))

    green_diamond = dict(markerfacecolor='g', marker='D')

    df_g1 = df.copy()

    df_g1 = df[df['tipodepropiedad'].isin(['Casa','Apartamento','Edificio','Casa en condominio','Duplex','Terreno','Terreno comercial'])]

    df_g1.tipodepropiedad.cat.remove_unused_categories(inplace=True)

    df_g1.boxplot(column='precio_m2', by='tipodepropiedad', ax=ax, flierprops=green_diamond)

    ax.set_title('Precio por tipo de propiedad', fontdict={'fontsize':20})

    fig.suptitle('')

    ax.set_xlabel('\n Tipo de propiedad', fontsize=18)

    ax.set_ylabel('Precio m2\n', fontsize=18)

    

plot_outliers(df)
# Filtro los outliers, usando maximos y minimos en base al cuantil 1 y 3 y el IQR.

print('Antes de filtrar: ', df.shape)



def is_outlier(group):

    Q1 = group.quantile(0.25)

    Q3 = group.quantile(0.75)

    IQR = Q3 - Q1

    precio_min = Q1 - 1.5 * IQR

    precio_max = Q3 + 1.5 * IQR

    return ~group.between(precio_min, precio_max)



df = df[~df.groupby('tipodepropiedad')['precio_m2'].apply(is_outlier)]



print('Despues de filtrar: ', df.shape)
# Vuelvo a graficar, para ver como cambiaron los outliers.

plot_outliers(df)
#Lleno los nulos de antiguedad con 0. Supongo que son todos nuevos.

df['antiguedad'].fillna(0, inplace=True)



# Quedaron solo 517 con antiguedad nula. Los reviso:

# df[df['antiguedad'].isnull()]



# Para estos restantes, tomo la decision de rellenarlos de acuerdo a el promedio de antiguedad segun el tipo de prop y la prov



# df['antiguedad'] = df.groupby(['tipodepropiedad','provincia'])['antiguedad'].transform(lambda x: x.fillna(x.mean()))
# df.isnull().sum()
# Los restantes, los elimino ya que son pocos y no puedo fijarle una antiguedad de forma tan directa

# df.dropna(subset=['antiguedad'], inplace=True)
df.isnull().sum()
# Hay 70000 filas en las que los metros totales son menores a los cubiertos. Esto es invalido. La realidad es que son muchas

# filas, por lo tano les seteo como metros totales, la misma cantidad que cubiertos.



df.loc[df['metrostotales']<df['metroscubiertos'], 'metrostotales'] = df['metroscubiertos']

# Aca deberia ir seteando las columnas nuevas que me parezcan utiles para analisis.

# Ideas: Precio por metro total, o agregar una columna con la cantidad de extras que tenga la casa.

df['precio_dolar'] = df['precio']/VALOR_CAMBIO_A_DOLAR

df['precio_m2'] = df['precio']/df['metrostotales']

df['extras'] = df['garages']+df['piscina']+df['usosmultiples']+df['gimnasio']

df[df.tipodepropiedad=='Lote']['metroscubiertos']
df[df.ciudad.isnull()].tipodepropiedad.value_counts()
df[df.metroscubiertos.isnull()].tipodepropiedad.value_counts()
df_terrenos = df[df.tipodepropiedad =='Local Comercial']

df_terrenos[df.metroscubiertos.isnull()][['metrostotales','metroscubiertos']]
df.shape