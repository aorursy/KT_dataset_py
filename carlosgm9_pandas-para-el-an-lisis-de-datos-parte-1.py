import pandas as pd

import numpy as np
# Conjuntos de datos con numero aleatorios

datos_1 = np.random.rand(20)

datos_2 = np.random.rand(20)



# Esta función nos ayuda a crear los índices del conjunto de datos

alumnos = []

for i in range(20):

    c = 'alumno_' + str(i)

    alumnos.append(c)
data = pd.DataFrame(

    {

        'Economía' : datos_1,

        'Administración' : datos_2,

    }, index = alumnos

)
# Muestra la cantidad de filas y columnas de un conjunto de datos

data.shape
# Retorna los primeros 5 resultados

data.head()
data.loc['alumno_0']
data.iloc[2]
# Es posible elegir un rango

data.iloc[2:4]
# También puedes elegir ciertas filas

data.iloc[[2,8]]
# Filtros condicionales

filtro1 = data.Economía > .5

filtro1.head()
# Retorna solo las columnas que son mas grandes al 50%

data[data.Economía > .50]
# Los datos que se van a cargar son los valores de la criptomoneda Ripple

path = '../input/d01_Ripple.csv'



data_ripple = pd.read_csv(path, index_col='Fecha')

data_ripple.head()
data_ripple.shape
print('Este conjunto de datos contiene {} filas y {} columnas'.format(data_ripple.shape[0], data_ripple.shape[1]))
data_ripple.index
serie = pd.Series([1,2,3,4,5])

serie
serie[0:3]
# Puedes retornar filas especificas por medio de una lista

serie[[2,4]]
# Crear una serie con índex personalizado

serie_1 = pd.Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])

serie_1
# Se puede imprimir por el nombre del índex

serie_1[['a', 'c']]
serie_1.index
fechas = pd.date_range('2017-09-15', '2017-09-19')

fechas
# Identamos la fecha al index y agregamos valores

temps = pd.Series([80,90,56,34,23], index = fechas)

temps
# Podemos ver datos de "x" fecha

temps['2017-09-17']
# Es posible hacer operaciones matemáticas en las mismas series

temps2 = pd.Series([5,5,5,5,5], index = fechas)

temps2
temps_sum = temps + temps2

temps_sum
# Pandas también provee funciones estadísticas.

temps_sum.mean()
temps_sum.sum()
df = pd.DataFrame(

    {

        'Mexico': temps,

        'Canada': temps2,

    }

)
df
df['Mexico']
pais = df['Mexico'].tolist()

type(pais)
type(df['Mexico'])
type(df)
estado_1 = 'Mexico'

estado_2 = 'Canada'



df[[estado_2, estado_1]]
# Si el nombre de la columna no tiene espacios, es posible ingresar a las propiedades de la columna de la siguiente manera:

df.Mexico
# También se puede usar de la siguiente manera:

df['NuevaColumna'] = df['Mexico'] * 3

df
# Para accesar a las columnas

df.columns
df.NuevaColumna[:3]
df.iloc[0:2]
df.loc['2017-09-15']
nombres = ['Carlos', 'Jose', 'Ivan', 'Gerardo', 'Manuel']

serie = pd.Series(nombres)

serie
# Con el multiplicador fuera de la lista es igual a decirle a pandas: "Repite esta función 3 veces"

pd.Series([2+3]*3)
# Se puede usar cada carácter de una lista como valor

pd.Series(list('hola mundo'))
# En el caso de los diccionarios, las claves de los diccionarios son las etiquetas.

pd.Series(

    {

        'Nombre' : 'Carlos',

        'Apellido': 'Lopez',

        'Edad': 23,

    }

)
# Puedes multiplicar un valor como si fuera una iteración

x = pd.Series(np.arange(0, 5))

x * 5
s = pd.Series([11,22,33,44,55])
s.values
s.index
# longitud de la serie

len(s)
s.size
calificaciones = [.99, .96, .78, .79, .89]

alumnos = ['Alejandro', 'Carlos', 'José', 'Arturo', 'Marco']



al = pd.Series(calificaciones, index=alumnos)

al
al.index
al['Carlos']
al.head(2)
al.tail(3)
al.take([1,4])
al['Carlos']
al.Carlos
# Es posible buscar una lista de índices

al[['Alejandro', 'Carlos']]
# También es posible buscar por posición

al[[0,3]]
al
al.iloc[3]
al.loc['Carlos']
al.loc['Carlos'] == al.Carlos == al[1]
al.loc['Carlos'] == al.Carlos == al[0]  
al.iloc[[0,2]]
al[1]
al.iloc[1] == al[1]
sl = pd.Series(np.arange(100, 120), index = np.arange(10,30))

sl
sl[0:10:2]
sl[10::3]
s1 = pd.Series([1,2], index = ['a','b'])

s2 = pd.Series([3,4], index = ['b','a'])
# Se suman los valores iguales del índex

s1 + s2
s1 * s2
s3 = pd.Series(3, index= s1.index)

s3
s4 = pd.Series(3, index = ['a', 'b', 'c'])

s4
# Retorna un NaN cuando multiplicamos un valor de la serie por otro que no existe

s3 * s4
# ¿Qué pasa si tenemos mas de un index igual?



a = pd.Series([10,23,43], index = ['a', 'a', 'b'])

a
b = pd.Series([15,3,4,57], index = ['a', 'a', 'b', 'c'])

b
a + b
s = pd.Series(np.arange(0, 5), index = list('abcde'))

logica = s >= 3

logica
s[logica]
# Hay técnicas que ayudan a constrir lógica para los filtros

s[(s >= 2) & (s < 4)]
# Con la función `.all()` puedes preguntarle a pandas si todos los valores cumplen con una condicion

(s >= 0).all()
# Función any()

s[s < 5].any()
# Función .sum()

(s >= 3).sum()
np.random.seed(123456)

s = pd.Series(np.random.rand(5), index = list('abcde'))

s
np.random.seed(123456)

s1 = pd.Series(np.random.rand(4), list('abcd'))

s1
s2 = s1.reindex(list('awxy'))

s2
# Es posible cambiar el tipo de objeto en una serie índex

c1 = pd.Series([10,20,30], index = [1,2,3])

c2 = pd.Series([100,200,300], index = ['1', '2', '3'])

c1 + c2
c2.index = c2.index.values.astype(int)

c1 + c2