# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print("Hola Mundo!!!!")
soy_una_variable = 5

soyOtraVariable = 10

mira1me = "hola hola"

foo = 5

bar = 10

print(foo + bar)
soy_un_string = "soy un string "

soy_un_string
soy_una_lista = [1,2,3,4]

soy_una_lista
soy_una_tupla = (1,2,3,4)

soy_una_tupla
soy_un_diccionario = {'Nombre':'Marco', 'Edad':25}

soy_un_diccionario
""" Ejemplo """

#ingrese un numero 

numero = int(input(" "))



if numero % 2 == 0:

    print("El numero > ", numero, " es par")

else:

    print("El numero > ", numero, "es impar")
""" ejemplo """



#Escribe tu nombre para ver la magia



nombre = input("")

for i in range(1, 10):

    print(i, "Hola: ", nombre)

    
""" ejemplo """

num = int(input(" "))



factorial = 1

for i in range(1, num+1):

    factorial = factorial *i

print("El factorial de", num, "es", factorial)
""" ejemplo """



nombre = input("")

i = 1

while i <=10:

    print(i, "", nombre)

    i+=1
#Definicion sin retorno

def suma(x,y):

    print("La suma es", x+y)



#llamado de la funcion

suma(5,5)
#Definicion con retorno

def suma(x,y):

    return x + y

resultado = suma(5,5)



print("La suma es > ", resultado)
"""Escribe tu codigo aqui"""
#Importando el modulo statistics

import statistics as stats



#creando una lista de numeros

edades = [22, 23, 26, 26, 26, 34, 34, 38, 40, 41]

print(stats.mean(edades))

print(stats.median(edades))

print(stats.mode(edades))

print(stats.variance(edades))

print(stats.stdev(edades))
import csv

# Abrir archivo csv

archivoCSV=open("/kaggle/input/municipios2/municipios.csv")

# Leer todos los registros

entrada = csv.reader(archivoCSV)

# Cada línea se muestra como una lista de campos

for registro in entrada:

    print(registro)
"""Tu codigo aqui"""
import matplotlib.pyplot as plt

lista1 = [9,4,3,14,7,11,20,31]

plt.plot(lista1)

plt.show()
#Titulos y etiquetas

lista1 = [9,4,3,14,7,11,20,31]

plt.plot(lista1)

plt.title("Título")

plt.xlabel("Abscisa")

plt.ylabel("Ordenada")

plt.show()
lista1 = [9,4,3,14,7,11,20,31]

plt.plot(lista1)

plt.title("Título")

plt.xlabel("abscisa")

plt.ylabel("ordenada")

#Rótulos eje X

indice = [0,1,2,3,4,5,6,7];

plt.xticks(indice, ["A", "B", "C", "D", "E", "F", "G", "H"])

#Rótulos eje Y

plt.yticks([0,10,20,30,40])

plt.show()


alumnos = ['Hugo', 'Paco', 'Luis']

df = pd.DataFrame(alumnos)

print (df)
alumnos = {

'Nombre' : ['Hugo', 'Paco', 'Luis'],

'Edad' : [10, 11, 12]

}

df = pd.DataFrame(alumnos)

print (df)
alumnos = {

'Nombre' : ['Hugo', 'Paco', 'Luis'],

'Edad' : [10, 11, 12]

}

df = pd.DataFrame(alumnos, columns = ['Nombre',

'Edad', 'Peso'])

print (df)
alumnos = {

'Nombre' : ['Hugo', 'Paco', 'Luis'],

'Edad' : [10, 11, 12],

'Peso' : [20, 21, 19]

}

df = pd.DataFrame(alumnos, columns = ['Nombre',

'Peso'])

print (df)
municipios = pd.read_csv("/kaggle/input/municipios2/municipios.csv")

print (municipios)
df = pd.read_csv("/kaggle/input/municipios2/municipios.csv")

#Primeros 3

print (df.head(3))

#Ultimos 2

print("\n")

print (df.tail(2))
#podemos realizar funciones estadisticas de manera mas sencilla.

df = pd.read_csv("/kaggle/input/municipios2/municipios.csv")

media = df["poblacion"].mean()

mediana = df["poblacion"].median()

std = df["poblacion"].std()

max = df["poblacion"].max()

min = df["poblacion"].min()

print("Media", media)

print("Mediana", mediana)

print("Desviacion Standar", std)

print("Min", min)

print("Max", max)
df = pd.read_csv("/kaggle/input/dolares/dolar.csv")

meses = df['Mes']

precios1 =df['2010']

precios2 =df['2017']

precios1.index =np.arange(1,13)

precios2.index =np.arange(1,13)

plt.plot(precios1, label = precios1.name, marker='x', linestyle=':', color='b')

plt.plot(precios2, label = precios2.name, marker='o', linestyle='--', color='r')

plt.legend(loc="upper left")

plt.title("Variación del precio del dolar durante el "+ precios1.name + " y " +

precios2.name)

plt.xlabel("Mes")

plt.ylabel("Precio")

plt.xticks(precios1.index)

plt.yticks([12,13,14,15,16,17,18,19])

plt.grid(True)

plt.show()