# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def evaluacion(nota):
    valoración="aprobado"
    if nota<5:
        valoracion="suspenso"
    return valoracion

print(evaluacion(4))
print("Verificación de acceso")
edad_usuario= int(input("Introduce tu edad: "))

if edad_usuario<18:
    print("No puedes pasar")
elif edad_usuario>100:
    print("Edad incorrecta")
else:
    print("Puedes pasar")

print("Fin del programa")


for estaciones_año in ["primavera", "verano", "otoño", "invierno"]:
    print(estaciones_año)

for i in range(0, 10):
    print(i)
s = ['Han y Chew', 'son', 'amigos']
for i in range(len(s)):
    print(i, s[i])
lista = range(1,11)

for x in lista:
    if x == 9:
        break
    if x == 5:
        continue

    print("Número: ",x)
for i in range(1,10):
    if True:
        pass
    elif False:
        pass
    print(i)
def tabla_de_multiplicar (n):
    for i in range(1,11):
        print (n, '*', i, '=', i*n)
        
tabla_de_multiplicar(8) 

def suma(a,b):
    return a+b
respuesta = suma(50,180)
print(respuesta)
def saludo(nombre, mensaje='Hola, ¿cómo estas?'):
    print(mensaje, nombre)
    
saludo('Luke')
def saludar(nombre, mensaje='Hola'): 
    print(mensaje, nombre)

saludar(mensaje="Buen día", nombre="Juancho")
def parametros(fija, *arbitraria):
    print(fija)
    for x in arbitraria:
        print(x)
        
parametros("valor fijo", "valor1", "valor2", "valor3")
def sumar(sueldo, comision):
    return sueldo + comision
info = {"sueldo": 1000000, "comision": 100000}
print(sumar(**info))
area_triangulo = lambda base,altura:(base*altura/2)
print(area_triangulo(7,5))

print(area_triangulo(9,6))
cadena = "fuiste la mejor\nforma de perder\nmi tiempo"
print(cadena)
def saludo(nombre: str) -> None:
    print("Que tal " + nombre)
nombre = "Jorge"
saludo(nombre)