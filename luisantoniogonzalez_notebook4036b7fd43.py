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
20+40

55 - 9*90
(5 - 10*90) / 60
77 / 10
809 / 20
507 // 32
50 % 11
67 * 2 + 80
9 ** 2
5 ** 8
x = 10

y = 50 * 2

x * y
m = 10

m*n
6 * 4.5 - 1

n = 3500 / 450

m = 200.30

n * m 

n + m

round(m, 2)





'naranjas y manzanas'
'did\'t'
"won't"
'"no," esta aqui.'
"\"Si,\" por ahí."
'"isn\'t," we go.'
'"Isn\'t," he eats.'

print('"Isn\'t," he eats.')

respuesta = 'no, el no fue.\nsi, fue el.' 

respuesta
'"Isn\'t," he eats.'

print('"Isn\'t," he eats.')

respuesta = 'no, el no fue.\nsi, fue el.'

print(respuesta)
print('C:\ningun\nacho')
print(r'C:\ningun\nacho')
print('''



lista del mercado: walmart



-agua            -pan

-leche           -zanahorias





''')
'do'+'la'* 4
'water'+ 'melon'
sal = 'agua'

sal + 'cate'
refran = ('camaron que se duerme '

         'se lo lleva la corriente')

refran
letra = ('nominal')

letra [4]
valor = ('sanadia')

valor [-2]
letra = ('filo')

letra [0:3]
palabra = ('xilofono')

palabra[:3] + palabra[3:]
palabra = ('xilofono')

palabra [-2:]
palabra = ('jojo')

palabra [2:50]
animal = ('lefante')

'e'+ animal [0:]
pais = 'estadosunidosmexicanos'

len (pais)
cadena = [1, 3, 2, 4, 5, 7, 8, 9]

cadena [5:]
cadena = [1, 3, 2, 4, 5, 7, 8, 9]

cadena [:]
fila = [1, 3, 2, 4, 5, 7, 8, 9]

fila + [10, 11, 12, 13]
fila = [2, 4, 5, 7, 8, 9]

fila [0] = 3

fila
fila = [2, 3, 4, 5, 6, 7]

fila.append(8)

fila
vocales = ['a', 'e', 'i', 'o', 'u']

vocales

vocales = ['a', 'e', 'i', 'o', 'u']

vocales [3:5] = ['O', 'U']

vocales
vocales = ['a', 'e', 'i', 'o', 'u']

vocales [0:2] = []

vocales
vocales = ['a', 'e', 'i', 'o', 'u']

vocales [:] = []

vocales
claves = ['do', 're', 'mi', 'fa', 'sol', 'la', 'si', 'do']

len (claves)
z = ['yo', 'ye', 'yi']

x = [45, 46, 47, 48]

c = [z, x]

c
z = ['yo', 'ye', 'yi']

x = [45, 46, 47, 48]

c = [z, x]

c [1]
z = ['yo', 'ye', 'yi']

x = [45, 46, 47, 48]

c = [z, x]

c [1][0]