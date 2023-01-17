# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
list1 = [11,2,3,15,8,13,21,34]
plt.plot(list1)
plt.title('tittle')
plt.xlabel('abscissa')
plt.ylabel('orderer')
list1 = [11,2,3,15,8,13,21,34]
plt.plot(list1)
plt.title('tittle')
plt.xlabel('abscissa')
plt.ylabel('orderer')
plt.ioff()
list2=[2,3,4,2,3,6,4,10]
plt.plot(list2)
plt.ion()
plt.plot(list2)
plt.ioff()  
list3 = [9,15,9,15,9,15,9,15]  
plt.plot(list3) 
plt.show()  
plt.title("Gráfica") 
plt.show() 
plt.grid(True) 
plt.show() 
plt.ion() 


plt.plot(list1, label = "Enero")
plt.plot(list2, label = "Febrero")
plt.plot(list3, label = "Marzo")
plt.legend()
plt.plot(list1, label = "Enero")
plt.plot(list2, label = "Febrero")
plt.plot(list3, label = "Marzo")
plt.legend(loc='upper left')
plt.plot(list1, label = "Enero")
plt.plot(list2, label = "Febrero")
plt.plot(list3, label = "Marzo")
plt.legend(loc = 'upper right')
plt.plot(list1, label = "Enero")
plt.plot(list2, label = "Febrero")
plt.plot(list3, label = "Marzo")
plt.legend(loc='center')
plt.plot(list1, marker='x', linestyle=':', color='b', label = "Enero")
plt.plot(list2, marker='*', linestyle='-', color='g', label = "Febrero")
plt.plot(list3, marker='o', linestyle='--', color='r', label = "Marzo")
plt.legend(loc="upper left")
plt.figure()  # Comenzamos un nuevo gráfico (figura)
list1 = [11,2,3,15,8,13,21,34]
plt.title("Título")
plt.xlabel("abscisa")
plt.ylabel("ordenada")
indice = np.arange(8)   # Declara un array
plt.xticks(indice, ("A", "B", "C", "D", "E", "F", "G", "H"))  
plt.yticks(np.arange(0,51,10))
plt.plot(list1)
paises = ("Alemania", "España", "Francia", "Portugal")
posicion_y = np.arange(len(paises))
unidades = (342, 321, 192, 402)
plt.barh(posicion_y, unidades, align = "center")
plt.yticks(posicion_y, paises)
plt.xlabel('Unidades vendidas')
plt.title("Ventas en Europa")
datos = [[1, 2, 3, 4], [3, 5, 3, 5], [8, 6, 4, 2]]
X = np.arange(4)
plt.bar(X + 0.00, datos[0], color = "b", width = 0.25)
plt.bar(X + 0.25, datos[1], color = "g", width = 0.25)
plt.bar(X + 0.50, datos[2], color = "r", width = 0.25)
plt.xticks(X+0.38, ["A","B","C","D"])