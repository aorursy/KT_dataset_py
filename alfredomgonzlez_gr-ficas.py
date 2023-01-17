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
import matplotlib.pyplot as plt
uno = np.arange(0,50,10)
uno
lista1 = [13,1, 5, 17, 6, 15, 31, 42]
plt.plot(lista1)
plt.title('Ventas en miles')
plt.xlabel('Valores')
plt.ylabel('Ventas')

plt.ioff()

lista3 = [2, 3, 4, 2, 3, 4, 6, 10]
plt.plot(lista3)
plt.ion
dos = np.arange(0, 210, 10)
plt.plot(lista2)
plt.subplot(1, 2, 1)
plt.plot((1, 2, 3, 4, 5))

plt.subplot(1,2,2)
plt.plot((5,4,3,2,1))


lista1 = [13,1, 5, 17, 6, 15, 31, 42]
lista2 = [2,1,4,8,6,5,13,7]
lista3 = [2, 3, 4, 2, 3, 4, 6, 10]
plt.plot(lista1, marker = 'o', linestyle = ':', color = 'g', label = 'Enero')
plt.plot(lista2, marker = '+', linestyle = ':', color = 'c', label = 'Febrero')
plt.plot(lista3, marker = 'x', linestyle = ':', color = 'b', label = 'Marzo')

plt.legend(loc = 'best')
plt.title('Ventas trimestrales')

plt.xlabel('Valores')
plt.ylabel('Miles de pesos')
plt.figure(figsize = (10,9), dpi = 80)  
Enero = [13,1, 5, 17, 6, 15, 31, 42]
Marzo = [2,1,4,8,6,5,13,7]
Febrero = [2, 3, 4, 2, 3, 4, 6, 10]
plt.plot(Enero, marker = 'o', linestyle = ':', color = 'g', label = 'Enero')
plt.plot(Febrero, marker = '+', linestyle = ':', color = 'c', label = 'Febrero')
plt.plot(Marzo, marker = 'x', linestyle = ':', color = 'b', label = 'Marzo')
indice= np.arange(8)
plt.xticks(indice,('A','B','C','D','E','F','G', 'H'))
plt.ysticks = np.arange(0, 101, 5)
plt.legend(loc = 'best')
plt.title('Ventas trimestrales')

plt.xlabel('Valores')
plt.ylabel('Miles de pesos')
paises = ('Alemania','España', 'Francia', 'Portugal')
posicion_y = np.arange(len(paises))
unidades_vendidas = (342, 234, 196, 356 )
plt.barh(posicion_y, unidades_vendidas, align = 'center')
plt.yticks(posicion_y, paises)
plt.xlabel('Unidades vendidas')
plt.title('Ventas en Europa')
paises_latam = ('México', 'Colombia', 'Chile', 'Costa Rica')
posiciony = np.arange (len(paises_latam))
vendido_latam = (456, 789, 526, 431)
plt.barh(posiciony, vendido_latam, align = 'center')
plt.yticks(posiciony, paises_latam)
plt.xlabel('Unidades vendidas')
plt.title('Ventas en Latinoamérica')

datos = [[1, 2, 3, 4], [2, 6, 3, 6], [8, 6, 3, 2]]
X = np.arange(4)
plt.bar(X + 0.00, datos[0], color= 'b', width = 0.25)
plt.bar(X + 0.25, datos[1], color= 'g', width = 0.25)
plt.bar(X + 0.50, datos[2], color= 'r', width = 0.25)
plt.xticks(X + 0.25, ['Ene', 'Feb', 'Mar', 'Abr'])
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
peras = [20, 10, 12, 34]
nombres = ['Hugo', 'Paco', 'Luis', 'Donald']
plt.pie(peras, labels = nombres)

tipos_imp = [b'/n', 'color', 'legal', 'doble carta']
vol_imp = [3456, 4500, 1350, 900]
expl = (0, 0.05, 0, 0)
plt.pie(vol_imp, explode = expl, labels =tipos_imp,autopct = '%1.2f%%', shadow = True)
plt.title('Volumen de Impresión Mensual')
plt.figure(figsize  = (15, 9))
plt.show()
import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
%matplotlib inline
 
# Generador de distribución de datos para regresión lineal simple
def generador_datos_simple(beta, muestras, desviacion):
    
    x = np.random.random(muestras) * 100
    e = np.random.randn(muestras) * desviacion
    y = x * beta + e
    return x.reshape((muestras,1)), y.reshape((muestras,1))
desviacion = 200
beta = 10
n = 50
x, y = generador_datos_simple(beta, n, desviacion)
plt.scatter(x, y)
plt.show()

modelo = linear_model.LinearRegression()
modelo.fit(x, y)
print( u'Coeficiente beta1: ', modelo.coef_[0])
y_pred = modelo.predict(x)
print( u'Error cuadrático medio: %.2f' % mean_squared_error(y, y_pred))
print (u'Estadístico R_2: %.2f' % r2_score(y, y_pred))

plt.scatter(x, y)
plt.plot(x, y_pred, color='green')
x_real = np.array([0, 100])
y_real = x_real*beta
plt.plot(x_real, y_real, color='blue')
plt.title('Energía consumida mensualmente')

plt.xlabel('Valor independiente')
plt.ylabel('Valor dependiente')
plt.show()
