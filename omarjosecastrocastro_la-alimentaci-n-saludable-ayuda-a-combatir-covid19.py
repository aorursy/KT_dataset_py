# SE IMPORTAN LOS DATOS DE LA CANTIDAD DE SUMINISTRO DE ALIMENTOS (En KG)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # importar biblioteca de visualización de datos de Python basada en matplotlib. 

import operator

import missingno #libreria usada para graficar la completitud de los datos

import statsmodels.api as sm





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

train = pd.read_csv("../input/dieta-covid19-8-2/Dieta_Covid19_8_2.csv") # Asignación de a la variable el .csv
# se revisa los primeros 5 registros de la data para ver los datos

train.head()
#Validamos la completitud de los datos



missingno.matrix (train, figsize = (30,10))
#Se elimino la columna Unit ya que todos los registros decian %

#Se realizó el cambio de puntos por comas

#se ajustan los porcentajes a 6 pociciones decimales y se multiplican por 100

#se solucionaron problemas de valores nulos y datos atipicos
#Datos basicos del archivo

print('Cantidad de Filas y columnas:',train.shape)

print('Nombre columnas:',train.columns)
# vemos los tipos de cada uno de los datos y se valida si son nulos o no. 

train.info()
# Se busca los tipos de datos numéricos que se tiene en el dataset



list(set(train.dtypes.tolist()))
#como ya se tienen los datos numéricos  se muestran los campos que son numéricos 



train_num = train.select_dtypes(include = ['float64', 'int64'])

train_num.head()
# Se realiza una descripción estadística de los datos numéricos

train_num.describe()
#Verificamos si hay correlacion entre las variables principales

corr = train.set_index('Deaths').corr()

sm.graphics.plot_corr(corr, xnames=list(corr.columns))

plt.figure(figsize=(1000,900))
#Verificamos si hay correlacion entre los datos

corr = train.set_index('Obesity').corr()

sm.graphics.plot_corr(corr, xnames=list(corr.columns))

plt.figure(figsize=(1000,900))
#Verificamos si hay correlacion entre los datos

corr = train.set_index('Recovered').corr()

sm.graphics.plot_corr(corr, xnames=list(corr.columns))

plt.figure(figsize=(1000,900))
# Se grafica todos los campos de tipos numéricos. 

train_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
#Ahora intentaremos encontrar qué características están cocorrelacionadas con los fallecidos. 

#Los almacenaremos en una var llamada relacionfallecidos .



train_num_corr = train_num.corr()['Deaths'][:-1] # -1 because the latest row is Age

relacionfallecidos = train_num_corr[abs(train_num_corr) > 0.1].sort_values(ascending=False)

print("Hay {} valores correlacionados con fallecidos:\n{}".format(len(relacionfallecidos), relacionfallecidos))
#Ahora intentaremos encontrar qué características están cocorrelacionadas con la obesidad. 

#Los almacenaremos en una var llamada relacionobesidad .



train_num_corr = train_num.corr()['Obesity'][:-1] # -1 because the latest row is Age

relacionobesidad = train_num_corr[abs(train_num_corr) > 0.1].sort_values(ascending=False)

print("Hay {} valores correlacionados con Obesidad:\n{}".format(len(relacionobesidad), relacionobesidad))
#Ahora intentaremos encontrar qué características están corelacionadas con los recuperados. 

#Los almacenaremos en una var llamada relacionrecuperados .



train_num_corr = train_num.corr()['Recovered'][:-1] # -1 because the latest row is Recovered

relacionrecuperados = train_num_corr[abs(train_num_corr) > 0.1].sort_values(ascending=False)

print("Hay {} valores correlacionados con recuperados:\n{}".format(len(relacionrecuperados), relacionrecuperados))
#Ahora intentaremos encontrar qué características están correlacionadas con los obesos. 

#Los almacenaremos en una var llamada relacionobesos .



train_num_corr = train_num.corr()['Obesity'][:-1] # -1 because the latest row is Obesity

relacionobesos = train_num_corr[abs(train_num_corr) > 0.1].sort_values(ascending=False)

print("Hay {} valores correlacionados con obesos:\n{}".format(len(relacionobesos), relacionobesos))
#Ahora intentaremos encontrar qué características están correlacionadas con las personas con desnutricion. 

#Los almacenaremos en una var llamada relaciondesnutricion .



train_num_corr = train_num.corr()['Undernourished'][:-1] # -1 because the latest row is Undernourished

relaciondesnutricion = train_num_corr[abs(train_num_corr) > 0.1].sort_values(ascending=False)

print("Hay {} valores correlacionados con desnutricion:\n{}".format(len(relaciondesnutricion), relaciondesnutricion))
# Se relaciona el campo Muerte con los otros campos numéricos. 



for i in range(0, len(train_num.columns), 5):

    sns.pairplot(data=train_num,

                x_vars=train_num.columns[i:i+5],

                y_vars=['Deaths'])
# Se relaciona el campo recuperados con los otros campos numéricos. 



for i in range(0, len(train_num.columns), 5):

    sns.pairplot(data=train_num,

                x_vars=train_num.columns[i:i+5],

                y_vars=['Recovered'])
# Se relaciona el campo obecidad con los otros campos numéricos. 



for i in range(0, len(train_num.columns), 5):

    sns.pairplot(data=train_num,

                x_vars=train_num.columns[i:i+5],

                y_vars=['Obesity'])
# Se relaciona el campo desnutricion con los otros campos numéricos. 



for i in range(0, len(train_num.columns), 5):

    sns.pairplot(data=train_num,

                x_vars=train_num.columns[i:i+5],

                y_vars=['Undernourished'])
corr = train_num.drop('Deaths', axis=1).corr() # We already examined Deaths correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
corr = train_num.drop('Recovered', axis=1).corr() # We already examined Recovered correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
corr = train_num.drop('Obesity', axis=1).corr() # We already examined Obesity correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
dataX =train[['Deaths']]

X_train = np.array(dataX)

y_train = train['Confirmed'].values



# Creamos el objeto de Regresión Linear

from sklearn import linear_model

regr = linear_model.LinearRegression()



# Entrenamos nuestro modelo

regr.fit(X_train, y_train)



# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)

y_pred = regr.predict(X_train)



# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente

print('Coefficients: \n', regr.coef_)

# Este es el valor donde corta el eje Y (en X=0)

print('Independent term: \n', regr.intercept_)

# Error Cuadrado Medio

print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))

# Puntaje de Varianza. El mejor puntaje es un 1.0

print('Variance score: %.2f' % r2_score(y_train, y_pred))
colores=['orange','blue']

tamanios=[30,60]



f1 = train['Deaths'].values

f2 = train['Confirmed'].values



asignar=[]

for index, row in train.iterrows():

    if(row['Deaths']>1):

        asignar.append(colores[0])

    else:

        asignar.append(colores[1])

    

plt.scatter(f1, f2, c=asignar, s=tamanios[0])

plt.show()



#Eje y se muestran los confirmados 

#Eje x se muestran los Muertos
import numpy

from matplotlib import pyplot

x = numpy.linspace(0, 2 * numpy.pi, 100)

f3 = pyplot.plot(x, (8.3532)*x+(2.966))

pyplot.plot(x, (8.3532)*x+(2.966))

pyplot.show()
import numpy

from matplotlib import pyplot



colores=['orange','blue']

tamanios=[30,60]



f1 = train['Deaths'].values

f2 = train['Confirmed'].values



asignar=[]

for index, row in train.iterrows():

    if(row['Deaths']>1):

        asignar.append(colores[0])

    else:

        asignar.append(colores[1])



x = numpy.linspace(0, 2 * numpy.pi, 100)

pyplot.plot(x, (8.3532)*x+(2.966), Color = 'Red')

plt.scatter(f1, f2, c=asignar, s=tamanios[0])

plt.show()

pyplot.show()







#Eje y se muestran los confirmados 

#Eje x se muestran los Muertos
## Predicción Si hay 2000 muerte cuantos contagiados se espera tener ? 

y_Dosmil = regr.predict([[2000]])

print(int(y_Dosmil))