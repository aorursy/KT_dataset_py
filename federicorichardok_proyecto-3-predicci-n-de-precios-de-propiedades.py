import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

path_dataset = '../input/datos_properati_limpios_model.csv'

df = pd.read_csv(path_dataset)
print("El dataset que vamos a trabajar aquí tiene {} observaciones".format(df.shape[0]))
df.head(5)

#de esta forma leo el dataframe para explorar un poco de que se trata
df.info()

#de esta forma me aseguro que no existen valores no numericos en el dataframe
X = df.drop(['price_aprox_usd'], axis=1)

y = df['price_aprox_usd']



# Realizá la separación a continuación en esta celda

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train.shape, \

y_train.shape, \

X_test.shape, \

y_test.shape



#De esta manera, simplemente corroboramos que la cantidad de Train y de Test tienen la misma geometria
# En esta celda cargá el regresor y realizá el entrenamiento

from sklearn.tree import DecisionTreeRegressor

from sklearn import tree



reg = DecisionTreeRegressor()

reg.fit(X_train, y_train)

# Acá realizá la predicción

y_pred = reg.predict(X_test)

y_pred

#aqui abajo se muestran los valores predichos para el conjunto de entrada X_test dando como resultado Y
# En esta celda calculá el rmse

from sklearn.metrics import mean_squared_error

import numpy as np





mse = mean_squared_error(y_test,y_pred)

print("RMSE entre valores de testeo y de prediccion: ", np.sqrt(mse))
max_depth_it = []

rmses_test= []

#for i in range(5,int(''.join(map(str, y_test.shape))), 5):

for i in range(1,30, 5):

    max_depth_it=max_depth_it+[i] #Este dato simplemente lo utilice para contar correctamente las iteraciones, es decir que vaya correctamente de 5 en 5

    reg = DecisionTreeRegressor(max_depth = i)

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test,y_pred)

    rmses_test= rmses_test + [np.sqrt(mse)]

max_depth_it = []

rmses_train= []

#for i in range(5,int(''.join(map(str, y_test.shape))), 5):

for i in range(1,30, 5):    

    max_depth_it=max_depth_it+[i] #Este dato simplemente lo utilice para contar correctamente las iteraciones, es decir que vaya correctamente de 5 en 5

    reg = DecisionTreeRegressor(max_depth = i)

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_train)

    mse = mean_squared_error(y_train,y_pred)

    rmses_train= rmses_train + [np.sqrt(mse)]
rmses_train=np.array(rmses_train)

rmses_test=np.array(rmses_test)

import matplotlib.pyplot as plt

%matplotlib inline 

plt.plot(range(1,30, 5), rmses_train, label='RMSE Training')

plt.plot(range(1,30, 5), rmses_test, label='RMSE Testing')

plt.ylim((0, 30000))

plt.legend(loc="best")

plt.title("RMSE Training vs RMSE Testing para árboles de decisión")

plt.show()
plt.plot(range(5,int(''.join(map(str, y_test.shape))), 5), rmses_train, label='RMSE Training')

plt.plot(range(5,int(''.join(map(str, y_test.shape))), 5), rmses_test, label='RMSE Testing')

plt.ylim((0, 30000))

plt.legend(loc="best")

plt.title("RMSE Training vs RMSE Testing para árboles de decisión")

plt.show()

#Aqui simplemente queria ver que pasaba si exageraba el max depth
X_train.shape, \

y_train.shape, \

X_test.shape, \

y_test.shape

#Aqui simplemente vuelvo a chequear que todo este correcto para proseguir.
# Realizá el entrenamiento y el cálculo de rmse en esta celda

from sklearn.neighbors import KNeighborsRegressor



neigh = KNeighborsRegressor(n_neighbors=1)

neigh.fit(X_train, y_train) 

y_pred = neigh.predict(X_test)

mse = mean_squared_error(y_test,y_pred)

print("RMSE entre valores de testeo y de prediccion: ", np.sqrt(mse))
# Calculá los cambio en el rmse en esta celda

n_neighbors_it = []

rmses_test= []



for i in range(1,30, 1):

    n_neighbors_it=n_neighbors_it+[i] #Este dato simplemente lo utilice para contar correctamente las iteraciones

    neigh = KNeighborsRegressor(n_neighbors= i)

    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)

    mse = mean_squared_error(y_test,y_pred)

    rmses_test= rmses_test + [np.sqrt(mse)]
n_neighbors_it = []

rmses_train= []



for i in range(1,30, 1):

    n_neighbors_it=n_neighbors_it+[i] #Este dato simplemente lo utilice para contar correctamente las iteraciones

    neigh = KNeighborsRegressor(n_neighbors=i)

    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_train)

    mse = mean_squared_error(y_train,y_pred)

    rmses_train= rmses_train + [np.sqrt(mse)]

    
plt.plot(range(1,30, 1), rmses_train, label='RMSE Training')

plt.plot(range(1,30, 1), rmses_test, label='RMSE Testing')

plt.ylim((0, 30000))

plt.legend(loc="best")

plt.title("RMSE Training vs RMSE Testing para KNN")

plt.show()
def nmsq2rmse(score):

    return np.sqrt(-score)
regressor = DecisionTreeRegressor(max_depth=5)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test,y_pred)

# Calculá en esta celda los cross_val_score

from sklearn.model_selection import cross_val_score

score = cross_val_score(regressor, X_train, y_train, scoring= "neg_mean_squared_error", cv=10)

print(nmsq2rmse(score).mean())

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

val_real = pd.Series(y_test.values)

val_pred = pd.Series(y_pred)
predicciones = pd.concat([val_real.rename('Valor real'),val_pred.rename('Valor Pred') ,abs(val_real-val_pred).rename('Dif(+/-)'),abs((val_real-val_pred)/(val_pred)*100).rename('Porcentaje de la diferencia')] ,  axis=1)
predicciones.head(10)