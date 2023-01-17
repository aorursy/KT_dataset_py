from datetime import datetime
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import datetime
import os
from pandas_profiling import ProfileReport
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
import collections
# os.chdir("E:/pythonProject") he seleccionado este directorio de trabajo. Comentar celda si no es necesario usarla
raw = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv")
raw = raw.rename(columns = {'cnt': 'Count', 't1': 'Temperature', 't2': 'Feels Like', 'hum': 'Humidity', 'wind_speed': 'Wind Speed',
                           'weather_code':'Weather Code', 'is_holiday':'Holiday', 'is_weekend':'Weekend', 'season':'Season'})
raw = raw.reset_index()
raw
# raw['Holiday'] = raw['Holiday'].replace(0, False)
# raw['Holiday'] = raw['Holiday'].replace(1, True)
# raw['Holiday'] = raw['Holiday'].astype('bool')

# raw['Weekend'] = raw['Weekend'].replace(0, False)
# raw['Weekend'] = raw['Weekend'].replace(1, True)
# raw['Weekend'] = raw['Weekend'].astype('bool')

raw['Season'] = raw['Season'].replace({0: 'Spring', 1: 'Summer', 2: 'Autumn', 3: 'Winter'})
raw['Weather Code'] = raw['Weather Code'].replace({1: 'Clear', 2: 'Few Clouds', 3: 'Broken Clouds', 4: 'Cloudy',
                                                  7: 'Light Rain', 10: 'Thunderstorm', 26: 'Snowfall', 94: 'Freezing Fog'})

raw['timestamp'] = pd.to_datetime(raw['timestamp'])
raw.dtypes
raw['hour'] = raw['timestamp'].apply(lambda time: time.hour) 
raw['month'] = raw['timestamp'].apply(lambda time: time.month)
raw['day_of_week'] = raw['timestamp'].apply(lambda time: time.dayofweek)

# Renombramos los días de la semana
date_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'} 
raw['day_of_week'] = raw['day_of_week'].map(date_names)

raw.drop('timestamp', axis = 1, inplace = True)
raw
profile = ProfileReport(raw)
profile
# Creación del dataset bueno

london = raw.join(pd.get_dummies(raw['Weather Code']), on = raw['df_index']).drop(columns = ['Weather Code'])\
.join(pd.get_dummies(raw['Season']), on = raw['df_index']).drop(columns = ['Season'])\
.join(pd.get_dummies(raw['day_of_week']), on = raw['df_index']).drop(columns = ['day_of_week'])\
.drop(columns = ['df_index'])
results = london['Count'] #variable objetivo
features = london.drop(columns = ['Count']) #variables independientes
x_train, x_test, y_train, y_test = train_test_split(features, results, test_size = 0.20, shuffle = False)
lr = LinearRegression()
lr.fit(x_train, y_train)
lr.score(x_test, y_test)
print("regresión lineal: ", mean_absolute_error(lr.predict(x_test), y_test),
      "\nPrediciendo la media: ", mean_absolute_error(np.ones([y_test.shape[0]]) * np.mean(y_test), y_test))
error_lr = y_test - lr.predict(x_test)
plt.hist(error_lr, bins = np.arange(-2000, 2000, 50)) #error sin normalizar

plt.title('Error regresión lineal (en núm. bicis)')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
stdscl = StandardScaler()
std_x_train = stdscl.fit_transform(x_train.values)
std_x_test = stdscl.fit_transform(x_test.values)
svr = SVR()
svr.fit(std_x_train, y_train)
print("SVR: ", mean_absolute_error(svr.predict(std_x_test), y_test),
      "\nPrediciendo la media: ", mean_absolute_error(np.ones([y_test.shape[0]]) * np.mean(y_test), y_test))
error_svr = y_test - svr.predict(std_x_test)
plt.hist(error_svr, bins = np.arange(-1000, 4600, 50)) #error sin normalizar

plt.title('Error SVR (en núm. bicis)')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
brr = BayesianRidge()
brr.fit(x_train, y_train)
print("Naive Bayes: ", mean_absolute_error(brr.predict(x_test), y_test),
      "\nPrediciendo la media: ", mean_absolute_error(np.ones([y_test.shape[0]]) * np.mean(y_test), y_test))
error_brr = y_test - brr.predict(x_test)
plt.hist(error_brr, bins = np.arange(-2000, 4600, 50)) #error sin normalizar

plt.title('Error NB (en núm. bicis)')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf.score(x_test, y_test) #En realidad, esta métrica no es muy importante, aunque una puntuación alta es alentadora.
print("Random Forest: ", mean_absolute_error(rf.predict(x_test), y_test),
      "\nPrediciendo la media: ", mean_absolute_error(np.ones([y_test.shape[0]]) * np.mean(y_test), y_test))
london['Count'].max()
plt.hist(london['Count'], bins = np.arange(0, 6500, 100))

plt.title('Distribución de la variable objetivo')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
rf.predict(x_test)
error = y_test - rf.predict(x_test)
plt.hist(error, bins = np.arange(-1500, 1500, 50)) #error sin normalizar

plt.title('Distribución del error de la predicción (RF)')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
error.mean(), error.std() #no está mal pero tenemos mucha desviación estándar
error_norm = error/y_test
error_norm
plt.hist(error_norm, bins = np.arange(-3, error_norm.max(), 0.1))

plt.title('Distribución del error normalizado de la predicción (RF)')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
error_norm.mean(), error_norm.std()
plt.scatter(y_test, error_norm) #el modelo se equivoca algo más cuanto más pequeño es el número de bicis que se cogen

plt.title('Error normalizado vs Número de bicis')
plt.xlabel('Número de bicis')
plt.ylabel('Error')
plt.scatter(rf.predict(x_test), y_test, alpha = 0.5)
plt.plot(y_test, y_test, color = 'magenta')
y_test[y_test < 1000]
rf.predict(x_test)[y_test < 1000]
error_pequeño = y_test[y_test < 1000] - rf.predict(x_test)[y_test < 1000]
error_pequeño.mean()
(y_test[y_test < 1000] - y_test[y_test < 1000].mean()).mean() #en recuentos pequeños parece "mucho" mejor predecir la media
plt.scatter(y_test[y_test < 1000], rf.predict(x_test)[y_test < 1000])

plt.xlabel('Número de bicis real')
plt.ylabel('Predicción')
plt.title('Predicción Random Forest vs Valores reales')
plt.scatter(y_test[y_test < 1000], error_pequeño)

plt.xlabel('Número de bicis')
plt.ylabel('Error')
plt.title('Error Random Forest a lo largo de las observaciones')
rf.get_params() #base para ver qué puedo tunear
rf.feature_importances_
importances = rf.feature_importances_
indices = np.argsort(importances)
feature_names = features.keys()

plt.title('Importancia de variables')
plt.barh(range(len(indices)), importances[indices], color = 'magenta', align = 'center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importancia relativa')
plt.show()
# añado una columna random

london['random'] = np.random.random(london['Clear'].size) #que tenga la misma longitud que las otras columnas!
results_2 = london['Count'] #variable objetivo
features_2 = london.drop(columns = ['Count']) #variables independientes
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(features_2, results_2, test_size = 0.20, shuffle = False)
rf_2 = RandomForestRegressor()
rf_2.fit(x_train_2, y_train_2)
rf_2.score(x_test_2, y_test_2)
print("Random Forest: ", mean_absolute_error(rf_2.predict(x_test_2), y_test_2),
      "\nPrediciendo la media: ", mean_absolute_error(np.ones([y_test.shape[0]]) * np.mean(y_test_2), y_test_2))
importances_2 = rf_2.feature_importances_
indices_2 = np.argsort(importances_2)
feature_names_2 = features_2.keys()

plt.title('Importancia de variables')
plt.barh(range(len(indices_2)), importances_2[indices_2], color = 'magenta', align = 'center')
plt.yticks(range(len(indices_2)), [feature_names_2[i] for i in indices_2])
plt.xlabel('Importancia relativa')
plt.show()
dict_no_ordenado = dict(zip(rf.feature_importances_, x_train.columns))
ordered = collections.OrderedDict(sorted(dict_no_ordenado.items()))
ordered
selected_features = np.array(list(ordered.values()))[-1:-15:-1]
rf.fit(x_train.loc[:,selected_features], y_train)
rf.score(x_test.loc[:, selected_features], y_test)
print("Random Forest: ", mean_absolute_error(rf.predict(x_test.loc[:,selected_features]), y_test),
      "\nPrediciendo la media: ", mean_absolute_error(np.ones([y_test.shape[0]]) * np.mean(y_test), y_test))
plt.scatter(london['hour'], london['Count'])

plt.xlabel('Hora')
plt.ylabel('Número de bicis')
plt.title('Número de bicis vs Hora del día')
# Random Forest con tuneo de parámetros
# Número de árboles
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)]
# Tuneamos el número de variables a usar en cada caso
max_features = ['auto', 'sqrt']
# Máximo número de niveles en cada árbol
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Sampsize mínimo para cada nodo
min_samples_split = [2, 5, 10]
# Sampsize mínimo para cada hoja del árbol
min_samples_leaf = [1, 2, 4]
# Método de selección
bootstrap = [True, False]
# Creamos la rejilla con los valores de arriba
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
rf_3 = RandomForestRegressor()
# Búsqueda aleatoria de parámetros con validación cruzada, 
# buscamos entre muchas combinaciones distintas.
rf_random = RandomizedSearchCV(estimator = rf_3, param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose = 3,
                               n_jobs = 5) #n_jobs = número de núcleos del procesador (cambiar en función del ordenador)
rf_random.fit(x_train, y_train)
print("Random Forest tuneado: ", mean_absolute_error(rf_random.predict(x_test), y_test),
      "\nPrediciendo la media: ", mean_absolute_error(np.ones([y_test.shape[0]]) * np.mean(y_test), y_test))
rf_random.best_params_
rf_random.predict(x_test)
error_random = y_test - rf_random.predict(x_test)
plt.hist(error_random, bins = np.arange(-1500, 1500, 50)) #error sin normalizar

plt.title('Distribución del error de la predicción (RF)')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
error_random.mean(), error_random.std() #seguimos con mucha desviación estándar
error_random_norm = error_random/y_test
error_random_norm
plt.hist(error_random_norm, bins = np.arange(-3, error_random_norm.max(), 0.1))

plt.title('Distribución del error normalizado de la predicción (RF)')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
error_random_norm.mean(), error_random_norm.std()
plt.scatter(y_test, error_random_norm, alpha = 0.3)

plt.title('Error normalizado vs Número de bicis')
plt.xlabel('Número de bicis')
plt.ylabel('Error')
plt.scatter(rf_random.predict(x_test), y_test, alpha = 0.5)
plt.plot(y_test, y_test, color = 'magenta') #de momento parece que el tuneo no está siendo exitoso
y_test[y_test < 1000]
rf_random.predict(x_test)[y_test < 1000]
error_pequeño_random = y_test[y_test < 1000] - rf_random.predict(x_test)[y_test < 1000]
error_pequeño_random.mean()
(y_test[y_test < 1000] - y_test[y_test < 1000].mean()).mean() #Sigue siendo mejor predecir la media en recuentos pequeños
plt.scatter(y_test[y_test < 1000], rf_random.predict(x_test)[y_test < 1000])
plt.plot(y_test[y_test < 1000], y_test[y_test < 1000], color = "magenta")

plt.xlabel('Número de bicis real')
plt.ylabel('Predicción')
plt.title('Predicción Random Forest vs Valores reales')
plt.scatter(y_test[y_test < 1000], error_pequeño_random)
plt.axhline(y=0, color='m', linestyle='-')

plt.xlabel('Número de bicis')
plt.ylabel('Error')
plt.title('Error Random Forest a lo largo de las observaciones')