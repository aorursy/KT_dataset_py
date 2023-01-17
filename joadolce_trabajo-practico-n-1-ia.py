# Importamos algunas librerías que vamos a utilizar a lo largo de este trabajo.

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



# Importamos el dataset previamente descrito.

path = "../input/league-of-legends-diamond-ranked-games-10-min"



df = pd.read_csv(path + "/high_diamond_ranked_10min.csv")

df.head()
# Eliminamos la columna 'gameId'

df = df.drop('gameId',  axis=1)
plt.figure(figsize=(20,15))

sns.heatmap(round(df.corr(),1), cmap="coolwarm", annot=True, linewidths=.5)

plt.show()
data = df

sns.set(font_scale=1.5)



plt.figure(figsize=(20,20))

sns.set_style("whitegrid")



# Cantidad de kills de cada equipo

plt.subplot(321)

sns.scatterplot(x='blueKills', y='redKills', hue='blueWins', data=data)

plt.title('KILLS totales de cada equipo')

plt.xlabel('Equipo Azul')

plt.ylabel('Equipo Rojo')

plt.grid(True)



# Cantidad de asistencias de cada equipo

plt.subplot(322)

sns.scatterplot(x='blueAssists', y='redAssists', hue='blueWins', data=data)

plt.title('ASISTENCIAS totales de cada equipo')

plt.xlabel('Equipo Azul')

plt.ylabel('Equipo Rojo')

plt.tight_layout(pad=1.5)

plt.grid(True)



# Cantidad total de oro de cada equipo

plt.subplot(323)

sns.scatterplot(x='blueTotalGold', y='redTotalGold', hue='blueWins', data=data)

plt.title('ORO total de de cada equipo')

plt.xlabel('Equipo Azul')

plt.ylabel('Equipo Rojo')

plt.tight_layout(pad=1.5)

plt.grid(True)



# Cantidad total de experiencia de cada equipo

plt.subplot(324)

sns.scatterplot(x='blueTotalExperience', y='redTotalExperience', hue='blueWins', data=data)

plt.title('EXPERIENCIA total de de cada equipo')

plt.xlabel('Equipo Azul')

plt.ylabel('Equipo Rojo')

plt.tight_layout(pad=1.5)

plt.grid(True)



# Cantidad total de Wards colocadas por cada equipo

plt.subplot(325)

sns.scatterplot(x='blueWardsPlaced', y='redWardsPlaced', hue='blueWins', data=data)

plt.title('WARDs totales colocadas de cada equipo')

plt.xlabel('Equipo Azul')

plt.ylabel('Equipo Rojo')

plt.tight_layout(pad=1.5)

plt.grid(True)



# Juntamos la cantidad total de minions por equipo

data['blueMinionsTotales'] = df['blueTotalMinionsKilled'] + df['blueTotalJungleMinionsKilled']

data['redMinionsTotales'] = df['redTotalMinionsKilled'] + df['redTotalJungleMinionsKilled']



# Total de minions asesinados por cada equipo

plt.subplot(326)

sns.scatterplot(x='blueMinionsTotales', y='redMinionsTotales', hue='blueWins', data=data)

plt.title('MINIONs totales asesinados de cada equipo')

plt.xlabel('Equipo Azul')

plt.ylabel('Equipo Rojo')

plt.tight_layout(pad=1.5)

plt.grid(True)



plt.show()
plt.figure(figsize=(20,20))

sns.set_style("whitegrid")



# Diferencia de experiencia y oro

plt.subplot(311)

sns.scatterplot(x='blueExperienceDiff', y='blueGoldDiff', hue='blueWins', data=data)

plt.title('Diferencia de ORO y EXPERIENCIA ')

plt.xlabel('Diferencia de Experiencia')

plt.ylabel('Diferencia de Oro')

plt.grid(True)



plt.show()
# Generamos las columnas de datos que vamos a utilizar de acuerdo a lo propuesto anteriormente.

df['blueKillsDiff'] = df['blueKills'] - df['redKills']

df['blueAssistsDiff'] = df['blueAssists'] - df['redAssists']

df['blueHeraldsDiff'] = df['blueHeralds'] - df['redHeralds']

df['blueDragonsDiff'] = df['blueDragons'] - df['redDragons']

df['blueTowersDestroyedDiff'] = df['blueTowersDestroyed'] - df['redTowersDestroyed']



# Asignamos las columnas previamente generadas en una tabla nueva lista para ser usada por los clasificadores.

X = df[['blueKillsDiff', 'blueAssistsDiff', 'blueHeraldsDiff', 'blueDragonsDiff', 

        'blueTowersDestroyedDiff', 'blueGoldDiff', 'blueExperienceDiff']]



# Asignamos la variable objetivo.

y = df['blueWins']



# Imprimimos la tabla

X.head()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

from prettytable import PrettyTable



# Creamos la tabla que nos permitirá mostrar las métricas obtenidas.

metricas = PrettyTable()

metricas.field_names = ['Clasificador', 'Exactitud', 'Recall', 'Precisión']



# Guardaremos los resultados en un vector para ser mostrados en la conclusión del trabajo.

resultados = []



# Normalizamos los datos

X = StandardScaler().fit(X).transform(X)



# Asignamos los valores de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 4)
from sklearn.linear_model import LogisticRegression



# Instanciamos el clasificador

LR = LogisticRegression()



# Hacemos fit a los datos y realizamos la predicción.

y_pred = LR.fit(X_train, y_train).predict(X_test)



# Métricas de evaluación

exactitud = accuracy_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

LR_confusion_matrix = confusion_matrix(y_test,y_pred)



# Anexamos los resultados para ser mostrados posteriormente

resultados.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas.add_row(['Regresión Logística', exactitud, recall, precision])
from sklearn.neighbors import KNeighborsClassifier



# Instanciamos el clasificador

KNN = KNeighborsClassifier()



# Hacemos fit a los datos y realizamos la predicción.

y_pred = KNN.fit(X_train, y_train).predict(X_test)



# Métricas de evaluación

exactitud = accuracy_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

KNN_confusion_matrix = confusion_matrix(y_test,y_pred)



# Anexamos los resultados para ser mostrados posteriormente

resultados.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas.add_row(['K-Nearest Neighbours', exactitud, recall, precision])
from sklearn.tree import DecisionTreeClassifier



# Instanciamos el clasificador

DT = DecisionTreeClassifier()



# Hacemos fit a los datos y realizamos la predicción.

y_pred = DT.fit(X_train, y_train).predict(X_test)



# Métricas de evaluación

exactitud = accuracy_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

DT_confusion_matrix = confusion_matrix(y_test,y_pred)



# Anexamos los resultados para ser mostrados posteriormente

resultados.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas.add_row(['Decision Tree', exactitud, recall, precision])
from sklearn.ensemble import RandomForestClassifier



# Instanciamos el clasificador

RF = RandomForestClassifier()



# Hacemos fit a los datos y realizamos la predicción.

y_pred = RF.fit(X_train, y_train).predict(X_test)



# Métricas de evaluación

exactitud = accuracy_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

RF_confusion_matrix = confusion_matrix(y_test,y_pred)



# Anexamos los resultados para ser mostrados posteriormente

resultados.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas.add_row(['Random Forest', exactitud, recall, precision])
# Imprimimos el título de la tabla.

print("Clasificadores Vanilla (Ordenados por Exactitud)")



# Ordenamos la tabla por la columna "Exactitud"

metricas.sortby = "Exactitud"



# Colocamos las filas en orden descendiente.

metricas.reversesort = True



# Imprimimos la tabla

print(metricas)
plt.figure(figsize=(20,10))



# Ploteamos la matriz de confusión de la 'Regresión Logistica'

plt.subplot(221)

sns.heatmap(LR_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('Regresión Logistica')

plt.tight_layout(pad=1.5)



# Ploteamos la matriz de confusión de la 'K-Nearest Neighbours'

plt.subplot(222)

sns.heatmap(KNN_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('K-Nearest Neighbours')

plt.tight_layout(pad=1.5)



# Ploteamos la matriz de confusión de la 'Decision Tree'

plt.subplot(223)

sns.heatmap(DT_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('Decision Tree')

plt.tight_layout(pad=1.5)



# Ploteamos la matriz de confusión de la 'Random Forest'

plt.subplot(224)

sns.heatmap(RF_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('Random Forest')

plt.tight_layout(pad=1.5)



plt.show()
# Importamos la clase de scikit-learn

from sklearn.model_selection import GridSearchCV



# Creamos la tabla que nos permitirá mostrar las métricas obtenidas.

metricas_grid_search = PrettyTable()

metricas_grid_search.field_names = ['Clasificador', 'Exactitud', 'Recall', 'Precisión']



# Guardaremos los resultados en un vector para ser mostrados en la conclusión del trabajo.

resultados_grid_search = []
# Colocamos los valores de parámetros que queremos que GridSearchCV pruebe por nosotros

grid_values = {'penalty': ['l1', 'l2'],

               'C':[.001,.009,0.01,.09,1,2,3,4,5,7,10,25],

               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

               'fit_intercept' : [True, False]}



# Instanciamos la clase con los parámetros previamente asignados

grid_clf_acc = GridSearchCV(LogisticRegression(), param_grid = grid_values, scoring = 'accuracy', verbose=False, n_jobs=-1)



# Seleccionamos la tabla entera, ya que el método se encargará de realizar la técnica de Cross-Validation

grid_clf_acc.fit(X, y)



# Imprimimos los mejores parámetros seleccionados por GridSearchCV

print("Parámetros elegidos: " + str(grid_clf_acc.best_params_) + "\n")



# Predecimos los valores

y_pred_acc = grid_clf_acc.predict(X)



# Métricas de evaluación

exactitud = accuracy_score(y,y_pred_acc)

recall = recall_score(y,y_pred_acc)

precision = precision_score(y,y_pred_acc)

LR_confusion_matrix = confusion_matrix(y,y_pred_acc)



# Anexamos los resultados para ser mostrados posteriormente

resultados_grid_search.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas_grid_search.add_row(['Regresión Logística', exactitud, recall, precision])
# Colocamos los valores de parámetros que queremos que GridSearchCV pruebe por nosotros

grid_values = {"n_neighbors": [3, 4, 5, 6, 7],

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}



# Instanciamos la clase con los parámetros previamente asignados

grid_clf_acc = GridSearchCV(KNeighborsClassifier(), param_grid = grid_values, scoring = 'accuracy', verbose=False, n_jobs=-1)



# Seleccionamos la tabla entera, ya que el método se encargará de realizar la técnica de Cross-Validation

grid_clf_acc.fit(X, y)



# Imprimimos los mejores parámetros seleccionados por GridSearchCV

print("Parámetros elegidos: " + str(grid_clf_acc.best_params_) + "\n")



# Predecimos los valores

y_pred_acc = grid_clf_acc.predict(X)



# Métricas de evaluación

exactitud = accuracy_score(y,y_pred_acc)

recall = recall_score(y,y_pred_acc)

precision = precision_score(y,y_pred_acc)

KNN_confusion_matrix = confusion_matrix(y,y_pred_acc)



# Anexamos los resultados para ser mostrados posteriormente

resultados_grid_search.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas_grid_search.add_row(['K-Nearest Neighbours', exactitud, recall, precision])
# Colocamos los valores de parámetros que queremos que GridSearchCV pruebe por nosotros

grid_values = {'max_depth': np.arange(1, 21),

               'min_samples_leaf': [1, 5, 10, 20, 50, 100]}



# Instanciamos la clase con los parámetros previamente asignados

grid_clf_acc = GridSearchCV(DecisionTreeClassifier(), param_grid = grid_values, scoring = 'accuracy', verbose=False, n_jobs=-1)



# Seleccionamos la tabla entera, ya que el método se encargará de realizar la técnica de Cross-Validation

grid_clf_acc.fit(X, y)



# Imprimimos los mejores parámetros seleccionados por GridSearchCV

print("Parámetros elegidos: " + str(grid_clf_acc.best_params_) + "\n")



# Predecimos los valores

y_pred_acc = grid_clf_acc.predict(X)



# Métricas de evaluación

exactitud = accuracy_score(y,y_pred_acc)

recall = recall_score(y,y_pred_acc)

precision = precision_score(y,y_pred_acc)

DT_confusion_matrix = confusion_matrix(y,y_pred_acc)



# Anexamos los resultados para ser mostrados posteriormente

resultados_grid_search.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas_grid_search.add_row(['Decision Tree', exactitud, recall, precision])
# Colocamos los valores de parámetros que queremos que GridSearchCV pruebe por nosotros

grid_values = {'max_features': ['auto', 'sqrt', 'log2'],

                'max_depth' : [4, 5, 6, 7, 8],

                'criterion' :['gini', 'entropy']}



# Instanciamos la clase con los parámetros previamente asignados

grid_clf_acc = GridSearchCV(RandomForestClassifier(), param_grid = grid_values, scoring = 'accuracy', verbose=False, n_jobs=-1)



# Seleccionamos la tabla entera, ya que el método se encargará de realizar la técnica de Cross-Validation

grid_clf_acc.fit(X, y)



# Imprimimos los mejores parámetros seleccionados por GridSearchCV

print("Parámetros elegidos: " + str(grid_clf_acc.best_params_) + "\n")



# Predecimos los valores

y_pred_acc = grid_clf_acc.predict(X)



# Métricas de evaluación

exactitud = accuracy_score(y,y_pred_acc)

recall = recall_score(y,y_pred_acc)

precision = precision_score(y,y_pred_acc)

RF_confusion_matrix = confusion_matrix(y,y_pred_acc)



# Anexamos los resultados para ser mostrados posteriormente

resultados_grid_search.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas_grid_search.add_row(['Random Forest', exactitud, recall, precision])
print("Clasificadores con GridSearchCV (Ordenados por Exactitud)")

metricas_grid_search.sortby = "Exactitud"

metricas_grid_search.reversesort = True

print(metricas_grid_search)
plt.figure(figsize=(20,10))



plt.subplot(221)

sns.heatmap(LR_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('Regresión Logistica')

plt.tight_layout(pad=1.5)



plt.subplot(222)

sns.heatmap(KNN_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('K-Nearest Neighbours')

plt.tight_layout(pad=1.5)



plt.subplot(223)

sns.heatmap(DT_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('Decision Tree')

plt.tight_layout(pad=1.5)



plt.subplot(224)

sns.heatmap(RF_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('Random Forest')

plt.tight_layout(pad=1.5)



plt.show()
# Calculamos la matriz de covarianza

matriz_covarianza = np.cov(X.T)



# Calculamos los autovalores y autovectores de la matriz

auto_valores, auto_vectores = np.linalg.eig(matriz_covarianza)

                                  

# A partir de los autovalores, calculamos la varianza explicada individual y la acumulada

total = sum(auto_valores)

varianza_explicada = [(i / total) * 100 for i in sorted(auto_valores, reverse=True)]

varianza_explicada_acumulada = np.cumsum(varianza_explicada)



# Graficamos la varianza explicada por cada autovalor, y la acumulada

plt.figure(figsize=(20,10))



plt.bar(range(7), varianza_explicada, alpha=0.5, align='center',label='Varianza individual explicada', color='b')

plt.step(range(7), varianza_explicada_acumulada, where='mid', linestyle='-', label='Varianza explicada acumulada', color='r')

plt.ylabel('Varianza Explicada')

plt.xlabel('Componentes Principales')

plt.legend(loc='best')

plt.tight_layout()



plt.show()
# Importamos la clase de scikit-learn

from sklearn.decomposition import PCA



# Creamos la tabla que nos permitirá mostrar las métricas obtenidas.

metricas_pca = PrettyTable()

metricas_pca.field_names = ['Clasificador', 'Exactitud', 'Recall', 'Precisión']



# Guardaremos los resultados en un vector para ser mostrados en la conclusión del trabajo.

resultados_pca = []



# Instanciamos una clase de PCA indicandole que utilizaremos 4 componentes principales

pca = PCA(n_components = 4)

X_PCA = pca.fit(X).transform(X)
# Colocamos los valores de parámetros que queremos que GridSearchCV pruebe por nosotros

grid_values = {'penalty': ['l1', 'l2'],

               'C':[.001,.009,0.01,.09,1,2,3,4,5,7,10,25],

               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

               'fit_intercept' : [True, False]}



# Instanciamos la clase con los parámetros previamente asignados

grid_clf_acc = GridSearchCV(LogisticRegression(), param_grid = grid_values, scoring = 'accuracy', verbose=False, n_jobs=-1)



# Seleccionamos la tabla entera, ya que el método se encargará de realizar la técnica de Cross-Validation

grid_clf_acc.fit(X_PCA, y)



# Imprimimos los mejores parámetros seleccionados por GridSearchCV

print("Parámetros elegidos: " + str(grid_clf_acc.best_params_) + "\n")



# Predecimos los valores

y_pred_acc = grid_clf_acc.predict(X_PCA)



# Métricas de evaluación

exactitud = accuracy_score(y,y_pred_acc)

recall = recall_score(y,y_pred_acc)

precision = precision_score(y,y_pred_acc)

LR_confusion_matrix = confusion_matrix(y,y_pred_acc)



# Anexamos los resultados para ser mostrados posteriormente

resultados_pca.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas_pca.add_row(['Regresión Logística', exactitud, recall, precision])
# Colocamos los valores de parámetros que queremos que GridSearchCV pruebe por nosotros

grid_values = {"n_neighbors": [3, 4, 5, 6, 7],

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}



# Instanciamos la clase con los parámetros previamente asignados

grid_clf_acc = GridSearchCV(KNeighborsClassifier(), param_grid = grid_values, scoring = 'accuracy', verbose=False, n_jobs=-1)



# Seleccionamos la tabla entera, ya que el método se encargará de realizar la técnica de Cross-Validation

grid_clf_acc.fit(X_PCA, y)



# Imprimimos los mejores parámetros seleccionados por GridSearchCV

print("Parámetros elegidos: " + str(grid_clf_acc.best_params_) + "\n")



# Predecimos los valores

y_pred_acc = grid_clf_acc.predict(X_PCA)



# Métricas de evaluación

exactitud = accuracy_score(y,y_pred_acc)

recall = recall_score(y,y_pred_acc)

precision = precision_score(y,y_pred_acc)

KNN_confusion_matrix = confusion_matrix(y,y_pred_acc)



# Anexamos los resultados para ser mostrados posteriormente

resultados_pca.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas_pca.add_row(['K-Nearest Neighbours', exactitud, recall, precision])
# Colocamos los valores de parámetros que queremos que GridSearchCV pruebe por nosotros

grid_values = {'max_depth': np.arange(1, 21),

               'min_samples_leaf': [1, 5, 10, 20, 50, 100]}



# Instanciamos la clase con los parámetros previamente asignados

grid_clf_acc = GridSearchCV(DecisionTreeClassifier(), param_grid = grid_values, scoring = 'accuracy', verbose=False, n_jobs=-1)



# Seleccionamos la tabla entera, ya que el método se encargará de realizar la técnica de Cross-Validation

grid_clf_acc.fit(X_PCA, y)



# Imprimimos los mejores parámetros seleccionados por GridSearchCV

print("Parámetros elegidos: " + str(grid_clf_acc.best_params_) + "\n")



# Predecimos los valores

y_pred_acc = grid_clf_acc.predict(X_PCA)



# Métricas de evaluación

exactitud = accuracy_score(y,y_pred_acc)

recall = recall_score(y,y_pred_acc)

precision = precision_score(y,y_pred_acc)

DT_confusion_matrix = confusion_matrix(y,y_pred_acc)



# Anexamos los resultados para ser mostrados posteriormente

resultados_pca.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas_pca.add_row(['Decision Tree', exactitud, recall, precision])
# Colocamos los valores de parámetros que queremos que GridSearchCV pruebe por nosotros

grid_values = {'max_features': ['auto', 'sqrt', 'log2'],

                'max_depth' : [4, 5, 6, 7, 8],

                'criterion' :['gini', 'entropy']}



# Instanciamos la clase con los parámetros previamente asignados

grid_clf_acc = GridSearchCV(RandomForestClassifier(), param_grid = grid_values, scoring = 'accuracy', verbose=False, n_jobs=-1)



# Seleccionamos la tabla entera, ya que el método se encargará de realizar la técnica de Cross-Validation

grid_clf_acc.fit(X_PCA, y)



# Imprimimos los mejores parámetros seleccionados por GridSearchCV

print("Parámetros elegidos: " + str(grid_clf_acc.best_params_) + "\n")



# Predecimos los valores

y_pred_acc = grid_clf_acc.predict(X_PCA)



# Métricas de evaluación

exactitud = accuracy_score(y,y_pred_acc)

recall = recall_score(y,y_pred_acc)

precision = precision_score(y,y_pred_acc)

RF_confusion_matrix = confusion_matrix(y,y_pred_acc)



# Anexamos los resultados para ser mostrados posteriormente

resultados_pca.append(exactitud)



# Formateamos los datos para mostrarlos como % en la tabla.

exactitud = str(round(exactitud * 100, 2)) + " %"

recall = str(round(recall * 100, 2)) + " %"

precision = str(round(precision * 100, 2)) + " %"



metricas_pca.add_row(['Random Forest', exactitud, recall, precision])
print("Clasificadores con PCA + GridSearch + Cross Validation (Ordenados por Exactitud)")

metricas_pca.sortby = "Exactitud"

metricas_pca.reversesort = True

print(metricas_pca)
plt.figure(figsize=(20,10))



plt.subplot(221)

sns.heatmap(LR_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('Regresión Logistica')

plt.tight_layout(pad=1.5)



plt.subplot(222)

sns.heatmap(KNN_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('K-Nearest Neighbours')

plt.tight_layout(pad=1.5)



plt.subplot(223)

sns.heatmap(DT_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('Decision Tree')

plt.tight_layout(pad=1.5)



plt.subplot(224)

sns.heatmap(RF_confusion_matrix, cmap="coolwarm", fmt=".0f",annot=True, linewidths=.5, annot_kws={"size": 16})

plt.xlabel("Prediccion")

plt.ylabel("Verdadero")

plt.title('Random Forest')

plt.tight_layout(pad=1.5)



plt.show()
plt.figure(figsize=(20,15))

sns.set_style("whitegrid")



modelos = ["Regresión Logística", "K-Nearest Neighbours", "Decision Tree", "Random Forest"]

grafico_resultados = pd.DataFrame({"Puntuación": resultados, "Modelos": modelos})

grafico_resultados_grid_search = pd.DataFrame({"Puntuación": resultados_grid_search, "Modelos": modelos})

grafico_resultados_pca = pd.DataFrame({"Puntuación": resultados_pca, "Modelos": modelos})



plt.subplot(311)

sns.barplot("Puntuación", "Modelos", data = grafico_resultados)

plt.ylabel("")

plt.title('Clasificadores Vanilla')

plt.tight_layout(pad=2)



plt.subplot(312)

sns.barplot("Puntuación", "Modelos", data = grafico_resultados_grid_search)

plt.ylabel("")

plt.title('Clasificadores con GridSearch + Cross Validation')

plt.tight_layout(pad=2)



plt.subplot(313)

sns.barplot("Puntuación", "Modelos", data = grafico_resultados_pca)

plt.title('Clasificadores con PCA + GridSearch + Cross Validation')

plt.ylabel("")



plt.show()
metricas.sortby = "Clasificador"

metricas_grid_search.sortby = "Clasificador"

metricas_pca.sortby = "Clasificador"



print("Clasificadores Vanilla")

print(metricas)

print("")

print("Clasificadores con GridSearch + Cross Validation")

print(metricas_grid_search)

print("")

print("Clasificadores con PCA + GridSearch + Cross Validation")

print(metricas_pca)