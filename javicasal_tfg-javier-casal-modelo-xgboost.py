

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Todas las librerías utilizadas en algún momento de la programación
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from xgboost.sklearn import XGBClassifier
from scipy.stats import uniform, randint
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.pyplot as plt
shap.initjs()
%matplotlib inline

np.random.seed(3)

from mpl_toolkits.mplot3d import Axes3D
import folium
from folium.plugins import HeatMap
#Importación de la base de datos
houses_file_path = '/kaggle/input/houses-mad/houses_Madrid_GPS.csv'
houses_data = pd.read_csv(houses_file_path)

houses_data.columns

#Estructura de los datos
houses_data.shape
#Eliminar filas que tengan variables vacías
houses_data = houses_data.dropna(axis=0)
#Como queda la estructura de los datos
houses_data.shape
#Histográmas de las variables
houses_data.hist(figsize = (18,16))
#Histograma/nomalidad de la variable objetivo buy_price
from scipy.stats import norm
sns.distplot(houses_data['buy_price'], fit=norm);
fig = plt.figure()

#Valores de simetría y curtosis antes de tratar
print("Skewness: %f" % houses_data['buy_price'].skew())
print("Kurtosis: %f" % houses_data['buy_price'].kurt())
#Matriz de correlaciones
corrmat = houses_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(houses_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Algunos scatter de las variables para representar la correlación
from matplotlib import pyplot
pyplot.scatter(houses_data["sq_mt_built"], houses_data["buy_price"])
pyplot.show()
pyplot.scatter(houses_data["n_bathrooms"], houses_data["buy_price"])
pyplot.show() 
pyplot.scatter(houses_data["n_rooms"], houses_data["buy_price"])
pyplot.show()
pyplot.scatter(houses_data["n_rooms"], houses_data["buy_price"])
pyplot.show()
pyplot.scatter(houses_data["n_bathrooms"], houses_data["buy_price"])
pyplot.show()
pyplot.scatter(houses_data["latitude"], houses_data["buy_price"])
pyplot.show()
#Gráfica de caja para ver valores extremos (los círculos negros)
plt.boxplot(houses_data["buy_price"])
plt.show()
#El percentil 90 de las X
Q90 = houses_data.quantile(0.90)


print(Q90)
#Un resumen de las estadísticas más importantes
houses_data.describe()
#Limpeza de valores extremos
print("Antes de quitar extremos: (17756, 14)")
index = houses_data[(houses_data['buy_price'] >= 1200000)|(houses_data['buy_price'] <= 70000)].index
houses_data.drop(index, inplace=True)
print("Despues de quitar extremos: " ,houses_data.shape)


#Cómo quedan después de quitar los valores extremos
houses_data.describe()
#Histográma de variable objetivo mucho más aceptable
from scipy.stats import norm
sns.distplot(houses_data['buy_price'], fit=norm);
fig = plt.figure()
res = stats.probplot(houses_data['buy_price'], plot=plt)
#Nuevos valores de asimetría y curtosis más aceptables tras eliminar valores extremos
print("Skewness: %f" % houses_data['buy_price'].skew())
print("Kurtosis: %f" % houses_data['buy_price'].kurt())
#Nuevos histogramas de las variables tras quitar valores extremos
houses_data.hist(figsize = (18,16))
#Nueva matriz de correlaciones
corrmat = houses_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(houses_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#Representación de los diferentes distritos del dataset
fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(houses_data['longitude'], houses_data['latitude'])
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
plt.show()
#Representación en un mapa de calor sobre la ciudad de Madrid de los inputs del modelo

# Encuentra la fila que tiene la casa más cara
maxpr=houses_data.loc[houses_data['buy_price'].idxmax()]

#Define la función para dibujar el mapa
def generateBaseMap(default_location=[40.4280, -3.675], default_zoom_start=9.4):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map

houses_data_copy = houses_data.copy()
houses_data_copy['count'] = 1
basemap = generateBaseMap()
# Añade un mapa tipo 'cartodbpositron'
folium.TileLayer('cartodbpositron').add_to(basemap)
s=folium.FeatureGroup(name='icon').add_to(basemap)
# Añade un marcador que indica el lugar de la casa más cara
folium.Marker([maxpr['latitude'], maxpr['longitude']],popup='Highest Price: $'+str(format(maxpr['buy_price'],'.0f')),
              icon=folium.Icon(color='green')).add_to(s)
# Añade el mapa de temperatura
HeatMap(data=houses_data_copy[['latitude','longitude','count']].groupby(['latitude','longitude']).sum().reset_index().values.tolist(),
        radius=12,max_zoom=13,name='Heat Map').add_to(basemap)
folium.LayerControl(collapsed=False).add_to(basemap)
basemap
#Selección de las variables dependiente e indepependiente
y = houses_data.buy_price
houses_features = ['sq_mt_built', 'n_rooms', 'n_bathrooms','latitude', 'longitude','floor',
                   'is_renewal_needed', 'is_new_development', 'has_lift', 'is_exterior', 'has_parking']
X = houses_data[houses_features]
#Implementación de algoritmo de árboles de decisión XGBoost

data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


xg_reg = xgb.XGBRegressor(objective ='reg:linear',
                          eta = 0.3,
                          min_child_weight = 1,
                          gamma = 0.1018506246381371,
                          colsample_bytree = 0.8629698417369874, 
                          learning_rate = 0.06164827794908118, 
                          max_depth = 5, 
                          alpha = 8.072986901537691, 
                          n_estimators = 127,
                          subsample= 0.6873761748867334)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
print(explained_variance_score(preds,y_test))
#Representación de un ejemplo de árbol de decisión del modelo
from xgboost import plot_tree
plot_tree(xg_reg)
fig = plt.gcf()
fig.set_size_inches(100, 70)


#Media de las predicciones
import statistics
statistics.mean(preds)
#Cálculo de los valores de Shapley, mediante la variable C se puede elegir que fila de la base de datos se quiere explicar

C = 58
explainer = shap.TreeExplainer(xg_reg)
shap_values = explainer.shap_values(X_test)




#Force plot de la fila C
shap.force_plot(explainer.expected_value, shap_values[C,:], X_test.iloc[C,:])
#Imprime el Precio original de la vivienda C y sus atributos
print ("Precio: " ,y_test.iloc[C], "\n" 
      ,X_test.iloc[C,:])
#Media de las variables a interpretar
X_test.mean()
X_importance = X_test
#Summary plots

shap.summary_plot(shap_values, X_test)

shap.summary_plot(shap_values, X_importance, plot_type='bar')
#Cálculo de los valores Shapley promedio por variable
shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df
#Dependence plot
shap.dependence_plot('sq_mt_built', shap_values, X_test, interaction_index="sq_mt_built")
#Otro summary plot
X_interaction = X_importance.iloc[:2500,:]

shap_interaction_values = shap.TreeExplainer(xg_reg).shap_interaction_values(X_interaction)

shap.summary_plot(shap_interaction_values, X_interaction, plot_type="compact_dot")
#Agregado de force plot, se pueden cambiar los ejes y si se desliza el cursero por los valores de Shapley los muestra
shap.force_plot(explainer.expected_value, shap_values, X_test)
#otro tipo de force plot
shap.waterfall_plot(explainer.expected_value, shap_values[C,:], feature_names=X.columns.values, max_display=5, show=True)
#Gráfico de la importacia de las variables nativo de XGBoost (para comparar)
xgb.plot_importance(xg_reg)
#Se declara la variable que reporte los mejores parámeetros para el modelo XGBoost
def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
#El código de abajo está desactivado porque busca los valores óptimos de los parámetros 
#repitiendo el cálculo del modelo, cambiando los parámetros en cada iteración
#tarda mucho tiempo