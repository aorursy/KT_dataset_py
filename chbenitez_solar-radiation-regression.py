#Importamos las librerias

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns
#Cargamos los datos

df_SolarRad = pd.read_csv("../input/SolarEnergy/SolarPrediction.csv")
df_SolarRad.head()
df_SolarRad.shape
#Cortamos los datos de la columna tomando los primeros 2 valores y transformandolos en enteros

df_SolarRad["Time"] = df_SolarRad.Time.str.slice(stop=2).astype(int)
df_SolarRad["Time"].describe()
#Creamos la nueva columna y la llenamos con valores Booleanos en caso de que cumpla la condicion

df_SolarRad["Morning"] = (df_SolarRad["Time"] >= 6) & (df_SolarRad["Time"] <= 12)

df_SolarRad["Afternoon"] = (df_SolarRad["Time"] >= 13) & (df_SolarRad["Time"] <= 19)

df_SolarRad["Night"] = (df_SolarRad["Time"] >= 20) & (df_SolarRad["Time"] <= 23)

df_SolarRad["EarlyMorning"] = (df_SolarRad["Time"] >= 0) & (df_SolarRad["Time"] <= 5)
#Vemos que se crearon correctamente

df_SolarRad.head(2)
from sklearn.preprocessing import LabelEncoder



lab_enc = LabelEncoder()

#Utilizamos fit_transform ya que entrenamos y aplicamos el cambio sobre el mismo conjunto de datos

df_SolarRad['Morning'] = lab_enc.fit_transform(df_SolarRad['Morning'])

df_SolarRad['Afternoon'] = lab_enc.fit_transform(df_SolarRad['Afternoon'])

df_SolarRad['Night'] = lab_enc.fit_transform(df_SolarRad['Night'])

df_SolarRad['EarlyMorning'] = lab_enc.fit_transform(df_SolarRad['EarlyMorning'])
df_SolarRad[["Time","Morning","Afternoon","Night","EarlyMorning"]].tail()
df_SolarRad["TimeSunRise"] = df_SolarRad.TimeSunRise.str.slice(stop=2).astype(int)

df_SolarRad["TimeSunSet"] = df_SolarRad.TimeSunSet.str.slice(stop=2).astype(int)
df_SolarRad["TimeSunRise"].describe()
df_SolarRad["TimeSunSet"].describe()
#Eliminamos las columnas

df_SolarRad = df_SolarRad.drop(columns=["TimeSunSet","TimeSunRise"])
plt.figure(1, figsize=(10,6)) 

plt.title("Radiation") 

sns.boxplot(df_SolarRad["Radiation"]) 



plt.figure(2, figsize=(10,6))

plt.title("Temperatura")

sns.boxplot(df_SolarRad["Temperature"])



plt.figure(3, figsize=(10,6)) 

plt.title("Presion")

sns.boxplot(df_SolarRad["Pressure"])
plt.figure(figsize=(12,6))

sns.heatmap(df_SolarRad.corr(),cmap='coolwarm',annot=True)
X = df_SolarRad[["Temperature","Pressure","Morning","Afternoon"]] #Nuestros features mas relevantes

y = df_SolarRad["Radiation"] #Separamos nuestro objetivo
#Importamos librerias para evaluar nuestros modelos

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.linear_model import LinearRegression



regression_linear = LinearRegression()



regression_linear.fit(X_train, y_train) #Entrenamos el modelo
y_pred = regression_linear.predict(X_test) #Aplicamos la prediccion
plt.scatter(y_test,y_pred) #Valores predichos 

plt.plot(y_test, y_test, 'r') #Nuestro valor real

plt.xlabel('Valor Real', fontsize = 15)  

plt.ylabel('Prediccion', fontsize = 15)  

plt.show()
#Aplicamos la raiz cuadrada (np.sqrt) al MSE para obtener el RMSE

rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

print("RMSE: ", rmse)
print("R^2: ", r2_score(y_test, y_pred))
from sklearn.tree import DecisionTreeRegressor 



tree_reg = DecisionTreeRegressor(random_state =12)



tree_reg.fit(X_train, y_train) 
y_pred = tree_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE: ", rmse)
plt.scatter(y_test,y_pred)

plt.plot(y_test, y_test, 'r')

plt.xlabel('Valor Real', fontsize = 15)  

plt.ylabel('Prediccion', fontsize = 15)  

plt.show()
print("R^2: ", r2_score(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor



regression_RF = RandomForestRegressor(n_estimators = 200, random_state =12)

regression_RF.fit(X_train, y_train)
y_pred = regression_RF.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE: ", rmse)
plt.scatter(y_test,y_pred)

plt.plot(y_test, y_test, 'r')

plt.xlabel('Valor Real', fontsize = 15)  

plt.ylabel('Prediccion', fontsize = 15)  

plt.show()
r2_score(y_test, y_pred)
from sklearn.model_selection import GridSearchCV
#Cargamos la grilla con los hiperparametros a comparar

param_grid ={'max_depth': [4, 6, 8, 10, 12], 'max_features': [1, 2, 3, 4]}
tree_reg = DecisionTreeRegressor(random_state=12)

#Pasamos los parametros a GridSearch 

grid_search = GridSearchCV(tree_reg, param_grid, cv=5,

                           scoring='r2', 

                           return_train_score=True)
#Entrenamos

grid_search.fit(X_train, y_train)
#Vemos los resultados de GridSearch 

results = pd.DataFrame(grid_search.cv_results_)

results.head()
print("El mejor score es:", grid_search.best_score_) 

print("Mejores parametros entcontrados:\n", grid_search.best_estimator_)
optimised_Tree = grid_search.best_estimator_
#Tomamos un ejemplo del conjunto de test

test_predict= X_test[50:51]

test_predict
print("Radiacion solar:",optimised_Tree.predict(test_predict), "watts por metro^2")