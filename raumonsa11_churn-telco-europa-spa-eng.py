## Importando librerias que vamos a usar

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 
# CARGA DE LOS DATASETS

bd_train = pd.read_csv("../input/train_churn_kg.csv") 

bd_test = pd.read_csv("../input/test_churn_kg.csv")
bd_train.shape ##Filas x Columnas
bd_test.shape
print(bd_train.columns.values)
bd_train.head()
#Verificación de valores perdidos

bd_train.isnull().sum()
## Visualizando datos Nulos

sns.heatmap(bd_train.isnull(),yticklabels=False,cbar=True,cmap='prism')
bd_train=bd_train.dropna(subset=['STATE_DATA', 'STATE_VOICE','CITY_DATA','CITY_VOICE'])

bd_train = bd_train.reset_index()[bd_train.columns]
## Validamos que la data no contrnga datos nulos

sns.heatmap(bd_train.isnull(),yticklabels=False,cbar=True,cmap='prism')
bd_train.isnull().sum()
## aprovechamos de limpitar la bd de test

bd_test=bd_test.dropna(subset=['STATE_DATA', 'STATE_VOICE','CITY_DATA','CITY_VOICE'])

bd_test = bd_test.reset_index()[bd_train.columns]
## convertir a entero las columnas de region y comuna

bd_train["STATE_DATA"] = bd_train["STATE_DATA"].astype(int)

bd_train["CITY_DATA"] = bd_train["CITY_DATA"].astype(int)

bd_train["STATE_VOICE"] = bd_train["STATE_VOICE"].astype(int)

bd_train["CITY_VOICE"] = bd_train["CITY_VOICE"].astype(int)
## Revisar distribución entre clientes que son Parque y No Parque (0 y 1)

## el cliente cero es que se va de la compañia



ax = sns.catplot(y="CHURN", kind="count", data=bd_train, height=2.6, aspect=2.5, orient='h')



##Flag_parque is variable target
bd_train['CHURN'].value_counts()
## visualizamos la correlación del resto de las variables con la variable target

plt.figure(figsize=(15,8))

bd_train.corr()['CHURN'].sort_values(ascending = False).plot(kind='bar')
plt.figure(figsize=(14,8))

sns.heatmap(bd_train.corr(), annot = True, cmap='YlOrRd',linewidths=.1)

plt.show()
g = sns.PairGrid(bd_train, y_vars=["CHURN"], x_vars=["MIN_PLAN", "PRICE_PLAN", "AVG_MIN_CALL_OUT_3"], height=4.5, 

                 hue="CHURN", aspect=1.1)

ax = g.map(plt.scatter, alpha=0.6)
## Promedio de vida de los clientes

## bd_train[bd_train['CHURN'] == 0]['DAYS_LIFE'].mean()



plt.figure(figsize=(12, 7))

sns.boxplot(x='CHURN',y='DAYS_LIFE',data=bd_train,palette='winter')
bd_train[['DAYS_LIFE', 'CHURN']].groupby(['CHURN'], as_index=False).mean().sort_values(by='CHURN', ascending=False)
## Fugados por estado / Churn per state

sns.countplot(x='STATE_DATA', hue='CHURN' , data=bd_train);
bd_train[(bd_train['CHURN'] == 0) & (bd_train['STATE_DATA'] == 100)]['ROA_LASTMONTH'].count()

bd_train[(bd_train['CHURN'] == 0) & (bd_train['STATE_DATA'] == 8)]['ROA_LASTMONTH'].count()

bd_train[(bd_train['CHURN'] == 0) & (bd_train['STATE_DATA'] == 5)]['ROA_LASTMONTH'].count()
## SEPARAMOS LA VARIABLE CATEGORICA Y ELIMINAMOS VARIABLES QUE NO APORTAN AL MODELO

X=bd_train[['DAYS_LIFE','MIN_PLAN','PRICE_PLAN','AVG_MIN_CALL_OUT_3', 'TOT_MIN_CALL_OUT']]



y=bd_train['CHURN']



# DATOS QUE INGRESARAN AL MODELO CUANDO SE ENCUENTRE ENTRENADO

X_to_pred =bd_test[['DAYS_LIFE','MIN_PLAN','PRICE_PLAN','AVG_MIN_CALL_OUT_3', 'TOT_MIN_CALL_OUT']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1301)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()  ## INSTANCIA DE CREACIÓN DEL MODELO RANDOM FOREST

rf.fit(X_train, y_train) ## ENTRENAMIENTO DEL MODELO

rf_pred_test = rf.predict(X_test) ##PREDICCIÓN CON LOS DATOS DE PRUEBA



rf_pred_final = rf.predict(X_to_pred) ##predicción con los datos no vistos



rf.score(X_train, y_train)



rf_score = round(rf.score(X_train, y_train)*100,2)

rf_score

reporte_rf = pd.DataFrame({"CNI_CUSTOMER": bd_test["CNI_CUSTOMER"], "CHURN" : rf_pred_final})

reporte_rf['CHURN'].value_counts()
## SEPARAMOS LA VARIABLE CATEGORICA Y ELIMINAMOS VARIABLES QUE NO APORTAN AL MODELO

X=bd_train[['DAYS_LIFE','MIN_PLAN','PRICE_PLAN','AVG_MIN_CALL_OUT_3', 'TOT_MIN_CALL_OUT']]



y=bd_train['CHURN']



# DATOS QUE INGRESARAN AL MODELO CUANDO SE ENCUENTRE ENTRENADO

X_to_pred =bd_test[['DAYS_LIFE','MIN_PLAN','PRICE_PLAN','AVG_MIN_CALL_OUT_3', 'TOT_MIN_CALL_OUT']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1301)
from sklearn.linear_model import LogisticRegression 

rl = LogisticRegression() # Creamos la instancia del modelo de regresión logistica

rl.fit(X_train,y_train) #entrenar el modelo 



pred_1 = rl.predict(X_test) ## Entrenanado los datos de prueba

pred_final = rl.predict(X_to_pred) ### Predicción nuevos datos

rl.score(X_train, y_train)



rl_score = round(rl.score(X_train, y_train)*100,2)

rl_score
reporte_rl = pd.DataFrame({"CNI_CUSTOMER": bd_test["CNI_CUSTOMER"], "CHURN" : pred_final})

reporte_rl['CHURN'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1301)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3) ##Instancia del Modelo

knn.fit(X_train, y_train) ##Entrenamiento inicial

pred_test = knn.predict(X_test)  ##Entrenamiento con datos de test
pred_final_knn = knn.predict(X_to_pred) ### Predicción nuevos datos

knn.score(X_train, y_train)

## Score del modelo

knn_score = round(knn.score(X_train, y_train)*100,2)

knn_score
reporte_knn = pd.DataFrame({"CNI_CUSTOMER": bd_test["CNI_CUSTOMER"], "CHURN" : pred_final_knn })

reporte_knn['CHURN'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1301)
from xgboost import XGBClassifier

xg = XGBClassifier()   ## Instancia del Modelo

xg.fit(X_train, y_train) ## entrnando con datos de prueba

pred_test_xg = xg.predict(X_test) ##Prediccion con datos de prueba

pred_final_xg  = xg.predict(X_to_pred)

xg.score(X_train, y_train)



xg_score = round(xg.score(X_train, y_train)*100,2)

xg_score
reporte_xg = pd.DataFrame({"CNI_CUSTOMER": bd_test["CNI_CUSTOMER"], "CHURN" : pred_final_xg })

reporte_xg['CHURN'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1301)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB () ## Instancia del modelo

nb.fit(X_train, y_train)

pred_test_nb =nb.predict(X_test) ## Prediccion datos prueba
pred_final_nb = nb.predict(X_to_pred)



nb.score(X_train, y_train)



nb_score = round(nb.score(X_train, y_train)*100,2)

nb_score
reporte_nb = pd.DataFrame({"CNI_CUSTOMER": bd_test["CHURN"], "CHURN" : pred_final_nb })

reporte_nb['CNI_CUSTOMER'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1301)
from sklearn.tree import DecisionTreeClassifier 

arbol = DecisionTreeClassifier()

arbol.fit(X_train, y_train)
pred_test_arbol = arbol.predict(X_test)
pred_final_arbol = arbol.predict(X_to_pred)

arbol.score(X_train, y_train)



arbol_score = round(arbol.score(X_train, y_train)*100,2)

arbol_score
reporte_arbol = pd.DataFrame({"CNI_CUSTOMER": bd_test["CNI_CUSTOMER"], "CHURN" : pred_final_arbol })

reporte_arbol['CHURN'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1301)
from sklearn.ensemble import ExtraTreesClassifier

arbol_2 = ExtraTreesClassifier ()

arbol_2.fit(X_train, y_train)

pred_test_arbol_2 = arbol_2.predict(X_test)
pred_final_arbol_2 = arbol_2.predict(X_to_pred)

arbol_2.score(X_train, y_train)



arbol_2_score = round(arbol_2.score(X_train, y_train)*100,2)

arbol_2_score
reporte_arbol_2 = pd.DataFrame({"CNI_CUSTOMER": bd_test["CNI_CUSTOMER"], "CHURN" : pred_final_arbol_2 })

reporte_arbol_2['CHURN'].value_counts()
Comp_modelos = pd.DataFrame({"Modelos": [ 'Random Forest',  'Regresion Logistica', 'KNN', 'XG','Naive Bayes', 'Arbol','Extra Arbol'], 

                       "Score": [rf_score, rl_score, knn_score, xg_score, nb_score, arbol_score, arbol_2_score]})
Comp_modelos.sort_values(by= "Score", ascending=False)
reporte_rf.to_csv('random_forest_report.csv', index = False)

reporte_knn.to_csv('knn_report.csv', index = False)

reporte_arbol.to_csv('arbol_report.csv', index = False)