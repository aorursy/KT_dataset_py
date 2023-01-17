##IMPORTE DE LIBRERIAS QUE MAS SERÁN USADAS

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
## CARGA DE LOS DATASETS

bd_train = pd.read_csv("../input/train .csv") ##70 %

bd_test = pd.read_csv("../input/test.csv") #30 %



bd_train.info() ## visualizar lo que tenemos
##VISUALIZACIÓN DE ESTRUCTUDA DE DATOS DE ENTRENAMIENTO

bd_train.head(5)
bd_train.loc[bd_train['MARCA_EQUIPO'] == 'SIN_DESCRIPCION','MARCA_EQUIPO'] = 17

bd_train.loc[bd_train['MARCA_EQUIPO'] == 'OTROS_EQUIPOS','MARCA_EQUIPO'] = 18
## VISUALIZAR FUGADOS Y NO FUGADOS DE BASE DE ENTRENAMIENTO

bd_train['FLAG_PARQUE'].value_counts()
sns.set_style('whitegrid') # Estilo de la grilla 

sns.countplot(x='FLAG_PARQUE',data=bd_train,palette='viridis') 
#FLAG PARQUE VS MOTIVO FUGA

sns.countplot(x='MOTIVO_FUGA',hue='FLAG_PARQUE',data=bd_train)
#FLAG PARQUE VS MOTIVO FUGA

sns.countplot(x='CIA_FUGA',hue='FLAG_PARQUE',data=bd_train)
## EN LAS 2 VISUALIZACIONES ANTERIORES PODEMOS ENETNDER QUE UN GRAN NUMERO DE FUGADOS SE VA A

## LA COMPAÑIA 99, MIENTRAS QUE EL MOTIVO MAS FRECUENTE DE FUGA ES 0 Y 21

## MAPA DE CALOR PARA VIUSALIZAR CORRELACIÓN DE VARIABLES

#MATRIZ DE CORRELACIÓN DE VARIABLES

plt.figure(figsize=(14,8))

sns.heatmap(bd_train.corr(), annot = True, cmap='YlOrRd',linewidths=.1)

plt.show()
cols = ['MIN_PLAN', 'VALOR_PLAN']

pp = sns.pairplot(bd_train[cols], size=1.8, aspect=1.8,

                  plot_kws=dict(edgecolor="k", linewidth=0.5),

                  diag_kind="kde" ,diag_kws=dict(shade=True))



fig = pp.fig 

fig.subplots_adjust(top=0.93, wspace=0.3)

t = fig.suptitle('Atributos Correlacionados', fontsize=14)

plt.show()
## VISUALIZAR LAS VARIABLES MAS CORRELACIONADAS

from scipy.stats import pearsonr

sns.jointplot(x='AVG_MIN_SAL_3',y='TOT_MIN_SAL_ULT_MES',data=bd_train,kind='reg',stat_func=pearsonr)

### PROMEDIO DE VIDA DE LOS CLIENTES

#PROMEDIO DE VIDA DE LA CATEGORÍA DE CLIENTES

plt.figure(figsize=(12, 7))

sns.boxplot(x='FLAG_PARQUE',y='VIDA_WOM',data=bd_train,palette='winter')
## SE PUEDE VER QUE EL PROMEDIO DE VIDA DE LOS CLIENTES 0, ES DECIR LOS FUGADOS ES DE 250 DÍAS APROX

##  POR OTRO LADO LOS CLIENTES QUE ESTAN VIVOS TIENEN UN PROMEDIO DE VIDA DE 300 DÍAS.
# PREPARANDO VARIABLES



## SEPARAMOS LA VARIABLE CATEGORICA Y ELIMINAMOS VARIABLES QUE NO APORTAN AL MODELO

X= bd_train.drop(columns=['FLAG_PARQUE','RUT_CLIENTE','NUMERO_ABONADO', 'MARCA_EQUIPO','MOTIVO_FUGA','CIA_FUGA'],axis=1) 

y=bd_train['FLAG_PARQUE']



# DATOS QUE INGRESARAN AL MODELO CUANDO SE ENCUENTRE ENTRENADO

X_to_pred=bd_test.drop(columns=['FLAG_PARQUE','RUT_CLIENTE','NUMERO_ABONADO', 'MARCA_EQUIPO','MOTIVO_FUGA','CIA_FUGA'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1301)



from sklearn.linear_model import LogisticRegression 

lrmodel = LogisticRegression() # Creamos la instancia del modelo de regresión logistica

lrmodel.fit(X_train,y_train) #entrenar el modelo 
# CALIDAD DEL MODELO



lrmodel.score(X_train, y_train)

logistic_score = round(lrmodel.score(X_train, y_train)*100,2)

logistic_score
## PREDICCIÓN CON LOS DATOS DE PRUEBA

prediccion=lrmodel.predict(X_test)  

error = np.mean((y_test - prediccion))

print('Error = ',error)
## PREDICCIÓN CON LOS DATOS DE TEST (LOS QUE EL MODELO NO CONOCE)

Y_Pred=lrmodel.predict(X_to_pred)
## ARMADO DE DATAFRAME CON PREDICCIONES

df_report = pd.DataFrame({"RUT_CLIENTE": bd_test["RUT_CLIENTE"], "FLAG_PARQUE" : Y_Pred})
df_report.head()
df_report.to_csv('RUT_cliente_pred.csv')
from sklearn.ensemble import RandomForestClassifier

ranfor = RandomForestClassifier()  ## INSTANCIA DE CREACIÓN DEL MODELO RANDOM FOREST

ranfor.fit(X_train, y_train) ## ENTRENAMIENTO DEL MODELO

pred_rf = ranfor.predict(X_test) ##PREDICCIÓN CON LOS DATOS DE PRUEBA



pred_rf2 = ranfor.predict(X_to_pred) ##predicción con los datos no vistos



ranfor.score(X_train, y_train)



ranfor_score = round(ranfor.score(X_train, y_train)*100,2)

ranfor_score
df_report2 = pd.DataFrame({"RUT_CLIENTE": bd_test["RUT_CLIENTE"], "FLAG_PARQUE" : pred_rf2})
df_report2.head()
df_report2.to_csv('Random.csv')
from sklearn.neighbors import KNeighborsClassifier



KNC = KNeighborsClassifier(n_neighbors=3)

KNC.fit(X_train, y_train)

Y_predKNC = KNC.predict(X_test)

print('Accuracy KNN : {}'.format(KNC.score(X_train, y_train)))
## CREAR UN DATAFRAME PARA COMENZAR CON LAS NUEVAS PREDICCIONES

Y_predKNC = KNC.predict(X_to_pred)
## REPORTE QUE GUARDA RESULTADOS

df_reportKNC = pd.DataFrame({"RUT_CLIENTE":bd_test["RUT_CLIENTE"], "FLAG_PARQUE" : Y_predKNC})
df_reportKNC .head()
df_reportKNC.to_csv('KNN.csv')
from sklearn.tree import DecisionTreeClassifier

arbol = DecisionTreeClassifier()

arbol.fit(X_train, y_train)

Y_pred_arbol = arbol.predict(X_test) ##Datos_Test

print('Accuracy Arbol : {}'.format(arbol.score(X_train, y_train)))
##CREAR DATAFRAME PARA LOS DATOS DESCONOCIDOS

Y_pred_arbol = arbol.predict(X_to_pred)
## REPORTE DE RESULTADOS

df_reportARBOL = pd.DataFrame({"RUT_CLIENTE":bd_test["RUT_CLIENTE"], "FLAG_PARQUE" : Y_pred_arbol})
df_reportARBOL.head()
df_reportARBOL.to_csv('arbol.csv')
logistic_score = round(lrmodel.score(X_train, y_train)*100,2)

print('Regresion_Logistica',logistic_score)

ranfor_score = round(ranfor.score(X_train, y_train)*100,2)

print('Random_Foresst', ranfor_score)

knn_score = round(KNC.score(X_train, y_train)*100,2)

print('KNN_Foresst', knn_score )

arbol_score = round(arbol.score(X_train, y_train)*100,2)

print('Decision Tree', arbol_score )
