#Principales librerias para la carga y preproceso de datos



import numpy as np # Libreria de algebra lineal

import pandas as pd # Libreria para manipulacion de datos

import matplotlib.pyplot as plt # matplotlib libreria para generar graficos y pyplot premite una interfaz como matlab

#Libreria de split



from sklearn.model_selection import train_test_split # X_train,X_test,y_train,y_test 
#Principales librerias para realizar modelos de prediccion



#Regresion

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression



#Principales librerias de metricas



from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error
datos = pd.read_csv("../input/avocado-prices/avocado.csv")

publicidad = pd.read_csv("../input/publicidad/publicidad.csv",sep=";")
publicidad.head(5)
import seaborn as sns

sns.set(font_scale = 1.5)

corr = publicidad.corr('spearman') 

plt.figure(figsize = ( 5 , 5 )) 

sns.heatmap(corr,annot=True,fmt='.2f',cmap="YlGnBu");

publicidad.shape
y = publicidad.iloc[:,0]

x =publicidad.iloc[:,1:4]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
lr = LinearRegression()

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print('Variance score: %.2f' % mean_squared_error(y_test, y_pred))
%config InlineBackend.figure_format = 'svg'

sns.set(font_scale = 1)

sns.distplot(y);