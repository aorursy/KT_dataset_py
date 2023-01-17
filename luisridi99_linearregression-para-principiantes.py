# Importar Librerias, matplotlib, numpy, pandas, seaborn, sklearn(linear_model, model_selection, metrics)

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
# Semilla random || random.seed || random.uniform

np.random.seed(101)

# X y Y

# Vamos a crear 50 valores aleatorios

x = np.linspace(0,50,50)

y = np.linspace(0,50,50)



x += np.random.uniform(-4,4,50)

y += np.random.uniform(-4,4,50)

# Mostraremos la data dispersa || scatter 

plt.scatter(x,y)

plt.xlabel('x')

plt.ylabel('y')

plt.title('Data Dispersa')
# Caracteristica

x[0]
# Valor real

y[0]
# Linear Regression || fit the model

model = LinearRegression()

model.fit(x.reshape(-1,1),y)
# Rsquared

model.score(x.reshape(-1,1),y)
# Predicciones

y_pred = model.predict(x.reshape(-1,1))
y_pred
# Evaluation metric || RMSE

RMSE = np.sqrt(mean_squared_error(y_pred,y))

RMSE
# Plotear la linea xd || plot x,y || plot x, predict || label || legend

plt.plot(x,y,'o',label='Valores reales')

plt.plot(x,y_pred,label='Regresion Lineal')

plt.xlabel('x')

plt.ylabel('y')

plt.legend()

plt.title('Linear Regression Fitted')
# Advertising csv

# path ../input/advertising.csv/Advertising.csv

Advertising = pd.read_csv('../input/advertising.csv/Advertising.csv')
# shape

Advertising.shape
# Entender los datos

Advertising.head()
# Visualizar data || pairplot(df,x_vars,y_vars,size,aspect)

sns.pairplot(Advertising,x_vars=['TV','radio','newspaper'],y_vars=['sales'],size=5,aspect=1)
# Matriz de calor || corr() || plt.figure || heatmat(m,annot,cmap)

m = Advertising.loc[:,'TV':].corr()

plt.figure(figsize=(10,10))

sns.heatmap(m,annot=True,cmap='Reds')
# Split the data || x, y, test_size, randomstate 42

X_train, X_test, Y_train, Y_test = train_test_split(Advertising.loc[:,['TV','radio']],Advertising.loc[:,'sales'],test_size=0.2,random_state=42)
# Iniciar el modelo

model_2 = LinearRegression()

# Ajustar el modelo

model_2.fit(X_train,Y_train)
# Rsquared

model_2.score(X_test,Y_test)
# Predict test

Y_pred = model_2.predict(X_test)
# RMSE test

RMSE = np.sqrt(mean_squared_error(Y_pred,Y_test))

RMSE
model_2.predict([[250.1,10.]])