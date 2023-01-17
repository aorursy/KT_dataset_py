!pip install vincent
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import folium as fol
import folium
import matplotlib.pyplot as plt
import vincent as vin
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from pandas.plotting import lag_plot
from folium import plugins
from matplotlib import pyplot, transforms
from matplotlib.pyplot import show 
from matplotlib.pyplot import figure
vin.core.initialize_notebook()
df_detenciones = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/00_DETENCIONES_ E_INVESTIGADOS_DE_GRUPO_PENAL.csv")
df_hechos_conocidos = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/01_HECHOS_CONOCIDOS_GRUPO_PENAL.csv")
df_victimizaciones = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/02_VICTIMIZACIONES_GRUPO_PENAL.csv")
df_hechos_esclarecidos = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/03_HECHOS_ESCLARECIDOS_GRUPO_PENAL.csv")
df_detenciones.head()
df_detenciones.shape
df_hechos_conocidos.head()
df_hechos_conocidos.shape
df_victimizaciones.head()
df_victimizaciones.shape
df_hechos_esclarecidos.head()
df_hechos_esclarecidos.shape
df_detenc_victim = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/02_DETENCIONES_VICTIMIZACIONES_PENAL.csv")
df_detenc_victim.head()
df_conocidos_esclarecidos = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/02_HECHOS_CONOCIDOS_ESCLARECIDOS_PENAL.csv")
df_conocidos_esclarecidos.head()
df_detenc_victim.describe()
df_conocidos_esclarecidos.describe()
sns.heatmap(df_detenc_victim[["Comunidad", "Grupo Penal", "Rango Edad", "Sexo", "Año","Detenciones e investigados","Victimizaciones"]].corr(), annot=True, cmap="Blues")
sns.heatmap(df_conocidos_esclarecidos[["Hechos conocidos","Hechos esclarecidos","Comunidad","Grupo Penal","Año"]].corr(), annot=True, cmap="Greens")
totales=df_detenc_victim[df_detenc_victim["Comunidad"]!="Total Nacional"].groupby(["Comunidad"])["Detenciones e investigados"].sum()
plt.figure(figsize=(15,10))
plt.pie(totales, labels = totales.index, autopct = '%1.1f%%')  
plt.show() 
totales1=df_conocidos_esclarecidos[df_conocidos_esclarecidos["Comunidad"]!="Total Nacional"].groupby(["Comunidad"])["Hechos conocidos"].sum()
plt.figure(figsize=((15,10)))
plt.pie(totales1, labels = totales1.index, autopct = '%1.1f%%')  
plt.show()
totales2=df_conocidos_esclarecidos[df_conocidos_esclarecidos["Comunidad"]!="Total Nacional"].groupby(["Comunidad"])["Hechos esclarecidos"].sum()

plt.figure(figsize=(15,10))
plt.pie(totales2, labels = totales2.index, autopct = '%1.1f%%')  
plt.show() 
totales_0=df_detenc_victim[df_detenc_victim["Comunidad"]!="Total Nacional"].groupby(["Año"])["Detenciones e investigados"].sum().reset_index()

fig, ax = plt.subplots()
etiquetas= totales_0["Año"].unique()

valores=totales_0["Detenciones e investigados"]
x = np.arange(len(etiquetas)) 

ancho=0.4
plt.bar(x - ancho/2, valores, ancho, color="m")

ax.set_xticks(x)
ax.set_xticklabels(etiquetas)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)

plt.show()
totales_0=df_detenc_victim[df_detenc_victim["Comunidad"]!="Total Nacional"].groupby(["Año"])["Victimizaciones"].sum().reset_index()

fig, ax = plt.subplots()
etiquetas= totales_0["Año"].unique()

valores=totales_0["Victimizaciones"]
x = np.arange(len(etiquetas)) 

ancho=0.4
plt.bar(x - ancho/2, valores, ancho, color="c")

ax.set_xticks(x)
ax.set_xticklabels(etiquetas)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)

plt.show()
totales2=df_detenc_victim[df_detenc_victim["Comunidad"]!="Total Nacional"].groupby(["Grupo Penal","Sexo"])["Detenciones e investigados"].sum()
totales2=totales2.reset_index()
fig, ax = plt.subplots(figsize=(8,8))
sns.barplot(x="Grupo Penal", y="Detenciones e investigados", hue="Sexo", data=totales2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right')
plt.show()
totales_0=df_detenc_victim[df_detenc_victim["Comunidad"]!="Total Nacional"].groupby(["Rango Edad"])["Detenciones e investigados"].sum().reset_index()

fig, ax = plt.subplots()
etiquetas= totales_0["Rango Edad"].unique()

valores=totales_0["Detenciones e investigados"]
x = np.arange(len(etiquetas)) 

ancho=0.4
plt.bar(x - ancho/2, valores, ancho, color="blue")

ax.set_xticks(x)
ax.set_xticklabels(etiquetas)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(10)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(10)

plt.show()
fig, ax = plt.subplots(figsize=(8,8))
sns.barplot(x="Grupo Penal", y="Detenciones e investigados", data=totales2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right')

plt.show()
plt.style.use('fivethirtyeight')
df_detenc_victim.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)
plt.show()
plt.style.use('fivethirtyeight')
df_conocidos_esclarecidos.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)
plt.show()
df_corre_vict = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/02_VICTIMIZACIONES_GRUPO_PENAL.csv")
df_corre_vict= df_corre_vict[df_corre_vict["Comunidad"]!="Total Nacional"].groupby(["Año"])["Victimizaciones"].sum().reset_index()
X = np.asanyarray(df_corre_vict[['Año']])
y = np.asanyarray(df_corre_vict[['Victimizaciones']])
X_train = X[:-1]
y_train = y[:-1]
X_test = np.array([2019]).reshape(1, -1)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
regresion_poly = linear_model.LinearRegression()
regresion_poly.fit(X_train_poly, y_train)
print ('w_1: ', regresion_poly.coef_)
print ('w_0: ',regresion_poly.intercept_)
sns.scatterplot(X_train.flatten(),y_train.flatten(), color="magenta") #entrenamiento
plt.plot(X_train, regresion_poly.coef_[0][1]*X_train + (regresion_poly.coef_[0][2]*X_train**2)+(regresion_poly.coef_[0][3]*X_train**3)+(regresion_poly.coef_[0][4]*X_train**4)+ regresion_poly.intercept_[0], '-b')
plt.show()
X_test_poly = poly.fit_transform(X_test)
X_test_poly
yhat = regresion_poly.predict(X_test_poly)
yhat
X = np.asanyarray(df_corre_vict[['Año']])
y = np.asanyarray(df_corre_vict[['Victimizaciones']])
X_train = X
y_train = y
X_test = np.array([2020]).reshape(1, -1)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
regresion_poly = linear_model.LinearRegression()
regresion_poly.fit(X_train_poly, y_train)
print ('w_1: ', regresion_poly.coef_)
print ('w_0: ',regresion_poly.intercept_)
sns.scatterplot(X_train.flatten(),y_train.flatten(), color="magenta") #entrenamiento
plt.plot(X_train, regresion_poly.coef_[0][1]*X_train + (regresion_poly.coef_[0][2]*X_train**2)+(regresion_poly.coef_[0][3]*X_train**3)+(regresion_poly.coef_[0][4]*X_train**4)+ regresion_poly.intercept_[0], '-b')
plt.show()
X_test_poly = poly.fit_transform(X_test)
X_test_poly
yhat = regresion_poly.predict(X_test_poly)
yhat
df1_corre_detenciones = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/00_DETENCIONES_ E_INVESTIGADOS_DE_GRUPO_PENAL.csv")
df1_corre_detenciones= df1_corre_detenciones[df1_corre_detenciones["Comunidad"]!="Total Nacional"].groupby(["Año"])["Detenciones e investigados"].sum().reset_index()
X = np.asanyarray(df1_corre_detenciones[['Año']])
y = np.asanyarray(df1_corre_detenciones[['Detenciones e investigados']])
X_train = X[:-1]
y_train = y[:-1]
X_test = np.array([2019]).reshape(1, -1)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
regresion_poly = linear_model.LinearRegression()
regresion_poly.fit(X_train_poly, y_train)
print ('w_1: ', regresion_poly.coef_)
print ('w_0: ',regresion_poly.intercept_)
sns.scatterplot(X_train.flatten(),y_train.flatten(), color="magenta") #entrenamiento
plt.plot(X_train, regresion_poly.coef_[0][1]*X_train + (regresion_poly.coef_[0][2]*X_train**2)+(regresion_poly.coef_[0][3]*X_train**3)+(regresion_poly.coef_[0][4]*X_train**4)+ regresion_poly.intercept_[0], '-b')
plt.show()
X_test_poly = poly.fit_transform(X_test)
X_test_poly
yhat = regresion_poly.predict(X_test_poly)
yhat
X = np.asanyarray(df1_corre_detenciones[['Año']])
y = np.asanyarray(df1_corre_detenciones[['Detenciones e investigados']])
X_train = X
y_train = y
X_test = np.array([2020]).reshape(1, -1)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
regresion_poly = linear_model.LinearRegression()
regresion_poly.fit(X_train_poly, y_train)
print ('w_1: ', regresion_poly.coef_)
print ('w_0: ',regresion_poly.intercept_)
sns.scatterplot(X_train.flatten(),y_train.flatten(), color="magenta") #entrenamiento
plt.plot(X_train, regresion_poly.coef_[0][1]*X_train + (regresion_poly.coef_[0][2]*X_train**2)+(regresion_poly.coef_[0][3]*X_train**3)+(regresion_poly.coef_[0][4]*X_train**4)+ regresion_poly.intercept_[0], '-b')
plt.show()
X_test_poly = poly.fit_transform(X_test)
X_test_poly
yhat = regresion_poly.predict(X_test_poly)
yhat
df_corre_hecho_esclareci = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/03_HECHOS_ESCLARECIDOS_GRUPO_PENAL.csv")
df_corre_hecho_esclareci= df_corre_hecho_esclareci[df_corre_hecho_esclareci["Comunidad"]!="Total Nacional"].groupby(["Año"])["Hechos esclarecidos"].sum().reset_index()
X = np.asanyarray(df_corre_hecho_esclareci[['Año']])
y = np.asanyarray(df_corre_hecho_esclareci[["Hechos esclarecidos"]])
X_train = X
y_train = y
X_test = np.array([2019]).reshape(1, -1)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
regresion_poly = linear_model.LinearRegression()
regresion_poly.fit(X_train_poly, y_train)
print ('w_1: ', regresion_poly.coef_)
print ('w_0: ',regresion_poly.intercept_)
sns.scatterplot(X_train.flatten(),y_train.flatten(), color="magenta") #entrenamiento
plt.plot(X_train, regresion_poly.coef_[0][1]*X_train + (regresion_poly.coef_[0][2]*X_train**2)+(regresion_poly.coef_[0][3]*X_train**3)+(regresion_poly.coef_[0][4]*X_train**4)+ regresion_poly.intercept_[0], '-b')
plt.show()
X_test_poly = poly.fit_transform(X_test)
X_test_poly
yhat = regresion_poly.predict(X_test_poly)
yhat
X = np.asanyarray(df_corre_hecho_esclareci[['Año']])
y = np.asanyarray(df_corre_hecho_esclareci[["Hechos esclarecidos"]])
X_train = X
y_train = y
X_test = np.array([2020]).reshape(1, -1)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
regresion_poly = linear_model.LinearRegression()
regresion_poly.fit(X_train_poly, y_train)
print ('w_1: ', regresion_poly.coef_)
print ('w_0: ',regresion_poly.intercept_)
sns.scatterplot(X_train.flatten(),y_train.flatten(), color="magenta") #entrenamiento
plt.plot(X_train, regresion_poly.coef_[0][1]*X_train + (regresion_poly.coef_[0][2]*X_train**2)+(regresion_poly.coef_[0][3]*X_train**3)+(regresion_poly.coef_[0][4]*X_train**4)+ regresion_poly.intercept_[0], '-b')
plt.show()
X_test_poly = poly.fit_transform(X_test)
X_test_poly
yhat = regresion_poly.predict(X_test_poly)
yhat
df1_corre_hecho_conocido = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/01_HECHOS_CONOCIDOS_GRUPO_PENAL.csv")
df1_corre_hecho_conocido= df1_corre_hecho_conocido[df1_corre_hecho_conocido["Comunidad"]!="Total Nacional"].groupby(["Año"])["Hechos conocidos"].sum().reset_index()
X = np.asanyarray(df1_corre_hecho_conocido[['Año']])
y = np.asanyarray(df1_corre_hecho_conocido[['Hechos conocidos']])
X_train = X[:-1]
y_train = y[:-1]
X_test = np.array([2019]).reshape(1, -1)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
regresion_poly = linear_model.LinearRegression()
regresion_poly.fit(X_train_poly, y_train)
print ('w_1: ', regresion_poly.coef_)
print ('w_0: ',regresion_poly.intercept_)
sns.scatterplot(X_train.flatten(),y_train.flatten(), color="magenta") #entrenamiento
plt.plot(X_train, regresion_poly.coef_[0][1]*X_train + (regresion_poly.coef_[0][2]*X_train**2)+(regresion_poly.coef_[0][3]*X_train**3)+(regresion_poly.coef_[0][4]*X_train**4)+ regresion_poly.intercept_[0], '-b')
plt.show()
X_test_poly = poly.fit_transform(X_test)
X_test_poly
yhat = regresion_poly.predict(X_test_poly)
yhat
X = np.asanyarray(df1_corre_hecho_conocido[['Año']])
y = np.asanyarray(df1_corre_hecho_conocido[['Hechos conocidos']])
X_train = X
y_train = y
X_test = np.array([2020]).reshape(1, -1)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
regresion_poly = linear_model.LinearRegression()
regresion_poly.fit(X_train_poly, y_train)
print ('w_1: ', regresion_poly.coef_)
print ('w_0: ',regresion_poly.intercept_)
sns.scatterplot(X_train.flatten(),y_train.flatten(), color="magenta") #entrenamiento
plt.plot(X_train, regresion_poly.coef_[0][1]*X_train + (regresion_poly.coef_[0][2]*X_train**2)+(regresion_poly.coef_[0][3]*X_train**3)+(regresion_poly.coef_[0][4]*X_train**4)+ regresion_poly.intercept_[0], '-b')
plt.show()
X_test_poly = poly.fit_transform(X_test)
X_test_poly
yhat = regresion_poly.predict(X_test_poly)
print(yhat)
data = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/00_DETENCIONES_ E_INVESTIGADOS_DE_GRUPO_PENAL.csv")
data_00= data[data["Comunidad"]!="Total Nacional"].groupby(["Año"])["Detenciones e investigados"].sum().reset_index()
plt.figure(figsize=(8,5))
sns.scatterplot(data_00["Año"], data_00["Detenciones e investigados"], color="r")
plt.show()

X = data_00["Año"].values
y = data_00["Detenciones e investigados"].values
data_00["Año"].describe()
X_norm =X/max(X)
Y_norm =y/max(y)
plt.figure(figsize=(8,5))
sns.scatterplot(X_norm, Y_norm, color="r")
plt.show()
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, X_norm, Y_norm)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
plt.figure(figsize=(8,5))
sns.scatterplot(X_norm, Y_norm, color="r")

beta_1 = 392.906035
beta_2 = 0.996188

Y_pred = sigmoid(X_norm, beta_1,beta_2)
plt.plot(X_norm, Y_pred)
plt.show()
sigmoid(1.00049529, beta_1,beta_2)
8914.0 * 0.9182763438877961/1
data_vic = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/02_VICTIMIZACIONES_GRUPO_PENAL.csv")
data_vic_00= data_vic[data_vic["Comunidad"]!="Total Nacional"].groupby(["Año"])["Victimizaciones"].sum().reset_index()
plt.figure(figsize=(8,5))
sns.scatterplot(data_vic_00["Año"], data_vic_00["Victimizaciones"], color="c")
plt.show()

X = data_vic_00["Año"].values
y = data_vic_00["Victimizaciones"].values
data_vic_00["Año"].describe()
X_norm =X/max(X)
Y_norm =y/max(y)
plt.figure(figsize=(8,5))
sns.scatterplot(X_norm, Y_norm, color="b")
plt.show()
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, X_norm, Y_norm)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
plt.figure(figsize=(8,5))
sns.scatterplot(X_norm, Y_norm, color="c")

beta_1 = 1087.950155
beta_2 = 0.998597

Y_pred = sigmoid(X_norm, beta_1,beta_2)
plt.plot(X_norm, Y_pred)
plt.show()
sigmoid(1.00049529, beta_1,beta_2)
195446.0 * 0.883/1
data_hecho_01 = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/01_HECHOS_CONOCIDOS_GRUPO_PENAL.csv")
data_conoci_00= data_hecho_01[data_hecho_01["Comunidad"]!="Total Nacional"].groupby(["Año"])["Hechos conocidos"].sum().reset_index()
plt.figure(figsize=(8,5))
sns.scatterplot(data_conoci_00["Año"], data_conoci_00["Hechos conocidos"], color="green")
plt.show()

X = data_conoci_00["Año"].values
y = data_conoci_00["Hechos conocidos"].values
data_conoci_00["Año"].describe()
X_norm =X/max(X)
Y_norm =y/max(y)
plt.figure(figsize=(8,5))
sns.scatterplot(X_norm, Y_norm, color="green")
plt.show()
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, X_norm, Y_norm)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

plt.figure(figsize=(8,5))
sns.scatterplot(X_norm, Y_norm, color="c")

beta_1 = 1025.392282
beta_2 = 0.998519

Y_pred = sigmoid(X_norm, beta_1,beta_2)
plt.plot(X_norm, Y_pred)
plt.show()
sigmoid(1.00049529, beta_1,beta_2)
218302.0 * 0.883/1
data_hecho_02 = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/03_HECHOS_ESCLARECIDOS_GRUPO_PENAL.csv")
data_esclare_00= data_hecho_02[data_hecho_02["Comunidad"]!="Total Nacional"].groupby(["Año"])["Hechos esclarecidos"].sum().reset_index()
plt.figure(figsize=(8,5))
sns.scatterplot(data_esclare_00["Año"], data_esclare_00["Hechos esclarecidos"], color="green")
plt.show()
X = data_esclare_00["Año"].values
y = data_esclare_00["Hechos esclarecidos"].values
data_esclare_00["Año"].describe()
X_norm =X/max(X)
Y_norm =y/max(y)
plt.figure(figsize=(8,5))
sns.scatterplot(X_norm, Y_norm, color="magenta")
plt.show()
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, X_norm, Y_norm)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
plt.figure(figsize=(8,5))
sns.scatterplot(X_norm, Y_norm, color="r")

beta_1 = 641.296100
beta_2 = 0.996723

Y_pred = sigmoid(X_norm, beta_1,beta_2)
plt.plot(X_norm, Y_pred)
plt.show()
sigmoid(1.00049529, beta_1,beta_2)
30841.0 * 0.918/1
data = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/00_DETENCIONES_ E_INVESTIGADOS_DE_GRUPO_PENAL.csv")
data_detenciones= data[data["Comunidad"]!="Total Nacional"].groupby(["Año"])["Detenciones e investigados"].sum().reset_index()
X = np.array(data_detenciones["Año"].astype(int))
X= X.reshape(-1,1)
y = np.array(data_detenciones["Detenciones e investigados"].astype(int))
clf = tree.DecisionTreeRegressor(random_state = 0)
clf.fit(X, y)
clf.predict(np.array([[2020]]))
data = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/02_VICTIMIZACIONES_GRUPO_PENAL.csv")
data_vistimizaciones= data[data["Comunidad"]!="Total Nacional"].groupby(["Año"])["Victimizaciones"].sum().reset_index()
X = np.array(data_vistimizaciones["Año"].astype(int))
X= X.reshape(-1,1)
y = np.array(data_vistimizaciones["Victimizaciones"].astype(int))
clf = tree.DecisionTreeRegressor(random_state = 0)
clf.fit(X, y)
clf.predict(np.array([[2020]]))
data = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/01_HECHOS_CONOCIDOS_GRUPO_PENAL.csv")
data_h_conocidos= data[data["Comunidad"]!="Total Nacional"].groupby(["Año"])["Hechos conocidos"].sum().reset_index()
X = np.array(data_h_conocidos["Año"].astype(int))
X= X.reshape(-1,1)
y = np.array(data_h_conocidos["Hechos conocidos"].astype(int))
clf = tree.DecisionTreeRegressor(random_state = 0)
clf.fit(X, y)
clf.predict(np.array([[2020]]))
data = pd.read_csv("/kaggle/input/ciber-crimen-espaa-2020/03_HECHOS_ESCLARECIDOS_GRUPO_PENAL.csv")
data_h_esclarecidos= data[data["Comunidad"]!="Total Nacional"].groupby(["Año"])["Hechos esclarecidos"].sum().reset_index()
X = np.array(data_h_esclarecidos["Año"].astype(int))
X= X.reshape(-1,1)
y = np.array(data_h_esclarecidos["Hechos esclarecidos"].astype(int))
clf = tree.DecisionTreeRegressor(random_state = 0)
clf.fit(X, y)
clf.predict(np.array([[2020]]))
comunidades_X = {"Andalucía":37.463274379, 
                "Aragón":41.5195355493, 
               "Asturias":43.292357861, 
               "Baleares":39.5751889864,  
               "Canarias":28.339798593, 
               "Cantabria":43.1975195366,
               "Castilla y León":41.7543962127, 
               "Castilla - La Mancha":39.5809896328, 
               "Cataluña":41.7985537834, 
               "Comunidad Valenciana":39.4015584598, 
               "Extremadura":39.1914992537,
               "Galicia":42.7567966298, 
               "Comunidad de Madrid":40.495082963, 
               "Murcia":38.0023679133,
               "Navarra":42.6672011468, 
               "País Vasco":43.0433630599, 
               "La Rioja":42.2748733608, 
               "Ceuta":35.8934069863, 
               "Melilla":35.2908279949}

comunidades_Y = {"Andalucía":-4.5756251361,
                "Aragón":-0.659846411976,
               "Asturias":-5.99350932547, 
               "Baleares":2.91229172079, 
               "Canarias":-15.6720984172, 
               "Cantabria": -4.03001213183,  
               "Castilla y León":-4.78188694026, 
               "Castilla - La Mancha": -3.00462777209, 
               "Cataluña":1.52905348544, 
               "Comunidad Valenciana": -0.554726732459, 
               "Extremadura": -6.15082693044,  
               "Galicia": -7.91056344066, 
               "Comunidad de Madrid": -3.71704006617, 
               "Murcia": -1.48575857531, 
               "Navarra":-1.6461117688, 
               "País Vasco":-2.61681792149, 
               "La Rioja": -2.51703983986, 
               "Ceuta":-5.34342403891, 
               "Melilla":-2.95053552337}
df_coodenadas=pd.DataFrame({"longitud":comunidades_X,"latitud":comunidades_Y})
df_coodenadas.reset_index(inplace= True)
df_coodenadas.rename(columns={"index":"Comunidad"}, inplace= True)

df_data_det = df_detenc_victim[df_detenc_victim["Comunidad"]!="Total Nacional"]
df_data_con = df_conocidos_esclarecidos[df_conocidos_esclarecidos["Comunidad"]!="Total Nacional"]

df_mapa_det = df_data_det.groupby(["Comunidad"])["Detenciones e investigados"].sum().reset_index()
df_mapa_vic = df_data_det.groupby(["Comunidad"])["Victimizaciones"].sum().reset_index()

df_mapa_con = df_data_con.groupby(["Comunidad"])["Hechos conocidos"].sum().reset_index()
df_mapa_esc = df_data_con.groupby(["Comunidad"])["Hechos esclarecidos"].sum().reset_index()

df_mapa = pd.merge(df_mapa_det, df_mapa_vic, on=["Comunidad"]).reset_index(drop=True)
df_mapa2 = pd.merge(df_mapa_con, df_mapa_esc, on=["Comunidad"]).reset_index(drop=True)

df_mapa_mostrar = pd.merge(df_mapa, df_mapa2, on=["Comunidad"]).reset_index(drop=True)

df_mapa_mostrar = pd.merge(df_mapa_mostrar, df_coodenadas, on=["Comunidad"])

df_mapa_mostrar.head()
latitude = 37.463274
longitude = -4.575625
es_map = folium.Map(location=[latitude, longitude], zoom_start=6)

def datos_anuales_x_comunidad(comunidad):
    df_x_comunidad_det = df_detenc_victim[df_detenc_victim["Comunidad"] == comunidad].groupby("Año")["Detenciones e investigados", "Victimizaciones"].sum().reset_index()
    df_x_comunidad_conoc = df_conocidos_esclarecidos[df_conocidos_esclarecidos["Comunidad"] == comunidad].groupby("Año")["Hechos conocidos", "Hechos esclarecidos"].sum().reset_index()
    df_anuales = pd.merge(df_x_comunidad_det, df_x_comunidad_conoc, on=["Año"])
    return df_anuales.set_index("Año")

def grafico(comunidad):
    datos = datos_anuales_x_comunidad(comunidad)
    line = vin.GroupedBar(datos, width=400, height=200)
    line.axis_titles(x='Año', y='Total')
    line.legend(title=comunidad)
    return line
import math
max_hechos_conocidos = df_mapa_mostrar["Hechos conocidos"].max()
min_hechos_conocidos = df_mapa_mostrar["Hechos conocidos"].mean()
max_radio = 18
min_radio = 14

def calc_radio(valor):
    calc = math.ceil(max_radio * valor / max_hechos_conocidos)
    if calc <= 10:
        calc = math.ceil(min_radio * valor / min_hechos_conocidos)
        if calc <= 2:
            calc = calc * 7
    return calc

for lat, lng, label, hechos in zip(df_mapa_mostrar.latitud, df_mapa_mostrar.longitud, df_mapa_mostrar.Comunidad, df_mapa_mostrar["Hechos conocidos"]):
    radio = calc_radio(hechos)
    popup = folium.Popup()
    folium.Vega(grafico(label), height=250, width=600).add_to(popup)
    folium.CircleMarker(
            [lng, lat],
            radius=radio,
            weight=1,
            color='#3186cc',
            fill_color='#3186cc',
            fill_opacity=0.6,
            fill=True,
            popup=popup
        ).add_to(es_map)    
es_map
"/kaggle/input/ciber-crimen-espaa-2020/03_HECHOS_ESCLARECIDOS_GRUPO_PENAL.csv"
df_mapa_mostrar2 = df_mapa_mostrar.groupby(["Comunidad"])["Detenciones e investigados"].sum().reset_index()

esp_map2 = fol.Map(location=[latitude, longitude], zoom_start=6)

fol.Choropleth(
    geo_data="/kaggle/input/geojson-mapa-espana/comunidades-autonomas-espanolas.geojson",
    name ='Detenciones e investigados',
    data = df_mapa_mostrar2,
    columns=['Comunidad', 'Detenciones e investigados'],
    key_on='properties.comunidade_autonoma',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Detenciones e investigados'
     
).add_to(esp_map2)

fol.LayerControl().add_to(esp_map2)

esp_map2


("/kaggle/input/ciber-crimen-espaa-2020/00_DETENCIONES_ E_INVESTIGADOS_DE_GRUPO_PENAL.csv")
("/kaggle/input/ciber-crimen-espaa-2020/01_HECHOS_CONOCIDOS_GRUPO_PENAL.csv")
("/kaggle/input/ciber-crimen-espaa-2020/02_VICTIMIZACIONES_GRUPO_PENAL.csv")
("/kaggle/input/ciber-crimen-espaa-2020/03_HECHOS_ESCLARECIDOS_GRUPO_PENAL.csv")




