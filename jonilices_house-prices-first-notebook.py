# Lectura y Carga de Datos
import pandas as pd
import numpy as np

#Representación de los Datos
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import plotly.express as px
init_notebook_mode(connected=True)
import plotly.graph_objs as go
%matplotlib inline

#Predicción
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression 

#Alertas
import warnings
warnings.filterwarnings('ignore')
#Carga del dataset de entrenamiento

df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_train_raw = df_train.copy() #Copia de seguridad por si tenemos algún problema

df_train.head()
df_train.columns
df_train.dtypes
#Análisis descriptivo de la columna 'SalePrice'

df_train["SalePrice"].describe()
# Representación en histograma de la columna 'SalePrice'

fig = px.histogram(df_train, x="SalePrice",
                   marginal="box", 
                   hover_data=df_train.columns)

fig.update_layout(title = 'Histograma del precio de venta',
                 xaxis_title = 'Precio de Venta',
                 yaxis_title = 'Número de Casas')

fig.show()
#Cálculo de la oblicuidad y la curtosis

print("La oblicuidad es: %f" % df_train['SalePrice'].skew())
print("La curtosis es: %f" % df_train['SalePrice'].kurt())
# Estudio de los datos que faltan

total = df_train.isnull().sum().sort_values(ascending=False)
porcentaje = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, porcentaje], axis=1, keys=['Total', 'Porcentaje'])
missing_data.head(20)
#Tratamiento del Missing Data
df_train = df_train.drop((missing_data[missing_data["Total"] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train["Electrical"].isnull()].index)

#Comprobación de que ya no tenemos valores faltantes
df_train.isnull().sum().max()
#Convertir las variables catégoricas en dummy

df_train_w_dummies = pd.get_dummies(df_train)
df_train_w_dummies.head(10)
#Relación de GrLivArea con SalePrice


fig = go.Figure(data = go.Scatter(x=df_train['GrLivArea'],
                                y=df_train['SalePrice'],
                                mode='markers')) # hover text goes here

fig.update_layout(title='Relación entre GrLivArea y SalePrice',
                 xaxis_title = 'GrLivArea',
                 yaxis_title = 'SalePrice')
fig.show()
#Relación de TotalBsmtSF con SalePrice


fig = go.Figure(data = go.Scatter(x=df_train['TotalBsmtSF'],
                                y=df_train['SalePrice'],
                                mode='markers')) # hover text goes here

fig.update_layout(title='Relación entre TotalBsmtSF y SalePrice',
                 xaxis_title = 'TotalBsmtSF',
                 yaxis_title = 'SalePrice')
fig.show()
#Relación de OverallQual con SalePrice

fig = px.box(df_train, x="OverallQual", y="SalePrice")

fig.update_layout(
    title = "Relación de OverallQual y SalePrice",
    xaxis = dict(showgrid = False, zeroline = False, showticklabels = False),
    yaxis = dict(zeroline = False, gridcolor = 'white')
)

fig.show()
#Relación de YearBuilt con SalePrice

fig = px.box(df_train, x="YearBuilt", y="SalePrice")

fig.update_layout(
    title = "Relación de YearBuilt y SalePrice",
    xaxis = dict(showgrid = False, zeroline = False, showticklabels = True),
    yaxis = dict(zeroline = False, gridcolor = 'white')
)

fig.show()
matriz_corr = df_train.corr()

#Definimos la figura con Seaborn
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(matriz_corr, vmax=.8, square=True)
ax.set_title('Representación de la matriz de correlación')

plt.show
k = 12 #Número de Variables
cols = matriz_corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)

f, ax = plt.subplots(figsize=(12, 9))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
ax.set_title('Representación de la correlación de SalePrice con 12 variables')

plt.show()
#Definimos las columnas con las que trabajaremos (obtenidas tras mirar el correlation plot anterior)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

#Definimos la figura
fig = px.scatter_matrix(df_train[cols],
                       title="Scatter Plots")

fig.update_traces(marker=dict(size=2.5))
fig.update_yaxes(nticks=2)


fig.show()
#Definimos las columnas que podríamos utilizar en nuestra regresión

feature_cols = ["GrLivArea", "TotalBsmtSF"]

X = df_train[feature_cols]
Y = df_train["SalePrice"]
#Empezamos modelizando

estimator = SVR(kernel="linear")       #Para una regresión lineal
selector = RFE(estimator, 2, step=1)   #Dejamos que sea el modelo quién elija cuál es la mejor opción
selector = selector.fit(X,Y)

selector.support_
#Cargamos la librería

from sklearn.linear_model import LinearRegression
#Como han sido escogidas ambas columnas, las utilizamos para nuestra predicción

X_pred = X[["GrLivArea", "TotalBsmtSF"]]

#Aplicamos el modelo

lm = LinearRegression()
lm.fit(X_pred, Y)
#Mostramos los resultados

print("Intercept : " + str(lm.intercept_))
print("Coefficients : " + str(lm.coef_))
print("Score : " + str(lm.score(X_pred, Y)))
#Definimos la figura

fig = go.Figure()

# Añadimos las trazas

fig.add_trace(go.Scatter(x=X["GrLivArea"], y=Y,
                    mode='markers',
                    name='Datos'))

fig.add_trace(go.Scatter(x=X["GrLivArea"],y = 81.84*X["GrLivArea"] -13582.5,
                        mode ='lines',
                        name = 'Predicción'))

fig.update_layout(title='Regresión Lineal',
                 xaxis_title = 'GrLivArea',
                 yaxis_title = 'SalePrice')

fig.show()
#Definimos la variable para poder hacer la limpieza

x_outl = df_train[["GrLivArea","SalePrice"]]
# Hacemos un filtrado directo para encontrar los outliers

x_outl[(x_outl["GrLivArea"] > 4000)]
x_outl_clean = x_outl.drop([523, 691, 1182, 1298])
#Creamos de nuevo el modelo

X = x_outl_clean["GrLivArea"]
X = X[:,np.newaxis]           #De esta manera evitamos el error pertinente
Y = x_outl_clean["SalePrice"]

lm = LinearRegression()
lm.fit(X,Y)
#Definimos la figura

fig = go.Figure()

# Añadimos las trazas

fig.add_trace(go.Scatter(x=x_outl_clean["GrLivArea"], y=x_outl_clean["SalePrice"],
                    mode='markers',
                    name='Datos'))

fig.add_trace(go.Scatter(x=x_outl_clean["GrLivArea"],y = 111.2*x_outl_clean["GrLivArea"] + 12582.04,
                        mode ='lines',
                        name = 'Predicción'))

fig.update_layout(title='Regresión Lineal',
                 xaxis_title = 'GrLivArea',
                 yaxis_title = 'SalePrice')

fig.show()
