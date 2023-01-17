# Importo las librerías básicas:



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore') # Para evitar los molestos avisos.

%matplotlib inline
# Asigno los datos a un dataframe:



df_train = pd.read_csv('../input/train.csv')

df_train.head(10)
# Echo un vistazo las columnas:



df_train.columns
# Resumen de estadística descriptiva:



df_train['SalePrice'].describe()
# Histograma:



sns.distplot(df_train['SalePrice']);
# Asimetría y curtosis:



print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
# Diagrama de dispersión grlivarea/saleprice:



var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', alpha = 0.5);
# Diagrama de dispersión totalbsmtsf/saleprice:



var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', alpha = 0.5);
# Diagrama de cajas overallqual/saleprice:



var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
# Matriz de correlación:



corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
# Matriz de correlación

k = 10 # Número de variables.

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale = 1.25)

hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)

plt.show()
corr = df_train.corr()

corr[['SalePrice']].sort_values(by = 'SalePrice',ascending = False).style.background_gradient()
# Scatter plot:



sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
# Missing data:



total = df_train.isnull().sum().sort_values(ascending = False)

percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

missing_data.head(20)
# Tratamiento de datos faltantes:



df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() # Para comprobar que no hay más datos desaparecidos.
# Estandarización de datos:



saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('Fuera de la distribución (por debajo):')

print(low_range)

print('\nFuera de la distribución (por arriba):')

print(high_range)
# Análisis bivariable SalePrice/GrLivArea:



var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', alpha = 0.5);
# Eliminación de valores:



df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
# Análisis bivariable SalePrice/TotalBsmtSF:



var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', alpha = 0.5);
# Histograma y gráfico de probabilidad normal:



sns.distplot(df_train['SalePrice'], fit = norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot = plt)
# Transformación de los datos:



df_train['SalePrice'] = np.log(df_train['SalePrice'])
# Histograma y gráfico de probabilidad normal sobre los datos transformados:



sns.distplot(df_train['SalePrice'], fit = norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot = plt)
# Histograma y gráfico de probabilidad normal:



sns.distplot(df_train['GrLivArea'], fit = norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot = plt)
# Transformación de datos:



df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
# Histograma y gráfico de probabilidad normal sobre los datos transformados:



sns.distplot(df_train['GrLivArea'], fit = norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot = plt)
# Histograma y gráfico de probabilidad normal:



sns.distplot(df_train['TotalBsmtSF'], fit = norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot = plt)
# Creación de la columna para una nueva variable categórica binaria (1 si area>0, 0 si area==0):



df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index = df_train.index)

df_train['HasBsmt'] = 0 

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
# Transformación de datos:



df_train.loc[df_train['HasBsmt'] == 1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
# Histograma y gráfico de probabilidad normal:



sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit = norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot = plt)
# Gráfico de dispersión:



plt.scatter(df_train['GrLivArea'], df_train['SalePrice'], alpha = 0.5);
# Gráfico de dispersión:



plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF'] > 0]['SalePrice'], alpha = 0.5);
# Convertir las variables categóricas en variables ficticias o dummies:



df_train = pd.get_dummies(df_train)