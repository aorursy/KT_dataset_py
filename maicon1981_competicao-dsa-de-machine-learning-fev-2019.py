import os

path = os.getcwd()
print(os.listdir('../input'))
import pandas as pd

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/dataset_treino.csv', sep = ',', encoding = 'utf-8')

df_test  = pd.read_csv('../input/dataset_teste.csv', sep = ',', encoding = 'utf-8')
df_train.head(5)
df_train.shape
df_test.shape
df_train.info()
df_train[df_train.isnull().any(axis=1)] 
df_train.isnull().values.any() 
df_train.isnull().any()
df_train.isnull().sum()
df_test.isnull().sum()
# Função para verificar características de uma variável

def info_variavel(df):

    print(df.head())

    print('')

    print('Quantidade de registros:', df.count())

    print('Possui valores missing?:', df.isnull().any())

    print('Quantidade de valores missing:', df.isnull().sum())

    print('')

    print('Valores únicos:')

    print(df.unique())

    print('')

    print('Contagem dos elementos:')

    print(df.value_counts())
# Intervalo de valores

def intervalo_valores(df):

    print('Valor mínimo:', df.min())

    print('Valor máximo:', df.max())
# Order: Identificador único

index = 'Order'
info_variavel(df_train[index])
del df_train[index]

del df_test['OrderId']
# Property Id: Identificador único

index = 'Property Id'
info_variavel(df_train[index])
del df_train[index]

#del df_test[index]
# Property Name

index = 'Property Name'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Parent Property Id

index = 'Parent Property Id'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Parent Property Name

index = 'Parent Property Name'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# BBL - 10 digits

index = 'BBL - 10 digits'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# NYC Borough, Block and Lot (BBL) self-reported

index = 'NYC Borough, Block and Lot (BBL) self-reported'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# NYC Building Identification Number (BIN)

index = 'NYC Building Identification Number (BIN)'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Address 1 (self-reported)

index = 'Address 1 (self-reported)'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Address 2

index = 'Address 2'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Postal Code

index = 'Postal Code'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Street Number

index = 'Street Number'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Street Name

index = 'Street Name'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Borough

index = 'Borough'
info_variavel(df_train[index])
df_train[index] = df_train[index].astype('category').cat.codes

df_test[index]  = df_test[index].astype('category').cat.codes
info_variavel(df_train[index])
df_train = df_train[df_train[index] > -1]
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# DOF Gross Floor Area

index = 'DOF Gross Floor Area'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Primary Property Type - Self Selected

index = 'Primary Property Type - Self Selected'
info_variavel(df_train[index])
df_train[index] = df_train[index].astype('category').cat.codes

df_test[index]  = df_test[index].astype('category').cat.codes
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# List of All Property Use Types at Property

index = 'List of All Property Use Types at Property'
info_variavel(df_train[index])
df_train[index] = df_train[index].astype('category').cat.codes

df_test[index]  = df_test[index].astype('category').cat.codes
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Largest Property Use Type

index = 'Largest Property Use Type'
info_variavel(df_train[index])
df_train[index] = df_train[index].astype('category').cat.codes

df_test[index]  = df_test[index].astype('category').cat.codes
intervalo_valores(df_train[index])
# Largest Property Use Type - Gross Floor Area (ft²)

index = 'Largest Property Use Type - Gross Floor Area (ft²)'
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# 2nd Largest Property Use Type

index = '2nd Largest Property Use Type'
info_variavel(df_train[index])
df_train[index] = df_train[index].astype('category').cat.codes

df_test[index]  = df_test[index].astype('category').cat.codes
intervalo_valores(df_train[index])
# 2nd Largest Property Use - Gross Floor Area (ft²)

index = '2nd Largest Property Use - Gross Floor Area (ft²)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# 3rd Largest Property Use Type

index = '3rd Largest Property Use Type'
info_variavel(df_train[index])
df_train[index] = df_train[index].astype('category').cat.codes

df_test[index]  = df_test[index].astype('category').cat.codes
intervalo_valores(df_train[index])
# 3rd Largest Property Use Type - Gross Floor Area (ft²)

index = '3rd Largest Property Use Type - Gross Floor Area (ft²)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Year Built

index = 'Year Built'
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Number of Buildings - Self-reported

index = 'Number of Buildings - Self-reported'
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Occupancy

index = 'Occupancy' 
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Metered Areas (Energy)

index = 'Metered Areas (Energy)'
info_variavel(df_train[index])
df_train[index] = df_train[index].astype('category').cat.codes

df_test[index]  = df_test[index].astype('category').cat.codes
intervalo_valores(df_train[index])
# Metered Areas  (Water)

index = 'Metered Areas  (Water)'
info_variavel(df_train[index])
df_train[index] = df_train[index].astype('category').cat.codes

df_test[index]  = df_test[index].astype('category').cat.codes
intervalo_valores(df_train[index])
# Site EUI (kBtu/ft²)

index = 'Site EUI (kBtu/ft²)'
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Weather Normalized Site EUI (kBtu/ft²)

index = 'Weather Normalized Site EUI (kBtu/ft²)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Weather Normalized Site Electricity Intensity (kWh/ft²)

index = 'Weather Normalized Site Electricity Intensity (kWh/ft²)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Weather Normalized Site Natural Gas Intensity (therms/ft²)

index = 'Weather Normalized Site Natural Gas Intensity (therms/ft²)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Weather Normalized Source EUI (kBtu/ft²)

index = 'Weather Normalized Source EUI (kBtu/ft²)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Fuel Oil #1 Use (kBtu)

index = 'Fuel Oil #1 Use (kBtu)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Fuel Oil #2 Use (kBtu)

index = 'Fuel Oil #2 Use (kBtu)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Fuel Oil #4 Use (kBtu)

index = 'Fuel Oil #4 Use (kBtu)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Fuel Oil #5 & 6 Use (kBtu)

index = 'Fuel Oil #5 & 6 Use (kBtu)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Diesel #2 Use (kBtu)

index = 'Diesel #2 Use (kBtu)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# District Steam Use (kBtu)

index = 'District Steam Use (kBtu)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Natural Gas Use (kBtu)

index = 'Natural Gas Use (kBtu)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Weather Normalized Site Natural Gas Use (therms)

index = 'Weather Normalized Site Natural Gas Use (therms)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Electricity Use - Grid Purchase (kBtu)

index = 'Electricity Use - Grid Purchase (kBtu)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Weather Normalized Site Electricity (kWh)

index = 'Weather Normalized Site Electricity (kWh)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Total GHG Emissions (Metric Tons CO2e)

index = 'Total GHG Emissions (Metric Tons CO2e)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Direct GHG Emissions (Metric Tons CO2e)

index = 'Direct GHG Emissions (Metric Tons CO2e)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Indirect GHG Emissions (Metric Tons CO2e)

index = 'Indirect GHG Emissions (Metric Tons CO2e)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Property GFA - Self-Reported (ft²)

index = 'Property GFA - Self-Reported (ft²)'
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Water Use (All Water Sources) (kgal)

index = 'Water Use (All Water Sources) (kgal)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Water Intensity (All Water Sources) (gal/ft²)

index = 'Water Intensity (All Water Sources) (gal/ft²)'
info_variavel(df_train[index])
df_train[index] = df_train[index].str.replace('Not Available', '0')

df_test[index]  = df_test[index].str.replace('Not Available', '0')
df_train[index] = df_train[index].astype('float64')

df_test[index]  = df_test[index].astype('float64')
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Source EUI (kBtu/ft²)

index = 'Source EUI (kBtu/ft²)'
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# Release Date

index = 'Release Date'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Water Required?

index = 'Water Required?'
info_variavel(df_train[index])
df_train[index] = df_train[index].astype('category').cat.codes

df_test[index]  = df_test[index].astype('category').cat.codes
info_variavel(df_train[index])
df_train = df_train[df_train[index] > -1]
info_variavel(df_train[index])
intervalo_valores(df_train[index])
# DOF Benchmarking Submission Status

index = 'DOF Benchmarking Submission Status'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Latitude

index = 'Latitude'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Longitude

index = 'Longitude'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Community Board

index = 'Community Board'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Council District

index = 'Council District'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# Census Tract

index = 'Census Tract'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# NTA

index = 'NTA'
info_variavel(df_train[index])
del df_train[index]

del df_test[index]
# ENERGY STAR Score - É a variável dependente!!

index = 'ENERGY STAR Score'
info_variavel(df_train[index])
intervalo_valores(df_train[index])
df_train.shape
df_test.shape
df_train.info()
df_test.info()
df_train.isnull().any()
df_test.isnull().any()
df_train.isnull().sum()
df_test.isnull().sum()
dataset = df_train.drop(['ENERGY STAR Score'], axis=1)

dataset['ENERGY STAR Score'] = df_train['ENERGY STAR Score']
# Coletando X e y

X = dataset.iloc[:,:-1]

y = dataset['ENERGY STAR Score'].values
X.head()
y
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 
# Sumário estatístico

X.describe()
# Correlação de Pearson

matriz_corr = X.corr(method = 'pearson')

matriz_corr.style.background_gradient()
# Plotando matriz de correlação

corr = X.corr()

_ , ax = plt.subplots( figsize =( 14 , 10 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = False, annot_kws = {'fontsize' : 12 }, linewidths=0.01)
from sklearn import linear_model

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
# Criando um modelo

modelo = GradientBoostingRegressor()
standardization = MinMaxScaler()

Stand_coef_linear_reg = make_pipeline(standardization, modelo)
Stand_coef_linear_reg.fit(X,y)

for feature_importances_, var in sorted(zip(map(abs, Stand_coef_linear_reg.steps[1][1].feature_importances_), dataset.columns[:-1]), reverse = True):

    print ("%6.3f %s" % (feature_importances_,var))
# Variáveis irrelevantes

cols_irr = [#'Source EUI (kBtu/ft²)',

            #'Largest Property Use Type',

            #'Primary Property Type - Self Selected',

            #'List of All Property Use Types at Property',

            #'Site EUI (kBtu/ft²)',

            #'Weather Normalized Site Electricity Intensity (kWh/ft²)',

            #'Year Built',

            #'Largest Property Use Type - Gross Floor Area (ft²)',

            #'Weather Normalized Source EUI (kBtu/ft²)',

            #'2nd Largest Property Use Type',

            #'2nd Largest Property Use - Gross Floor Area (ft²)',

            'Property GFA - Self-Reported (ft²)',

            'Weather Normalized Site EUI (kBtu/ft²)',

            'Water Use (All Water Sources) (kgal)',

            'Water Intensity (All Water Sources) (gal/ft²)',

            'Electricity Use - Grid Purchase (kBtu)',

            'Borough',

            'Natural Gas Use (kBtu)',

            'Fuel Oil #2 Use (kBtu)',

            'Indirect GHG Emissions (Metric Tons CO2e)',

            'Total GHG Emissions (Metric Tons CO2e)',

            'Direct GHG Emissions (Metric Tons CO2e)',

            'Occupancy',

            'Weather Normalized Site Natural Gas Intensity (therms/ft²)',

            '3rd Largest Property Use Type - Gross Floor Area (ft²)',

            'Weather Normalized Site Natural Gas Use (therms)',

            'Weather Normalized Site Electricity (kWh)',

            'Number of Buildings - Self-reported',

            'Fuel Oil #4 Use (kBtu)',

            'Water Required?',

            'Metered Areas (Energy)',

            'Metered Areas  (Water)',

            'Fuel Oil #5 & 6 Use (kBtu)',

            'Fuel Oil #1 Use (kBtu)',

            'District Steam Use (kBtu)',

            'Diesel #2 Use (kBtu)',

            '3rd Largest Property Use Type'

           ]
# Removendo variáveis irrelevantes

X = X.drop(columns = cols_irr)
X.head()
y
# Transformando os dados para a mesma escala (entre 0 e 1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols = X.columns

X[cols] = scaler.fit_transform(X)

X.head()
# Padronizando os dados (0 para a média, 1 para o desvio padrão)

#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()

#cols = X.columns

#X[cols] = scaler.fit_transform(X)

#X.head()
Xc = sm.add_constant(X)

modelo_v1 = sm.OLS(y, Xc)

modelo_v2 = modelo_v1.fit()
modelo_v2.summary()
modelo = linear_model.LinearRegression(normalize = False, fit_intercept = True)
def r2_est(X,y):

    return r2_score(y, modelo.fit(X,y).predict(X).clip(1,100))
#def mean_absolute_error(X,y):

#    return mean_absolute_error(y, modelo.fit(X,y).predict(X))
def mean_absolute_error(y, y_pred):

    return (1 / len(y)) * np.sum(np.abs(y - y_pred))
print ('Baseline R2: %0.3f' %  r2_est(X,y))
predictions = modelo.fit(X,y).predict(X).clip(1,100)

predictions
#print ('MAE test score: %0.3f' %  mean_absolute_error(X,y))
print ('MAE test score: %0.3f' %  mean_absolute_error(y, predictions))
# MAE

# Mean Absolute Error

# É a soma da diferença absoluta entre previsões e valores reais.

# Fornece uma ideia de quão erradas estão nossas previsões.

# Valor igual a 0 indica que não há erro, sendo a previsão perfeita (a função cross_val_score inverte o valor)



# Import dos módulos

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

#from sklearn.svm import SVR



# Definindo os valores para o número de folds

num_folds = 10

num_instances = len(X)

seed = 998



# Preparando os modelo

modelos = []

modelos.append(('LINEAR', LinearRegression()))

modelos.append(('RIDGE', Ridge()))

modelos.append(('LASSO', Lasso()))

modelos.append(('ELASTIC', ElasticNet()))

modelos.append(('KNN', KNeighborsRegressor()))

modelos.append(('CART', DecisionTreeRegressor()))

#modelos.append(('SVM', SVR()))



# Avaliando cada modelo

resultados = []

nomes = []



for nome, modelo in modelos:

    kfold = model_selection.KFold(n_splits = num_folds, random_state = seed)

    cv_results = model_selection.cross_val_score(modelo, X, y, cv = kfold, scoring = 'neg_mean_absolute_error')

    resultados.append(cv_results)

    nomes.append(nome)

    msg = "%s: %.3f (%.3f)" % (nome, cv_results.mean(), cv_results.std())

    print(msg)
# Boxplot para comparar os algoritmos

fig = plt.figure()

fig.suptitle('Comparação dos Algoritmos')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# MAE

# Mean Absolute Error

# É a soma da diferença absoluta entre previsões e valores reais.

# Fornece uma ideia de quão erradas estão nossas previsões.

# Valor igual a 0 indica que não há erro, sendo a previsão perfeita (a função cross_val_score inverte o valor)



# Import dos módulos

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



# Definindo os valores para o número de folds

num_folds = 10

num_instances = len(X)

seed = 998



# Preparando os modelo

modelos = []

modelos.append(('AD', AdaBoostRegressor()))

modelos.append(('BG', BaggingRegressor()))

modelos.append(('ET', ExtraTreesRegressor()))

modelos.append(('GB', GradientBoostingRegressor()))

modelos.append(('RF', RandomForestRegressor()))

modelos.append(('XGB', XGBRegressor()))



# Avaliando cada modelo

resultados = []

nomes = []



for nome, modelo in modelos:

    kfold = model_selection.KFold(n_splits = num_folds, random_state = seed)

    cv_results = model_selection.cross_val_score(modelo, X, y, cv = kfold, scoring = 'neg_mean_absolute_error')

    resultados.append(cv_results)

    nomes.append(nome)

    msg = "%s: %.3f (%.3f)" % (nome, cv_results.mean(), cv_results.std())

    print(msg)
# Boxplot para comparar os algoritmos

fig = plt.figure()

fig.suptitle('Comparação dos Algoritmos')

ax = fig.add_subplot(111)

plt.boxplot(resultados)

ax.set_xticklabels(nomes)

plt.show()
# Identificador

index = df_test['Property Id']

df_test = df_test.drop(columns = ['Property Id'])
# Removendo variáveis irrelevantes

df_test = df_test.drop(columns = cols_irr)
# Coletando X_teste

X_test = df_test
# Transformando os dados para a mesma escala (entre 0 e 1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols = X_test.columns

X_test[cols] = scaler.fit_transform(X_test)

X_test.head()
def r2_est(X,y):

    return r2_score(y, modelo.fit(X,y).predict(X).clip(1,100))
def mean_absolute_error(y, y_pred):

    return (1 / len(y)) * np.sum(np.abs(y - y_pred))
# Selecionando modelo

from sklearn.ensemble import GradientBoostingRegressor

modelo = GradientBoostingRegressor()
model_fit = modelo.fit(X,y)
# Gerando as previsões para os dados de treino

predictions = model_fit.predict(X).clip(1,100)

predictions
print ('Baseline R2: %0.3f' %  r2_est(X,y))
print ('MAE test score: %0.3f' %  mean_absolute_error(y, predictions))
# Gerando as previsões para os dados de teste

predictions = model_fit.predict(X_test).clip(1,100)

predictions
# Gerando arquivo final com previsões



submission = pd.DataFrame()

submission['Property Id'] = index

submission['score'] = predictions.round().astype(int)



submission.to_csv('submission.csv', index = False)
submission.head(20)