# Modelo de regressao supervisionada



# prevendo a medida de desempenho energetico de predios (ENERGY STAR Score)



# importando os modulos

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import warnings

from math import isnan

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings('ignore')

#%matplotlib inline
# Lendo o conjunto de dados de treino e validacao

dataset = pd.read_csv('../input/dataset_treino.csv')

validacao = pd.read_csv('../input/dataset_teste.csv')
# Dimensao dos dados

print("Dataset: ",dataset.shape) # Possui a variavel target (ENERGY STAR Score)

print("Validacao: ",validacao.shape)
# Criando um objeto para receber os ID dos dados de validacao para submissao

label = validacao['Property Id']



# Excluindo os ID dos conjuntos de dados

dataset = dataset.drop('Order', axis = 1)

validacao = validacao.drop('OrderId', axis = 1)



# Criando o atributo ENERGY STAR Score com valor NA para concatenar os dados

validacao['ENERGY STAR Score'] = np.nan



# Concatenando os dados para tratamento e exploracao

df = pd.concat([dataset, validacao], ignore_index=True)
# Visualizando o shape do novo conjunto de dados

df.shape
# Visualizando as primeiros linhas dos dados

df.head().T
# Verificando os tipos de dados

# Existem atributos numericos que estao classificados como strings

# pois estao com valores not avaliable que serao subistituidos por zero

df.dtypes
# Convertendo release date para o tipo data

df['Release Date'] = pd.to_datetime(df['Release Date'], format = '%m/%d/%Y %I:%M:%S %p')



# Criando um atributo com o mes 

df['month'] = df['Release Date'].dt.month
# lista com os atributos numericos

atr_num = ['2nd Largest Property Use - Gross Floor Area (ft²)','3rd Largest Property Use Type - Gross Floor Area (ft²)',

             'Census Tract','Community Board','Council District','DOF Gross Floor Area','Diesel #2 Use (kBtu)',

             'Direct GHG Emissions (Metric Tons CO2e)','District Steam Use (kBtu)','ENERGY STAR Score',

             'Electricity Use - Grid Purchase (kBtu)','Fuel Oil #1 Use (kBtu)','Fuel Oil #2 Use (kBtu)',

             'Fuel Oil #4 Use (kBtu)','Fuel Oil #5 & 6 Use (kBtu)','Indirect GHG Emissions (Metric Tons CO2e)',

             'Largest Property Use Type - Gross Floor Area (ft²)','Natural Gas Use (kBtu)',

             'Number of Buildings - Self-reported','Occupancy','Property GFA - Self-Reported (ft²)',

             'Site EUI (kBtu/ft²)','Source EUI (kBtu/ft²)','Total GHG Emissions (Metric Tons CO2e)',

             'Water Intensity (All Water Sources) (gal/ft²)','Water Use (All Water Sources) (kgal)',

             'Weather Normalized Site EUI (kBtu/ft²)','Weather Normalized Site Electricity (kWh)',

             'Weather Normalized Site Electricity Intensity (kWh/ft²)','Weather Normalized Site Natural Gas Intensity (therms/ft²)',

             'Weather Normalized Site Natural Gas Use (therms)','Weather Normalized Source EUI (kBtu/ft²)','Year Built', 'month']



# Criando um subset somente com os dados numericos

numericos = df[atr_num]



# Subistituindo os valores Not Available para zero (0)

numericos = numericos.replace('Not Available', 0)



# Convertendo os valores para tipo numerico

numericos = numericos.apply(pd.to_numeric)



# Checando os tipos

numericos.dtypes
# Buscando por valores missing nos atributos numericos

# os valores NA do atributo ENERGY STAR Score se refere aos dados devalidacao

numericos.isnull().sum().sort_values(ascending = False)
# Os atributos 'Census Tract', 'Community Board','Council District' serao excluidos

# pois existem muitos valores NA

numericos = numericos.drop(['Census Tract', 'Community Board','Council District'], axis = 1)



# Quanto ao atributo DOF Gross Floor Area irei subistituir os valores na pela mediana

# Calculando a mediana

mediana = numericos['DOF Gross Floor Area'].median()



# Subistituindo os valores NA pela mediana

numericos['DOF Gross Floor Area'] = numericos['DOF Gross Floor Area'].fillna(mediana)
# Dados tratados

numericos.isnull().sum().sort_values(ascending = False).head(5)
# Sumarizando os atributos numericos

numericos.describe().T
# Analisando os atributos do tipo string

# Criando um subset somente com os dados do tipo texto

texto = df.drop(atr_num, axis = 1)



# Checando os tipos

# Latitude e longitude estao como tipo numericos, mas serao descondirados

# assim como demais atributos ID e enderecos

texto = texto.drop(['Latitude','Longitude','Street Number','Street Name', 'BBL - 10 digits',

                   'Parent Property Id', 'Borough',

                   'Parent Property Name', 'Postal Code', 'Property Id', 'DOF Benchmarking Submission Status',

                   'Address 1 (self-reported)','Address 2', 'NYC Borough, Block and Lot (BBL) self-reported',

                   'NYC Building Identification Number (BIN)', 'Property Name', 'Release Date',

                   'NTA'], axis = 1)

texto.dtypes
# Visualizando os dados do tipo texto

texto.head().T
# O atributo 'List of All Property Use Types at Property' possui varios valores dentro da mesma variavel

# sera criado uma matriz com os atributos de cada linha



# lista com os valores unicos do atributo

lista = texto['List of All Property Use Types at Property'].unique()



# Dividindo os valores separados por virgula

lista = [np.chararray.split(item, ',').tolist() for item in lista]



# Deixando a lista apenas com valores unicos

tipos = list(itertools.chain.from_iterable(lista))

tipos = list(set(tipos))



# imprimindo os tipos (10 primeiros itens)

tipos[:10]
# Removendo o espaco em branco antes dos valores

tipos = [item.lstrip() for item in tipos]



# Novamente deixando os dados apendas com valores unicos

tipos = list(set(tipos))



# removendo o nome (etc.)

tipos.remove('etc.)')



# Dados tratados

tipos[:10]
# Criando uma matriz com valores zeros para tratar os atributos da lista



# definindo o tamanho da matriz

linhas = df.shape[0]

colunas = len(tipos)



# criando a matriz de zeros e convertendo para o tipo data frame

df_tipos = np.zeros([linhas, colunas], dtype = int)

df_tipos = pd.DataFrame(data = df_tipos, columns=tipos)



# Matriz criada

df_tipos.iloc[:,:10].head().T
# Lista com os valores do atributo list of property

list_prop = texto['List of All Property Use Types at Property']



# Preenchendo a matriz com o valor (um) quando existe o item da lista na coluna

colunas = df_tipos.columns

for item in colunas:

    df_tipos[item] = [1 if i in item else 0 for i in list_prop]

    

# Data frame tratado

df_tipos.head(20).T
# Verificando por valores NA

texto.isnull().sum().sort_values(ascending = False)
# Criando um objeto que ira receber o valor que mais se repete no atributo 'Water Required?

moda = texto['Water Required?'].mode().values[0]



# Substituind valores NA pela Moda

texto['Water Required?'] = texto['Water Required?'].fillna(moda)



# Nova checagem dos valores NA

texto.isnull().sum().sort_values(ascending = False).head()
# Resumo dos dados

texto.describe().T
# removendo a list property do dataset

texto = texto.drop('List of All Property Use Types at Property', axis = 1)



# convertendo os dados para numericos

le = LabelEncoder()

for column in texto.columns:

    texto[column] = le.fit_transform(texto[column])

          



# Visualizando os dados tratados

texto.head().T
# Concatenando os dados numericos e textos

# Criando um novo dataset com os dados tratados

df_tratados = pd.concat([texto, df_tipos, numericos], axis = 1)



# Conferindo o shape dos dados

print(df.shape)

print(df_tratados.shape)
# objeto com os nomes dos atributos para criar um novo data frame apos a normalizacao

nomes = df_tratados.drop('ENERGY STAR Score', axis = 1).columns



# Dividindo os dados para normalizacao

X = df_tratados.drop('ENERGY STAR Score', axis = 1)

y = df_tratados['ENERGY STAR Score']



# Normalizando os dados

scala = StandardScaler().fit_transform(X)



# Criando o novo dataframem

df_normal = pd.DataFrame(data = scala, columns = nomes)

df_normal['ENERGY STAR Score'] = y



# Checando a transformacao

df_normal.head().T
# Separando os dados de treino e validacao para criar o modelo

validacao_new = df_normal.iloc[6622:,:]

dataset_new = df_normal.iloc[:6622,:]
# Extracao de atributos por importancia



# Separando os dados para criar o modelo

X = dataset_new.drop('ENERGY STAR Score', axis = 1).values

nomes = dataset_new.drop('ENERGY STAR Score', axis = 1).columns

y = dataset_new['ENERGY STAR Score'].values



# Criando o modelo

modelo_ft = XGBRegressor().fit(X,y)



# Visualizando os atributos por importancia

features = pd.DataFrame({"Atributos" : nomes, "Score" :modelo_ft.feature_importances_}).sort_values(by = 'Score', ascending=False)

features.head(10)
# Separando os atributos para criar o modelo

X = dataset_new.drop('ENERGY STAR Score', axis = 1).values

y = dataset_new['ENERGY STAR Score'].values



# Conjuntos de dados de treino e de teste

x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 8)



# Preparando os modelo

modelos = []

modelos.append(('LR', LinearRegression()))

modelos.append(('RDG', Ridge()))

modelos.append(('LASSO', Lasso()))

modelos.append(('ELAST', ElasticNet()))

modelos.append(('KNN', KNeighborsRegressor()))

modelos.append(('DECISION', DecisionTreeRegressor()))

modelos.append(('SVR', SVR()))

modelos.append(('XGB', XGBRegressor()))

modelos.append(('RF', RandomForestRegressor()))

modelos.append(('GB', GradientBoostingRegressor()))

modelos.append(('ET', ExtraTreesRegressor()))



# loop com varios modelos para testar o de melhor resultado

for nome, modelo in modelos:

    modelo.fit(x_treino,y_treino)

    

    # Prevendo os dados

    previsao = modelo.predict(x_teste)

    

    # avaliando o erro absoluto medio

    print(nome, mean_absolute_error(y_teste, previsao))
# otimizando o modelo



#Selecionando os 9 melhores atributos

feat_best = features.iloc[:,0].tolist()[:9]



# Separando os dados para criar o modelo

X = dataset_new[feat_best].values

y = dataset_new['ENERGY STAR Score'].values



# Conjuntos de dados de treino e de teste

x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.20, random_state = 8)



# Criando o modelo

modelo_xgb = XGBRegressor().fit(x_treino,y_treino)



# Prevendo os dados

previsao = modelo_xgb.predict(x_teste)



# avaliando o erro absoluto medio

print(mean_absolute_error(y_teste, previsao))



# avaliando o R2

r2_score(y_teste, previsao)
# otimizando o modelo - definindo parametros



#Selecionando os 9 melhores atributos

feat_best = features.iloc[:,0].tolist()[:9]



# Separando os dados para criar o modelo

X = dataset_new[feat_best].values

y = dataset_new['ENERGY STAR Score'].values



# Conjuntos de dados de treino e de teste

x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.20, random_state = 8)



modelo_xgb_best = XGBRegressor(learning_rate = 0.2,

                               max_depth = 4,

                               max_features = 1.0,

                               min_samples_leaf = 3,

                               eval_metric = 'mae').fit(x_treino, y_treino)

# Prevendo os dados

previsao = modelo_xgb_best.predict(x_teste)



# avaliando o erro absoluto medio

print(mean_absolute_error(y_teste, previsao))



# avaliando o R2

r2_score(y_teste, previsao)
# Criando objeto com os atributos preditores dos dados de valicadao

val = validacao_new[feat_best].values



# prevendo os dados

y_val = modelo_xgb_best.predict(val)



# Arredondando os dados e tratando valores maiores que 100 e menores que zero

y_val = [0 if item < 0 else 100 if item > 100 else round(item,0) for item in y_val]
# Gerando dataset para submissao

df_val = pd.DataFrame({'Property Id' : label, 'score' : y_val})

df_val.head()
# gravando o arquivo para submissao

df_val.to_csv('submissao4.csv',index=False)