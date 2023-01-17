import warnings

warnings.filterwarnings("ignore")



import pandas as pd

from pandas import DataFrame

import numpy as np

import math

import os

import random

from datetime import datetime, timedelta

import time

import matplotlib.pyplot as plt

import matplotlib as mpl

from IPython.display import Image

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge

from sklearn.utils import shuffle

import seaborn as sns   

from scipy.stats import boxcox

import warnings

import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics, preprocessing

%matplotlib inline
# Lendo o dataset de treino e teste



train = pd.read_csv("../input/dataset_treino.csv", 

                    parse_dates = True, low_memory = False)



test = pd.read_csv("../input/dataset_teste.csv", 

                    parse_dates = True, low_memory = False)





# Lendo o dataset com informacoes das lojas





lojas = pd.read_csv("../input/lojas.csv", 

                    low_memory = False)

print('O Dataframe de treino possui ' + str(test.shape[0]) + ' linhas e ' + str(test.shape[1]) + ' colunas')
print('O Dataframe de treino possui ' + str(train.shape[0]) + ' linhas e ' + str(train.shape[1]) + ' colunas')
print('O Dataframe de lojas possui ' + str(lojas.shape[0]) + ' linhas e ' + str(lojas.shape[1]) + ' colunas')
test.head(10)
##### Visualizando as primeiras 10 linhas do dataframe de lojas

lojas.head(10)
# examinar os tipos de dados e estatísticas descritivas 

print (train.info ()) 

print (train.describe ())
print (lojas.info ()) 

print (lojas.describe ())
# Verificando a existencia de dados missing no dataset de treino (dados faltantes)

train.isnull().sum()
train['Open'].value_counts().plot(kind='bar', figsize=(6,6))

plt.title('situação da loja')

plt.xlabel('Open')

plt.ylabel('Frequency')

plt.show()
test['Open'].value_counts().plot(kind='bar', figsize=(6,6))

plt.title('situação da loja')

plt.xlabel('Open')

plt.ylabel('Frequency')

plt.show()
##### Visualizando as primeiras 10 linhas do dataframe de traino

train.head(10)
plt.figure(figsize=(14,3))

Insulin_plt = train.groupby(train['Sales']).Open.count().reset_index()

sns.distplot(train[train.Open == 0]['Sales'], color='red', kde=False, label='Loja fechada')

sns.distplot(train[train.Open == 1]['Sales'], color='green', kde=False, label='Loja aberta')

plt.legend()

plt.title('Histograma dos valores das vendas, dependendo da situação da loja')

plt.show()
# Verificação de lojas fechadas

lojas_fechadas = train[(train.Open == 0) & ((train.Sales == 0))]
lojas_fechadas.head()
print('Existem ' + str(lojas_fechadas.shape[0]) + ' lojas fechadas')
# Verificando se existem lojas fechadas com vendas

train[(train.Open == 0) & (train.Sales != 0)].count()
lojas_aberta_sem_venda = train[(train.Open != 0) & (train.Sales == 0)]
lojas_aberta_sem_venda.head()
print('Existem ' + str(lojas_aberta_sem_venda.shape[0]) + ' lojas abertas sem vendas')
train.shape
print(train[(train["Open"] == 0)].head(10))
# Criando a coluna TicketMedio para verificar a relação entre vendas e consumidores

train['TicketMedio'] = train['Sales']/train['Customers']

train['TicketMedio2'] = train.groupby("Store")["TicketMedio"].mean()

train['TicketMedio2'] = (train["TicketMedio2"].fillna(train.groupby("Store")["TicketMedio"].transform("mean")))

train['TicketMedio'] = train['TicketMedio2']

# Criando a coluna com a média geral de Clientes de cada loja

train['MediaClientes'] = train.groupby("Store")["Customers"].mean()

train['MediaClientes'] = (train["MediaClientes"].fillna(train.groupby("Store")["Customers"].transform("mean")))

#

train.drop(['TicketMedio2'], axis = 1, inplace = True)

train['TicketMedio'].describe()
train.head(10).sort_values(by=['Store'], ascending=False)
# Verificando se existe algum valor missing

train.isnull().sum()
# Criando as Colunas mes e ano 

train['Year'] = pd.DatetimeIndex(train['Date']).year

train['Month'] = pd.DatetimeIndex(train['Date']).month
train["StateHoliday"].value_counts()
train["SchoolHoliday"].value_counts()
cleanup_nums = {"StateHoliday": {"a": 1, "b": 2, "c": 3}}
# Convertendo a coluna StateHoliday para números

train.replace(cleanup_nums, inplace=True)
train.info()
# Verificando a relação entre Vendas e Clientes

N = 2

colors = np.random.rand(N)

plt.scatter(train['Sales'], train['Customers'],  color='olive') 

plt.xlabel ('Sales')

plt.ylabel ('Customers')

plt.show() 
# Verificando a relação entre Ticket Médio e Clientes 

plt.scatter(train['Customers'], train['TicketMedio']) 

plt.xlabel ('Customers')

plt.ylabel ('TicketMedio')

plt.show() 
##### Visualizando as primeiras 10 linhas do dataframe de lojas

lojas.head(10)
print (lojas.info ()) 

print (lojas.describe ())
lojas["PromoInterval"].value_counts()
lojas["StoreType"].value_counts()
lojas["Assortment"].value_counts()
cleanup_nums = {"PromoInterval": {"Jan,Apr,Jul,Oct": 1, "Feb,May,Aug,Nov": 2, "Mar,Jun,Sept,Dec": 3},

                "StoreType": {"a": 1, "b": 2, "c": 3, "d": 4},

                "Assortment": {"a": 2, "b": 2, "c":3 }}
# Convertendo as colunas PromoInterval, StoreType e Assortment para números

lojas.replace(cleanup_nums, inplace=True)
lojas.head(10)
# Verificando a existencia de dados missing no dataset das lojas

lojas.isnull().sum()
lojas.groupby('Promo2')["Promo2SinceWeek","Promo2SinceYear","PromoInterval"].count()
lojas['Promo2SinceWeek'] = lojas.apply(lambda row: 0 if (pd.isna(row['Promo2SinceWeek'])) else row['Promo2SinceWeek'], axis=1)

lojas['Promo2SinceYear'] = lojas.apply(lambda row: 0 if (pd.isna(row['Promo2SinceYear'])) else row['Promo2SinceYear'], axis=1)

lojas['PromoInterval'] = lojas.apply(lambda row: 0 if (pd.isna(row['PromoInterval'])) else row['PromoInterval'], axis=1)
from sklearn.preprocessing import Imputer

imputer = Imputer().fit(lojas)

lojas_imputed = imputer.transform(lojas)
lojas_tratadas = pd.DataFrame(lojas_imputed, columns=lojas.columns.values)
lojas_tratadas.head(10)
# Verificando a existencia de dados missing no dataset das lojas_tratadas

lojas_tratadas.isnull().sum()
# Agora que já realizei a transformações necessárias nos dados, farei a junção dos datasets train e lojas

train_model = pd.merge(train, lojas_tratadas, how = 'left', on='Store')
train_model.head(10).sort_values(by=['Customers'], ascending=False)
# Eliminando as colunas Customers e Date

train_model = train_model.drop('Customers', axis=1)

train_model = train_model.drop('Date', axis=1)
# Analisando as correlações

corr = train_model.corr()

_ , ax = plt.subplots( figsize =( 16 , 14 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 10 })
Vendas = train_model['Sales']

train_model = train_model.drop(['Sales'], axis=1)

train_model['Sales'] = Vendas
train_model.shape
train_model.head(10)
test.head(10)
test.isnull().sum()
test["StateHoliday"].value_counts()
cleanup_nums = {"StateHoliday": {"a": 1, "b": 2, "c": 3}}
# Convertendo a coluna StateHoliday para números

test.replace(cleanup_nums, inplace=True)
test["Open"].value_counts()
test_nan = test[(test.Open.isnull())]
test_nan.head()
test_nan.isnull().sum()
test['Open'] = test.apply(lambda row: 1 if (pd.isna(row['Open'])) else row['Open'], axis=1)
test['Year'] = pd.DatetimeIndex(test['Date']).year

test['Month'] = pd.DatetimeIndex(test['Date']).month
test.head()
merge_test = train_model[['Store','TicketMedio','MediaClientes']]
test_merge = merge_test.groupby('Store')['TicketMedio','MediaClientes'].max()
test_merge = test_merge.reset_index()
test_merge.head()
test_merge2 = test.merge(test_merge, left_on='Store', right_on='Store')
test_merge3 = pd.merge(test_merge2, lojas_tratadas, how = 'left', on='Store')
test_model = test_merge3
test_model = test_model.drop('Date', axis=1)
test_model.head(10)
# Função para efetuar a Normalização

def normalize(df,columns,tipo):

    df = df.convert_objects(convert_numeric=True)

    result = df.copy()

    for feature_name in df.columns:

        if feature_name not in columns: 

            if tipo == 'min-max':

                max_value = df[feature_name].max()

                min_value = df[feature_name].min()

                if (max_value - min_value) == 0:

                    result[feature_name] = (df[feature_name] - min_value) / 1

                else:

                    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

            else: 

                if tipo == 'media':  

                    std_value = df[feature_name].std()

                    mean_value = df[feature_name].mean()

                    if std_value == 0:

                        result[feature_name] = ((df[feature_name] - mean_value) / 1)

                    else:

                        result[feature_name] = ((df[feature_name] - mean_value) / std_value)

    return result
#train_model.sort_index(axis=1,inplace=True)
# Creiando os datasets de treino e teste normalizados 

#train_model_N = normalize(train_model,["Store","Sales"],'min-max') 

#test_model_N  = normalize(test_model,["Store","Id"],'min-max') 

train_model_N = normalize(train_model,["Store","Sales"],'media') 

test_model_N  = normalize(test_model,["Store","Id"],'media') 

train_stores_model = dict(list(train_model_N.groupby('Store')))

test_stores_model = dict(list(test_model_N.groupby('Store')))
# Aplicando o algoritmo XGBoost com validação cruzada. 



# Após varias tentativas de melhorar o desenpenho do modelo, percebi que utilizando todo o conjunto de dados, 

# tanto de treino como de teste, meu melhor rmspe foi de 0,82723, então resolvi aplicar o modelo, considerando o

# conjunto de dados de cada loja individualemnte, o que fez melhorar significativamente a precisão do modelo.



import scipy.stats as st



one_to_left = st.beta(10, 1)  

from_zero_positive = st.expon(0, 50)



params = {  

    "n_estimators": st.randint(3, 40),

    "max_depth": st.randint(3, 40),

    "learning_rate": st.uniform(0.01, 0.4),

    "colsample_bytree": one_to_left,

    "subsample": one_to_left,

    "gamma": st.uniform(0, 10),

    'reg_alpha': from_zero_positive,

    'seed': [42],

    "min_child_weight": from_zero_positive,

}



fit_params = {'early_stopping_rounds' : [10]}



preds = pd.Series()

y_preds = list()

for i in test_stores_model:  

    store = train_stores_model[i]

    X_train = store.drop(["Sales", "Store"],axis=1)

    y_train = store["Sales"]

    X_test  = test_stores_model[i].copy()   

    store_ind = X_test["Id"]

    X_test.drop(["Id","Store"], axis=1,inplace=True)

    X_train.sort_index(axis=1,inplace=True)

    X_test.sort_index(axis=1,inplace=True)

    xg_train = xgb.DMatrix(X_train, label=y_train)

    xg_test  = xgb.DMatrix(X_test)

    bst = xgb.XGBRegressor()

    print('Treinando a loja ' + str(i))

    gs = RandomizedSearchCV(bst, params, n_jobs=1)  

    gs.fit(X_train, y_train)

    y_pred = gs.predict(X_test)

    y_pred = y_pred.tolist()

    preds = preds.append(pd.Series(y_pred, index=store_ind))    
preds = pd.DataFrame({ "Id": preds.index, "Sales": preds.values})
# Salvando 

preds.to_csv("sample_submission.csv", sep=',', index=False)