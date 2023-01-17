# Importação de bibliotecas



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
# Importação do arquivo

df  = pd.read_csv('../input/autos.csv', sep=',', encoding='Latin1')



# Lista detalhes do arquivo

df.info()
#Inspecionando o vendedor

print(df.groupby('seller').size())
# Removendo gewerblich

df = df[df.seller == 'privat']

df=df.drop('seller',1)



# Analisando tipo da oferta

print(df.groupby('offerType').size())
df = df[df.offerType == 'Angebot']

df=df.drop('offerType',1)
# Ins[[ecionando 'price'.

# Aplicado função para remover a notação científica ao exibir o valor.

df['price'].describe().apply(lambda x: format(x, 'f'))
# Plotando a distribuição do valor

plt.figure(figsize=(12,6))

sns.distplot(df['price'])

plt.xlabel('Valor')

plt.ylabel("Quantidade")

plt.title("Distribuição do valor do veículo")
# Filtrando dados pelo preço

df = df[df.price > 100]

df = df[df.price < 100000]



# Inspecionando a variável 'price' filtrada

df['price'].describe().apply(lambda x: format(x, 'f'))
# Plotando o resultado filtrado

plt.figure(figsize=(12,6))

sns.distplot(df['price'])

plt.xlabel('Valor')

plt.ylabel("Quantidade")

plt.title("Distribuição do valor do veículo")
# Inspecionando a mariável 'yearOfRegistration'

df['yearOfRegistration'].describe()
# Filtrando dados por ano

df = df[(df.yearOfRegistration >= 1970) & (df.yearOfRegistration < 2019)]



# Inspecionando os dados filtrados

df['yearOfRegistration'].describe()
# Inspecionando variável 'powerPS'

print(df['powerPS'].describe())
# Filtrando vareiável 'powerPS'

df = df[(df.powerPS > 50) & (df.powerPS < 900)]



# Resultado da variável

print(df['powerPS'].describe())
# Top 10 Maiores carros por potência

df[['name','powerPS','price']].sort_values(by=['powerPS'], ascending=False).head(10)
# Remove linhas que contenham valores nulo

df = df.dropna()



# Exibe as informações atuais do data frame

df.info()
# Printa exemplo dos dados

df.head(5)
# Transformando dummies

dummy_vehicleType = pd.get_dummies(df['vehicleType'])

df = pd.concat([df, dummy_vehicleType], axis = 1)



dummy_gearbox = pd.get_dummies(df['gearbox'])

df = pd.concat([df, dummy_gearbox], axis = 1)



dummy_fuelType = pd.get_dummies(df['fuelType'])

df = pd.concat([df, dummy_fuelType], axis = 1)



dummy_brand = pd.get_dummies(df['brand'])

df = pd.concat([df, dummy_brand], axis = 1)



dummy_notRepairedDamage = pd.get_dummies(df['notRepairedDamage'])

df = pd.concat([df, dummy_notRepairedDamage], axis = 1)



df.head(3)
# Importando train_test_split

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, random_state=66)



train.shape, test.shape
# Importação do modelo

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=66, n_estimators=200, n_jobs=-1)
# Lista de coluna a serem removidas do modelo

removed_cols = [

    'dateCrawled',

    'name',

    'abtest',

    'lastSeen',

    'nrOfPictures',

    'dateCreated',

    'notRepairedDamage',

    'brand',

    'fuelType',

    'monthOfRegistration',

    'model',

    'gearbox',

    'vehicleType',

    'postalCode',

    'price']



# Diferença entre as colunas a serem removidas e as demais, gera a lista de features, 

# que são as colunas que serão usadas no modelo.

feats = [c for c in df.columns if c not in removed_cols]
# Treinamento do modelo

rf.fit(train[feats], train['price'])

preds = rf.predict(test[feats])
# Importação do modelo de erro

from sklearn.metrics import mean_squared_log_error

mean_squared_log_error(test['price'], preds)
# Criando uma coluna com o valor rpevisto na base de teste.

test['price_predict'] = rf.predict(test[feats])
# Exibindo alguns resultados comparados: preditos x reais

test[['price', 'price_predict']].head(30)
# Exemplo radon de dados que foram preditos

sample_data=test.sample(n=200, random_state=66)



# Plota geáfico que compara valores reais x valores preditos

sns.regplot(x="price", y="price_predict", data=sample_data)

plt.xlabel('Preço Real')

plt.ylabel("Preço Predito")

plt.title("Preço real x Preço Predito")
# Correlação entre os dados

df[feats[:20]].corr()
# Correlação

f,ax = plt.subplots(figsize=(13,15))

sns.heatmap(df[feats[:20]].corr(), annot=True, fmt='.2f', ax=ax, linecolor='black', lw=4)

plt.title("Matriz de Correlação")
# Avaliando a importância das variáveis



pd.Series(rf.feature_importances_[:15], index=feats[:15]).sort_values().plot.barh()

plt.xlabel('Importância em %')

plt.ylabel("Variável")

plt.title("Importância das variáveis")
# Cálculo do R2 do modelo

score = rf.score(train[feats], train['price'])

print(score)