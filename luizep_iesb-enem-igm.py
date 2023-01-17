import matplotlib.pyplot as plt # Bibilioteca util para criar gráficos

import pandas as pd # Bibilioteca para auxiliar a importar e maniular nossos dataframes

from sklearn.tree import DecisionTreeClassifier #responsável pela geração do modelo 

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

import numpy as np # Bibilioteca útil para realizar operações matemáticas

import seaborn as sns # Bibilioteca utilizada para dar um toque especial nos gráficos

#import chardet   #Trabalha com leitura de arquivos, acredito que n será necessário utiliza=lá

plt.style.use('ggplot') #Customização de gráficos

plt.style.use("seaborn-white")

import os

import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
colors = [['#0D47A1','#1565C0','#1976D2','#1E88E5','#2196F3'],

          ['#311B92','#512DA8','#673AB7','#9575CD','#B39DDB'],

          ['#1B5E20','#388E3C','#4CAF50','#81C784','#66BB6A'],

          ['#E65100','#EF6C00','#F57C00','#FB8C00','#FF9800'],

          ['#3E2723','#4E342E','#5D4037','#6D4C41','#795548'],

          ['#BF360C','#D84315','#E64A19','#F4511E','#FF5722'],

          ['#880E4F','#AD1457','#C2185B','#D81B60','#E91E63']

         ]
train = '../input/train.csv'

test = '../input/test.csv'



dftrain = pd.read_csv(train)

dftest = pd.read_csv(test)
dftrain.head(2)
dftest.head(2)
#Dimensão  datasets

print("Dimensão  dataset de treino")

print("Colunas:", dftrain.shape[1],"\nlinhas:", dftrain.shape[0])

print("-")

print("Dimensão  dataset de teste")

print("Colunas:", dftest.shape[1],"\nlinhas:", dftrain.shape[0])
#Checando o tipo dos dados treino

dftrain.dtypes
#Checando o tipo dos dados teste

dftest.dtypes
#Exibindo as 10 primeiras linhas como matriz transposta

dftrain.head(10).T
#Indicando que quero remover a coluna inteira e salvando a alteração direto no dataset

#dftrain.drop(['Unnamed: 0','porte', 'populacao', 'area', 'participacao_transf_receita','servidores', 'comissionados',

# 'comissionados_por_servidor','perc_pop_econ_ativa', 'taxa_empreendedorismo','anos_estudo_empreendedor', 'gasto_pc_saude',

# 'hab_p_medico','exp_vida',], axis=1, inplace=True)



#dftest.drop(['Unnamed: 0','porte', 'populacao', 'area', 'participacao_transf_receita','servidores', 'comissionados',

# 'comissionados_por_servidor','perc_pop_econ_ativa', 'taxa_empreendedorismo','anos_estudo_empreendedor', 'gasto_pc_saude',

# 'hab_p_medico','exp_vida',], axis=1, inplace=True)
#Listando novamente as colunas para validar a remoção

#dftrain.columns
#Agrupando Municipios e suas respectivas notas médias

#dftrain.groupby('municipio')['nota_mat'].mean()
#Calculando a simetria dos dados

#Um valor zero indica uma distribuição simétrica, um valor maior que zero ou menor indica uma distribuição assimétrica.

dftrain.skew()
dftrain.groupby('estado')['idhm'].mean().sort_values().plot(kind='barh', figsize=(15,8), title="IDH-M por estado", grid=True)

plt.xlabel("IDH-M")

plt.ylabel("ESTADO")

plt.show()
#dftrain.groupby('estado')['densidade_dem'].mean()

dem = dftrain[['densidade_dem', 'regiao']].copy()

#dem['densidade_dem'] = dem['densidade_dem'].values.astype('int')



#convertendo o tipo de dados das colunas do dataset de cópia 'dem'

dem['densidade_dem'] = dem['densidade_dem'].astype('category').cat.codes

dem.groupby('regiao')['densidade_dem'].mean().sort_values().plot(kind='barh', figsize=(15,8),grid=True, legend=True, color=random.choice(colors), title="Densidade demográfica por região")

plt.ylabel("REGIÃO")

plt.xlabel("DENSIDADE DEMOGRÁFICA")

plt.show()
explode = (0.1, 0, 0, 0,0)

dftrain.groupby('regiao')['populacao'].mean().plot(kind='pie',labeldistance=1.1, explode=explode,autopct='%1.0f%%', title="Percentual médio da distribuição da população por região", shadow=True, startangle=90)

plt.ylabel(" ")

plt.show()
ax1 = dftrain.plot(kind='scatter', x='exp_anos_estudo', y='nota_mat', color='r',edgecolor='black',figsize=(15,5))    

ax2 = dftrain.plot(kind='scatter', x='gasto_pc_educacao', y='nota_mat', color='g', edgecolor='black',figsize=(15,5))    

ax3 = dftrain.plot(kind='scatter', x='jornada_trabalho', y='nota_mat', color='b', edgecolor='black',figsize=(15,5))    
#dftrain.groupby((dftrain.nota_mat >  500 ))['estado'].count()
sp = dftrain[dftrain['estado'] == 'SP']

sp.groupby('municipio')['nota_mat'].mean().sort_values(ascending=False).head(20).plot(kind='barh',rot=0, figsize=(15,8),color=random.choice(colors), title="Top 20 - Maior nota média de matemática por município - SP")

plt.xlabel("Nota")

plt.ylabel("Município")

plt.show()

dftrain.groupby(('municipio','estado', 'regiao'))['nota_mat'].mean().sort_values(ascending=False).head(20)
dftrain.groupby(('municipio','estado', 'regiao'))['nota_mat'].mean().sort_values(ascending=False).head(20).plot(kind="barh", figsize=(15,8),color=random.choice(colors))

plt.xlabel("Nota")

plt.show()
dftrain.groupby(('municipio','estado', 'regiao'))['nota_mat'].mean().sort_values(ascending=True).head(20).plot(kind="barh",figsize=(15,8),color=random.choice(colors), title="Top 20 Menor nota média em matemática por município")

plt.xlabel("Nota")

plt.show()
dftrain.groupby('estado')['nota_mat'].describe()
dftrain['nota_mat'].describe()
#sns.distplot(dftrain['nota_mat'].dropna(),kde=False,color='orange',bins = 20)

#plt.show()
dftrain[['nota_mat','exp_anos_estudo', 'jornada_trabalho','exp_vida']].hist(bins=20,alpha=0.5,color='Green')

plt.show()
# Quartil · Percentil 

#plt.figure(figsize=(5,5))

sns.boxplot(y='nota_mat',data=dftrain,palette='winter')

plt.show()
#dftrain.groupby('exp_vida')['nota_mat'].mean().plot(figsize=(16,8) ,marker='.', color="green")

#plt.ylabel("nota_mat")

#plt.title("Experiência de vida vs nota de matemática")

#plt.show()
#Exibindo as notas de matemática com o hist

#dftrain['nota_mat'].hist(color='green',bins=40,figsize=(8,4))

#plt.show()
media_mt_estado = dftrain.groupby('estado')['nota_mat'].max().sort_values()

media_mt_estado.plot(title = 'Maiores notas de matemática por estado', grid = False, kind='barh',color='black', figsize=(15,8))

plt.xlabel('Nota')

plt.ylabel('Estado')

plt.show()
contagem_nota = dftrain.groupby('estado')['nota_mat'].mean().head(10).sort_values().plot(kind = 'barh',color='black', grid= False, figsize=(15,3))

for ix in contagem_nota.patches:

    ia =ix.get_bbox()

    contagem_nota.annotate("{: .0f}".format(ia.x1 + ia.x0), (ix.get_width() + 1 , ix.get_y() - 0.05))

plt.ylabel("Estado")

plt.xlabel("Nota")

plt.title("TOP 10 - Maiores notas médias em matemática classificadas por estado")

plt.show()

#print("Figura 02")
contagem_not = dftrain.groupby('regiao')['nota_mat'].mean().sort_values().plot(kind = 'barh',color='black', grid= False, figsize=(15,3))

for ix in contagem_nota.patches:

    ia =ix.get_bbox()

    contagem_nota.annotate("{: .0f}".format(ia.x1 + ia.x0), (ix.get_width() + 1 , ix.get_y() - 0.05))

plt.ylabel("Estado")

plt.xlabel("Nota")

plt.title("Nota média de matemática classificada por região")

plt.show()
#visualizando o estado com a maior nota em matemática

#media_mt_municipio = dftrain.groupby('municipio')['nota_mat'].mean().head(20).sort_values()

#media_mt_estado.plot(title = 'TOP 20 - Maiores notas médias em matemática classificadas por estado', grid = False, kind='barh', figsize=(15,10), color='black')

#plt.xlabel('Nota')

#plt.ylabel('Estado')

#plt.show()
#fig, ax = plt.subplots(figsize=(13,7))

#sns.violinplot(x="municipio", y="codigo_mun", hue='nota_mat', data=dftrain, split=True, bw=0.05 , ax=ax)

#plt.title('Survivals for Age and Pclass ')

#plt.show()
#plt.subplots(figsize=(10,6))

#sns.barplot(x='nota_mat' , y='exp_vida' , data = dftrain)

#plt.ylabel("Survival Rate")

#plt.title("Survival as function of NameLenBin")

#plt.show()
#Criando uma função veirificando

#def estado_mun(municipio_):

#    municipio_es = dftrain[dftrain['municipio'] == municipio_ ]

#    municipio_es = municipio_es.groupby('estado')[['municipio', 'nota_mat']].mean()

#    return municipio_es
#media_mt_municipio = dftrain.groupby('municipio')['nota_mat'].mean().head(20).sort_values()
#Verificando valores nulos no dataset de treino

dftrain.isnull().sum().sort_values(ascending=False)
#Plotando as colunas com maior incidência de valores nulos

sns.heatmap(dftrain.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
#Verificando valores nulos no dataset de teste

dftest.isnull().sum().sort_values(ascending=False).head(10)
sns.heatmap(dftest.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
#Preenchendo valores nulos do dataset de treino

#dftrain['nota_mat'].fillna(dftrain['nota_mat'].mean, inplace=True)

#dftrain['gasto_pc_educacao'].fillna(dftrain['gasto_pc_educacao'].mean, inplace=True)

#dftrain['indice_governanca'].fillna(dftrain['indice_governanca'].mean, inplace=True)

#dftrain['ranking_igm'].fillna(dftrain['ranking_igm'].mean, inplace=True)

#dftrain['idhm'].fillna(dftrain['idhm'].mean, inplace=True)

#dftrain['densidade_dem'].fillna(dftrain['densidade_dem'].mean, inplace=True)

#dftrain['nota_redacao'].fillna(dftrain['nota_redacao'].mean, inplace=True)

#dftrain['nota_linguagem'].fillna(dftrain['nota_linguagem'].mean, inplace=True)

#dftrain['nota_humanas'].fillna(dftrain['nota_humanas'].mean, inplace=True)

#dftrain['nota_ciencias'].fillna(dftrain['nota_ciencias'].mean, inplace=True)

#dftrain['exp_anos_estudo'].fillna(dftrain['exp_anos_estudo'].mean, inplace=True)
#Preenchendo valores nulos do dataset de teste

#dftest['indice_governanca'].fillna(dftest['indice_governanca'].mean, inplace=True)

#dftest['ranking_igm'].fillna(dftest['ranking_igm'].mean, inplace=True)

#dftest['gasto_pc_educacao'].fillna(dftest['gasto_pc_educacao'].mean, inplace=True)

#dftest['idhm'].fillna(dftest['idhm'].mean, inplace=True)
#Preenchendo valores nulos

dftrain.fillna(1, inplace=True)

dftest.fillna(1, inplace=True)
#Verificando novamente os valores nulos no dataset de treino

dftrain.isnull().sum().sort_values(ascending=False).head(10)
#Verificando novamente os valores nulos no dataset de teste

dftest.isnull().sum().sort_values(ascending=False).head(10)
#Transformando o tipo de dados do dataset para aplicar o modelo de ml

for col in dftrain.columns:

    if dftrain[col].dtypes == 'object':

        dftrain[col] = dftrain[col].astype('category').cat.codes
#Criando uma cópia do dataset de teste para utilização final

codtest  = dftest.copy()
#Preenchendo valores nulos

codtest.fillna(1, inplace=True)
#Só irei precisar disso

codtest['codigo_mun'].head()
#Transformando o tipo de dados do dataset de teste para aplicar o modelo de ml

for col in dftest.columns:

    if dftest[col].dtypes == 'object':

        dftest[col] = dftest[col].astype('category').cat.codes
#dftest['codigo_mun'] = dftest['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))
dftrain.info()
dftrain.dtypes
from sklearn.metrics import mean_squared_error
feats = [c for c in dftrain.columns if c not in ['nota_mat']]
#feats
#Treinando o modelo

#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

#A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the

#dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always 

#the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).



#Tunning the model 



#n_estimators = number of trees in the foreset

#max_features = max number of features considered for splitting a node

#max_depth: É a profundida máxima da árvore, profundida demais pode gerar um sistema super especializado nos dados de treinamento, 

#também conhecido como overfitting. Profundida de menos vai diminuir a capacidade de generalização do modelo.

#min_samples_split = min number of data points placed in a node before the node is split

#min_samples_leaf = min number of data points allowed in a leaf node

#bootstrap = method for sampling data points (with or without replacement)

#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74





rf = RandomForestRegressor(n_estimators=500, random_state=None, n_jobs=-1,max_depth=1, max_features=1)

rf
# Atribuindo a variável y a coluna 'nota_mat'

y = 'nota_mat'
rf.fit(dftrain[feats], dftrain[y])
# Criando o conjunto de treino e validação

train, valid = train_test_split(dftrain,random_state=0, shuffle=False)
x = dftrain[feats]

y = dftrain[y]
#tree = RandomForestRegressor(n_estimators=80, random_state=80, n_jobs=-1, max_depth=10)
rf.fit(x,y)
rf.score(x,y)
valid_preds = rf.predict(valid[feats])
valid_preds
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
for col in dftest.columns:

    if dftest[col].dtypes == 'object':

        dftest[col] = dftest[col].astype('category').cat.codes
for col in dftrain.columns:

    if dftrain[col].dtypes == 'object':

        dftrain[col] = dftest[col].astype('category').cat.codes
#dftest['codigo_mun'] = dftest['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))
#dftest.dtypes
#Separando o dataset para submeter ao kaggle

submission = pd.DataFrame()

submission['codigo_mun'] = codtest['codigo_mun']

submission['nota_mat'] = rf.predict(dftest)
submission['codigo_mun'] = submission['codigo_mun'].apply(lambda x: x.replace('ID_ID_', ''))

submission['codigo_mun'] = submission['codigo_mun'].values.astype('int64')
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh(figsize=(6,6))

plt.show()
submission.dtypes
submission.to_csv('submission.csv', index=False)
dftrain['nota_mat'].describe().loc[['mean', 'max', 'min']]
submission['nota_mat'].describe().loc[['mean', 'max', 'min']]
codtest.shape
x_test = dftrain['nota_mat'].describe().loc[['mean', 'max', 'min']]

x_predict = submission['nota_mat'].describe().loc[['mean', 'max', 'min']]



plt.figure(figsize=(10, 5))

plt.scatter(x_predict,x_test)

plt.title('Predicted vs. Actual')

plt.xlabel('Nota atual - test')

plt.ylabel('Notas preditas - submission')

plt.plot([min(x_test), max(x_test)], [min(x_predict), max(x_predict)])