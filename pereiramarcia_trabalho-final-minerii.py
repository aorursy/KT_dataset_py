# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Carregamento do Dataset:
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
df.shape
# Visualizando os dados:
df.head()
# Verificando uma amostra aleatória
df.sample(5)
# Verificando os tipos dos dados e os tamanhos
df.info()
# Verificando quantas observações possuem esta característica: 
df[df['MORTDUE'].isnull() & df['VALUE'].isnull()]
# Usando o comando drop para excluir os registros que possuem valores nulos nas 2 varíaveis com dados 
#sobre o atual financiamento.
df.dropna(how='all', subset = ['MORTDUE', 'VALUE'], inplace=True)
# conferindo se o comando de exclusão deu certo: 
df[df['MORTDUE'].isnull() & df['VALUE'].isnull()]
# verificando a quantidade de casos enquadrados nesta situação (nulo em MORTDUE ou VALUE:
df[df['MORTDUE'].isnull() | df['VALUE'].isnull()]
# imputando o valor da Propriedade pelo valor da dívida:
df['VALUE'] = df.apply(lambda row: row['MORTDUE'] if np.isnan(row['VALUE']) else row['VALUE'],axis=1)
# imputando o valor da dívida pelo valor da propriedade: 
df['MORTDUE'] = df.apply(lambda row: row['VALUE'] if np.isnan(row['MORTDUE']) else row['MORTDUE'],axis=1)
# Conferindo que não existem mais registros com valores nulos nas colunas MORTDUE e VALUE:
df[df['MORTDUE'].isnull() | df['VALUE'].isnull()]
# Verificando os primeiros dados após efeito da imputação
df.head(10)
# Verificando como ficou o Dataset após esta primeira imputação:
df.info()
# Verificando a quantidade de dados nulos:
df[df['REASON'].isnull()]
# Verificando as classes da variável 'REASON' e a frequência de cada um delas.
df['REASON'].value_counts()
# Verificando as proporção dos dados distribuídos entre as variáveis 'REASON' e 'BAD':
totals=pd.crosstab(df['REASON'],df['BAD'],margins=True).reset_index()
percentages = pd.crosstab(df['REASON'],
   df['BAD']).apply(lambda row: row/row.sum(),axis=1).reset_index()
totals


# Verificando os percentuais cruzados entre 'REASON' e 'BAD':
percentages
# Realizando a imputação da variável 'REASON':
df['REASON'].fillna('DebtCon', inplace=True)

# Verificando como ficou a distribuição de frequências da variável 'REASON':
df['REASON'].value_counts()
# Verificando como ficou o Dataset:
df.info()
# Verificando as classes e frequência da variável 'JOB':
df['JOB'].value_counts()
# Verificando os registros nulos na variável 'JOB':
df[df['JOB'].isnull()]
# Imputação dos valores nulos da variável 'JOB'pelo valor da classe 'Other':
df['JOB'].fillna('Other', inplace=True)
# Verificando como ficou a distribuição das classes da variável 'JOB':
df['JOB'].value_counts()
#Verificando como ficou o Dataset:
df.info()
# Verificando a distribuição da variável:
df['YOJ'].value_counts()
# Verificando os dados nulos da variável 'YOJ':
df[df['YOJ'].isnull()]
# Verificando o resumo de medidas estísiticas da variável 'YOJ':
df['YOJ'].describe()
# Verificando uma medida estatística extra, a mediana:
df['YOJ'].median()
# Verificando a distribuição dos dados por meio do Histograma:
df['YOJ'].plot.hist(bins=50)
# Efetuando a imputação pela Mediana: 
df['YOJ'].fillna(7, inplace=True)
# Verificando o dataset:
df.info()
# Verificando as classes e distribuição da 'DEROG':
df['DEROG'].value_counts()
# Verificando a quantidade de registros com dados missing na variável 'DEROG': 
df[df['DEROG'].isnull()]
# Verificando as classes e distribuição da 'DELINQ':
df['DELINQ'].value_counts()
# Verificando a quantidade de registros com dados missing na variável 'DELINQ': 
df[df['DELINQ'].isnull()]
# Verificando as classes e frequência da variável 'CLAGE':
df['CLAGE'].value_counts()
df[df['CLAGE'].isnull()]
# Verificando o resumo de medidas estísiticas da variável 'CLAGE':
df['CLAGE'].describe()
# Verificando a Mediana:
df['CLAGE'].median()
# Verificando a distribuição dos dados por meio do Histograma:
df['CLAGE'].plot.hist(bins=50)
# Verificando as classes e distribuição entre estas para a variável 'NINQ': 
df['NINQ'].value_counts()
# Verificando os dados nulos para a variável 'NINQ':
df[df['NINQ'].isnull()]
# Sumário estatístico para 'NINQ':
df['NINQ'].describe()
# Verificando a Mediana:
df['NINQ'].median()
# Verificando o Histograma:
df['CLAGE'].plot.hist(bins=50)
# Verificando classes e frequência para variável 'CLNO':
df['CLNO'].value_counts()
# Verificando os valores Nulos: 
df[df['CLNO'].isnull()]
# Verificando as medidas estatísticas:
df['CLNO'].describe()
# Verificando a Mediana:
df['CLNO'].median()
# Verificando o Histograma:
df['CLNO'].plot.hist(bins=50)
# Verificando as classes e distribuição da variável 'DEBTINC':
df['DEBTINC'].value_counts()
# Verificando os valores Nulos: 
df[df['DEBTINC'].isnull()]
# Verificando o resumo estatístico:
df['DEBTINC'].describe()
# Verificando a Mediana: 
df['DEBTINC'].median()
# Verificando o histograma: 
df['DEBTINC'].plot.hist(bins=50)
# Imputação dos dados nulos:
df['DEROG'].fillna(0, inplace=True)
df['DELINQ'].fillna(0, inplace=True)
df['CLAGE'].fillna(173.48, inplace=True)
df['NINQ'].fillna(1, inplace=True)
df['CLNO'].fillna(20, inplace=True)
df['DEBTINC'].fillna(33.79, inplace=True)
# Verificando o dataset após todo o tratamento de Dados 'Missing's':
df.info()
# Importando 
import seaborn as sns

# Verificando inicialmente a Tabela Cruzada

y = df['BAD'].astype(object) 
count = pd.crosstab(index = y, columns="count")
percentage = pd.crosstab(index = y, columns="frequency")/pd.crosstab(index = y, columns="frequency").sum()
pd.concat([count, percentage], axis=1)
# Plotando o gráfico da Frequência da Variável Resposta 'BAD'
ax = sns.countplot(x=y, data=df).set_title("Distribuição da Variável Resposta 'BAD'")
# Plotando o Gráfico de Barras Empilhadas mostrando a relação entre a variável 'BAD' e "JOB":
JOB=pd.crosstab(df['JOB'],df['BAD'])
JOB.div(JOB.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='JOB x BAD', figsize=(4,4))
REASON=pd.crosstab(df['REASON'],df['BAD'])
REASON.div(REASON.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='REASON x BAD', figsize=(4,4))
sns.stripplot(x='BAD', y='MORTDUE', data=df, linewidth=1)
sns.stripplot(x='BAD', y='DEBTINC', data=df, linewidth=1)
# Verificando as variáveis do dataset:
df.info()
# Transformando as variáveis 'REASON' e 'JOB' em 'dummies':
df = pd.get_dummies(df, columns=['REASON', 'JOB'])
# Verificando como ficou o dataset após o processo de transformações em 'dummies':
df.info()
# Visualizando o dataset após o processo de 'dummies':
df.head().T
df.corr()
# Importando o pacote matplotlib
import matplotlib.pyplot as plt
#Create Correlation matrix
corr = df.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10,8))
#Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()
# Importando o pacote do Sklearn:
from sklearn.model_selection import train_test_split
# Separando os dados de Treino, Validação e Teste, usando a proporção 80/20:
train, test = train_test_split(df, test_size=0.20, random_state=42)
train, valid = train_test_split(train, test_size=0.20, random_state=42)
train.shape, valid.shape, test.shape
# Definindo colunas de entrada. Excluiremos a Variável Resposta e as duas com colinearidade:
feats = [c for c in df.columns if c not in ['BAD', 'VALUE', 'REASON_HomeImp']]
feats
# Importando o pacote RandomForestClassifier necessário para rodar o modelo:

from sklearn.ensemble import RandomForestClassifier
# Instanciando o modelo com 200 árvores de decisão
rf = RandomForestClassifier(n_estimators=200, random_state=42) 
# Treinando o Modelo: 
rf.fit(train[feats], train['BAD'])
# Fazendo as previsões para os dados de Validação:
preds_val= rf.predict(valid[feats])
preds_val
# Importando o pacote necessário para verificarmos a acurácio do modelo

from sklearn.metrics import accuracy_score
# Verificando a predição nos dados de Validação:
accuracy_score(valid['BAD'], preds_val)
# Verificando a acurácia do modelo nos dados de teste:
preds_test = rf.predict(test[feats])
accuracy_score(test['BAD'], preds_test)
# avaliando a importância de cada coluna (cada variável de entrada)

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
#importando a biblioteca necessária para plotar o gráfico de Matriz de Confusão

import scikitplot as skplt
# Gerando a Matriz de Confusão 
skplt.metrics.plot_confusion_matrix(valid['BAD'], preds_val)
# Fazendo uma cópia do Dataset para a aplicação do outro modelo. Chamaremos de df1: 
df1 = df.copy()
# Separando o dataframe em dados de treino e Teste. Não será apartado dados para validação visto que usaremos
# a validação cruzada onde faremos várias validações com dados aleatórios do dataset de treino.

# Importando o train_test_split
from sklearn.model_selection import train_test_split

# Separando treino e teste
train, test = train_test_split(df1, test_size=0.20, random_state=42)

# Não vamos mais usar o dataset de validação

train.shape, test.shape
# definindo colunas de entrada

feats = [c for c in df1.columns if c not in ['BAD', 'VALUE', 'REASON_HomeImp']]
# Importar o modelo
from xgboost import XGBClassifier

# Instanciar o modelo
xgb = XGBClassifier(n_estimators=200, n_jobs=-1, random_state=42, learning_rate=0.05)
# Usando o Cross validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(xgb, train[feats], train['BAD'], n_jobs=-1, cv=5) # estimator= xgb

# Definiremos 5 splits para realizar a validação cruzada:
scores, scores.mean() 
# Usando o XGB para treinamento e predição 

xgb.fit(train[feats], train['BAD'])
# Fazendo predições
preds = xgb.predict(test[feats])
# Medir o desempenho do modelo
from sklearn.metrics import accuracy_score

accuracy_score(test['BAD'], preds)
pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()