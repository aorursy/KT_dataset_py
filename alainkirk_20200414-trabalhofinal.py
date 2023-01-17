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
# Carga do arquivo

df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')



# Dimensões do Dataframe

df.shape
# Análise das variáveis

df.info()
# Guardando o dataframe original

df_br = df



# Trocando nome das colunas:

df_br.rename(columns={

    'BAD': 'TIPO_PAGADOR',

    'LOAN': 'VALOR',

    'MORTDUE': 'HIPOTECA',

    'VALUE': 'VALOR_PROPRIEDADE',

    'REASON': 'OBJETIVO',

    'JOB': 'TRABALHO',

    'YOJ': 'TEMPO_EMPRESA',

    'DEROG': 'QTD_REL_NEGATIVO',

    'DELINQ': 'QTD_INADIMPLENTE',

    'CLAGE': 'IDADE_LINHA_COM',

    'NINQ': 'QTD_CRED_RECENT',

    'CLNO': 'QTD_LINHA_CRED',

    'DEBTINC': 'TAXA'}, inplace=True)

# Listando novamenta as colunas

for col in df_br.columns:

    print(col)
# Verificando o tipo de dados

df_br.dtypes
# Listagem dos primeiros dados

df_br.head()
# Alteração do conteúdo dos dados TIPO_PAGADOR



# Transformando 1 em 'MAU' e 0 em 'BOM' na coluna TIPO_PAGADOR, com alteração do tipo de dado da coluna

# Passo 1: criação do dicionário

dic_tipo_pagador = {1: 'MAU', 0: 'BOM'}



# Passo 2: substituição com alteração do tipo de dado da coluna

df_br['TIPO_PAGADOR'] = df_br['TIPO_PAGADOR'].replace(dic_tipo_pagador).astype(str)
# Listagem dos primeiros dados

df_br.head()
# Verificação do tipo de dados

df_br.dtypes
# Verificando os tipos de 'OBJETIVO':

df_br.groupby('OBJETIVO')['OBJETIVO'].count()
# Alteração do conteúdo dos dados 'OBJETIVO'



# Transformando DebtCon em 'Consolidação da Dívida' e 'HomeImp' em 'Melhoria da Casa' na coluna 'OBJETIVO'

# Passo 1: criação do dicionário

dic_objetivo = {'DebtCon': 'Consolidação da Dívida', 'HomeImp': 'Melhoria da Casa'}



# Passo 2: substituição dos dados

df_br['OBJETIVO'] = df_br['OBJETIVO'].replace(dic_objetivo)



# Verificação

df_br.groupby('OBJETIVO')['OBJETIVO'].count()
# Verificando os tipos de 'TRABALHO'

df_br.groupby('TRABALHO')['TRABALHO'].count()
# Alteração do conteúdo dos dados 'TRABALHO'



# Transformando

    # 'Mgr'     em 'Gerente'

    # 'Office'  em 'Funcionário Público'

    # 'Other'   em 'Outro'

    # 'ProfExe' em 'Executivo'

    # 'Sales'   em 'Vendedor'

    # 'Self'    em 'Autônomo'



# Passo 1: criação do dicionário

dic_trabalho = {

    'Mgr'     :'Gerente',

    'Office'  :'Funcionário Público',

    'Other'   :'Outro',

    'ProfExe' :'Executivo',

    'Sales'   :'Vendedor',

    'Self'    :'Autônomo'}



# Passo 2: substituição dos dados

df_br['TRABALHO'] = df_br['TRABALHO'].replace(dic_trabalho)



# Verificação

df_br.groupby('TRABALHO')['TRABALHO'].count()
# Preenchendo com zero as colunas de valor nulas (Nan)

# df_br.fillna(0,inplace=True)
# Listagem dos primeiros dados

df_br.head()
# Análise Estatística

df_br.describe().T
# Analisando as colunas:

df['VALOR'].plot.hist(bins=50)
# Analisando as colunas:

df['HIPOTECA'].plot.hist(bins=50)
# Importando seaborn para gráficos

import seaborn as sns



# Formatar o gráfico

import matplotlib.pyplot as plt
# Plotando gráfico para TIPO_PAGADOR

plt.figure(figsize=(15,5))



# Somente os 10 primeiros

sns.countplot(data=df_br, x='TIPO_PAGADOR',order=df_br.TIPO_PAGADOR.value_counts().index)

                                                

plt.show()
# Plotando gráfico para TRABALHO

plt.figure(figsize=(15,5))



# Somente os 10 primeiros

sns.countplot(data=df_br, x='TRABALHO',order=df_br.TRABALHO.value_counts().index)

                                                

plt.show()
# Plotando gráfico para OBJETIVO

plt.figure(figsize=(15,5))



# Somente os 10 primeiros

sns.countplot(data=df_br, x='OBJETIVO',order=df_br.OBJETIVO.value_counts().index)

                                                

plt.show()
# Relacionar MAUS E BONS PAGADORES e VALOR

plt.figure(figsize=(15,5))

sns.stripplot(x='TIPO_PAGADOR',y='VALOR',data=df_br, linewidth=1)
# Relacionar TIPO_PAGADOR e TEMPO_EMPRESA

plt.figure(figsize=(15,5))

sns.stripplot(x='TIPO_PAGADOR',y='TEMPO_EMPRESA',data=df_br, linewidth=1)
# Relacionar TRABALHO e VALOR

plt.figure(figsize=(15,5))

plt.xticks(rotation=90)

sns.stripplot(x='TRABALHO',y='VALOR',data=df_br, linewidth=1)
# Boxplot entre TRABALHO e VALOR

plt.figure(figsize=(15,5))

plt.xticks(rotation=0)

sns.boxplot(df_br['TRABALHO'],df_br['VALOR'])
# Analisando o valor pelo tempo de empresa



plt.figure(figsize=(15,5))

sns.pointplot(x='TEMPO_EMPRESA', y='VALOR',data=df_br, color='green')

plt.title('Valor pelo Tempo de Empresa')

plt.grid(True, color='silver')
# Relacionar MAUS E BONS PAGADORES e VALOR

plt.figure(figsize=(15,5))

sns.stripplot(x='TIPO_PAGADOR',y='HIPOTECA',data=df_br, linewidth=1)
# Valores por tipo de trabalho

plt.figure(figsize=(15,5))

sns.barplot(y='VALOR',x='TRABALHO',data=df_br)
# Análise por Plotagem do Gráfico

sns.set(rc={'figure.figsize':(15,5)})

plt.figure()

sns.distplot(df_br['VALOR'], bins=50)

plt.xticks(rotation=0)

plt.show()
# Análise por Plotagem do Gráfico

sns.set(rc={'figure.figsize':(15,5)})

plt.figure()

sns.distplot(df_br['HIPOTECA'], bins=50)

plt.xticks(rotation=0)

plt.show()
# Análise por Plotagem do Gráfico

sns.set(rc={'figure.figsize':(15,5)})

plt.figure()

sns.distplot(df_br['TEMPO_EMPRESA'], bins=50)

plt.xticks(rotation=0)

plt.show()
# Análise do cruzamento de 'VALOR' e valor de 'HIPOTECA'

sns.scatterplot(data=df_br, x='VALOR', y='HIPOTECA')
# Cruzamento também de 'VALOR' e o valor da PROPRIEDADE

sns.scatterplot(data=df_br, x='VALOR', y='VALOR_PROPRIEDADE')
# Verificando a Correlação das variáveis



f, ax = plt.subplots(figsize=(15,6))

sns.heatmap(df_br.corr(), annot=True, fmt='.2f', linecolor='black',ax=ax, lw=.7)
# Cópia do dataframe obtendo valores dummies

df_br2 = pd.get_dummies(df_br, columns=['OBJETIVO','TRABALHO'])
# Definição da independentes (target)

feats = [coluna for coluna in df_br2.columns if coluna not in ['TIPO_PAGADOR']]
# Preenchendo com zero as colunas de valor nulas (Nan)

df_br2.fillna(0,inplace=True)
# Separação do dataframe para modelo

from sklearn.model_selection import train_test_split

train, valid = train_test_split(df_br2, test_size=0.2, random_state=42)
# Importação daS bibliotecas para o modelo

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
# Execução do modelo

rf = RandomForestClassifier(n_estimators=200, random_state=42)

rf.fit(train[feats], train['TIPO_PAGADOR'])
# Modelo aplicado na base de validação

preds_val = rf.predict(valid[feats])
# Verificação da acurácia

accuracy_score(valid['TIPO_PAGADOR'], preds_val)
# Plotagem do resultado do modelo RandomForrest

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# Matriz de Confusão

import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(valid['TIPO_PAGADOR'],preds_val)
# Importação da biblioteca

from sklearn.model_selection import cross_val_score
# Execução

scores = cross_val_score(rf, train[feats], train['TIPO_PAGADOR'], n_jobs=-1, cv=5)
# Analisando o resultado e a média

scores, scores.mean()
# Plotando o resultado

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# Importando a biblioteca k-means

from sklearn.cluster import KMeans
# Variáveis escolhidas para teste do modelo

variaveis = df_br2[['HIPOTECA','VALOR', 'TEMPO_EMPRESA']]
# Cálculo da soma dos erros quadrados

sse = []



for k in range (1, 12):

    kmeans = KMeans(n_clusters=k, random_state=42).fit(variaveis)

    sse.append(kmeans.inertia_)

print(sse)
# Método Elbow 

import matplotlib.pyplot as plt



plt.plot(range(1, 12), sse, 'bx-')

plt.xlabel('Clusters')

plt.ylabel('Soma dos Erros Quadrados')

plt.show()
# Número de clusters ideal observado é de 4, conforme observado no gráfico

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)

cluster_id = kmeans.fit_predict(variaveis)

cluster_id
# Armazenando no dataframe

variaveis['cluster_id'] = cluster_id
# Análise do dataframe

variaveis.sample(10).T
# Visualizando os centroides e agrupamentos

plt.scatter(variaveis.values[:,0], variaveis.values[:,1], c=kmeans.labels_)

plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], marker="x", s=200, color='green')

plt.show()