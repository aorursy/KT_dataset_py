# Dados das Colunas



# MAU1 =  1 cliente inadimplente no empréstimo 0 = empréstimo reembolsado
# EMPRÉSTIMOMontante do pedido de empréstimo
# MORTDUEValor devido da hipoteca existente
# VALORValor da propriedade atual
# RAZÃODebtCon = consolidação da dívida HomeImp = melhoria da casa
# TRABALHOSeis categorias ocupacionais
# YOJAnos no emprego atual
# DEROGNúmero de principais relatórios depreciativos
# DELINQNúmero de linhas de crédito inadimplentes
# CLAGEIdade da linha comercial mais antiga em meses
# NINQNúmero de linhas de crédito recentes
# CLNONúmero de linhas de crédito
# DEBTINCRácio dívida / rendimento


# Renomeação de dados #


# SITUACAO -  1 = cliente inadimplente no empréstimo 0 = empréstimo reembolsado
# VALOR_EMPRESTIMO - Montante do pedido de empréstimo
# VALOR_HIPT - Valor devido da hipoteca existente
# VALOR_PROP_ATUAL - valor da propriedade atual
# MOTIVO - consolidação da dívida HomeImp = melhoria da casa
# TRABALHO_PROF - Seis categorias profissionais (Manager, Office, Other, Prof.Executive, Sales, Self)
# TEMP_TRA - Anos no emprego atual
# N_REL_DREP - Número de principais relatórios depreciativos
# QTD_LINHAS_CREDITO_INAD - número de linhas de crédito inadimplentes
# IDADE_MAIS_ANTIGA_LN - Idade da linha comercial mais antiga em meses
# QTD_LINHAS_CREDITO_RECENT - Número de linhas de crédito recentes
# QTD_LINHAS_CREDITO - Número de linhas de crédito
# RECEITAS_DIVIDA_TX_MES - Taxa de receita da dívida (é a porcentagem da renda bruta mensal de um consumidor que é destinada ao pagamento de dívidas)


# importação das bibliotecas

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

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
# Importação o arquivo e definição do nome

df = pd.read_csv("/kaggle/input/hmeq-data/hmeq.csv")

df.head(100)

# Renomeando as colunas

df.columns = ['SITUACAO', 'VALOR_EMPRESTIMO', 'VALOR_HIPT', 'VALOR_PROP_ATUAL', 'MOTIVO', 'TRABALHO_PROF', 
              'TEMP_TRA', 'N_REL_DREP', 'QTD_LINHAS_CREDITO_INAD', 'IDADE_MAIS_ANTIGA_LN', 'QTD_LINHAS_CREDITO_RECENT', 
              'QTD_LINHAS_CREDITO', 'RECEITAS_DIVIDA_TX_MES']        


df.head()

# imputação dos valores faltantes #

df.fillna(df.mean(), inplace=True)
df.head(100)
# Quantitativo da variável SITUACAO

qtd = df['SITUACAO'].value_counts()

qtd
#DEMONSTRAÇÃO GRÁFICA DA SITUACAO

%matplotlib inline
import seaborn as sns

df['SITUACAO'].value_counts().plot.bar()
# Distribuição comparativa entre os domínios das variáveis SITUACAO, MOTIVO e TRABALHO_PROF


def showBalance(df, col):
    for c in col:
        print('Distribuição da Coluna: ', c,'\n',df[c].value_counts(normalize=True),'\n')
    else:
       pass
        
showBalance(df, col=['MOTIVO','TRABALHO_PROF','SITUACAO'])
  
      
 
# Quantitativo por Tipo de Profissão 

df['TRABALHO_PROF'].value_counts().plot.bar()
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
fig = df.VALOR_EMPRESTIMO.hist(bins=30)
fig.set_title('Distribuição de Empréstimos (VALOR_EMPRESTIMO)')
fig.set_ylabel('Quantidade de Observações com os Empréstimos no eixo X')
# Distribuição entre situacao e profissao

qtd1 = df.groupby(['SITUACAO'])['TRABALHO_PROF'].value_counts()

qtd1
# Avaliando as variáveis categóricas em relacao ao pefil do pagador - GRÁFICO

TRABALHO_PROF=pd.crosstab(df['TRABALHO_PROF'],df['SITUACAO'])
TRABALHO_PROF.div(TRABALHO_PROF.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='Tipos de Empregos e Clientes', figsize=(8,8))
MOTIVO=pd.crosstab(df['MOTIVO'],df['SITUACAO'])
MOTIVO.div(MOTIVO.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title='Situações e Razões', figsize=(5,5))
# colunas que são do tipo object

df.select_dtypes('object').head()
# Eliminando os NA #

df2 = df.copy()
df2.dropna(axis=0,how='any',inplace= True)
df2.info(), df2.isna().any() 
df2.shape
# Tabela sem (NAN) 

df2.head(100)
# Analisando a quantidade dos dados da coluna MOTIVO - Object

df2['MOTIVO'].value_counts()
# Analisando a  quantiade dos dados da coluna TRABALHO_PROF - Object

df2['TRABALHO_PROF'].value_counts()

# tranformando as colunas de object em categoria com codigos #

for col in df2.columns:
    if df2[col].dtype == 'object':
        df2[col]= df2[col].astype('category').cat.codes

# Verificando a tranformação para dammy dos dados da coluna MOTIVO

df2['MOTIVO'].value_counts()

# Verificando a tranformação para dammy  dos dados da coluna TRABALHO_PROF 

df2['TRABALHO_PROF'].value_counts()
# importando a biblioteca

from sklearn.model_selection import train_test_split
#Etapa 1- Primeiro Separando em Treino e Teste

treino, teste = train_test_split(df2, random_state=42)

treino.shape, teste.shape

# Verificando os valores da variavel SITUACAO - TREINO

treino['SITUACAO'].value_counts(normalize=True)
#Verificando os valores da variavel SITUACAO - TESTE

teste['SITUACAO'].value_counts(normalize=True)
# separar as colunas para usar no modelo

usadas_treino = [c for c in treino.columns if c not in ['SITUACAO']]

# Modelo RandomForest - Imputação / predição / acuracia  

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(treino[usadas_treino],treino['SITUACAO'])
rf_pred = rf.predict(teste[usadas_treino])
accuracy_score(teste['SITUACAO'], rf_pred)


# verificação do resultado de predição

teste['SITUACAO'] = rf.predict(teste[usadas_treino])

teste['SITUACAO'].value_counts(normalize=True)


# Após a aplicação das bases de treino e teste para predição do modelo RandomForest, pode-se observar que o mesmo obteve
# um nível de acuracia de 91%.

# O que pode ser demsotrado por meio dos valores da base de teste e predição da variavel SITUAÇÃO:
    
# BASE DE TESTE:
    
# 0    0.787572
# 1    0.212428


# PREDIÇÃO: 
    
# 0    0.833815
# 1    0.166185    



# Modelo GBM - Imputação / predição / acuracia 

from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=200, random_state=42)
gbm.fit(treino[usadas_treino], treino['SITUACAO'])
gbm_pred = gbm.predict(teste[usadas_treino])
accuracy_score(teste['SITUACAO'], gbm_pred)
# verificação do resultado de predição

teste['SITUACAO'] = gbm.predict(teste[usadas_treino])

teste['SITUACAO'].value_counts(normalize=True)


# Após a aplicação das bases de treino e teste para predição do modelo GBM, pode-se observar que o mesmo obteve
# um nível de acuracia de 96%.

# O que pode ser demostrado por meio dos valores da base de teste e predição da variavel SITUAÇÃO:
    
# BASE DE TESTE:
    
# 0    0.787572
# 1    0.212428


# PREDIÇÃO: 
    
# 0    0.825867
# 1    0.174133   



# Modelo XGBoost - Imputação / predição / acuracia 
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=200, random_state=42)
xgb.fit(treino[usadas_treino], treino['SITUACAO'])
xbm_pred = xgb.predict(teste[usadas_treino])
accuracy_score(teste['SITUACAO'], xbm_pred)


# verificação do resultado de predição

teste['SITUACAO'] = xgb.predict(teste[usadas_treino])

teste['SITUACAO'].value_counts(normalize=True)


# Após a aplicação das bases de treino e teste para predição do modelo GBM, pode-se observar que o mesmo obteve
# um nível de acuracia de 94%.

# O que pode ser demostrado por meio dos valores da base de teste e predição da variavel SITUAÇÃO:
    
# BASE DE TESTE:
    
# 0    0.787572
# 1    0.212428


# PREDIÇÃO: 
    
# 0    0.817919
# 1    0.182081  



# Verificando e avaliando a importancia de cada coluna para o modelo RF

pd.Series(rf.feature_importances_, index=usadas_treino).sort_values().plot.barh()

# Verificando e avaliando a importancia de cada coluna para o modelo GBM

pd.Series(gbm.feature_importances_, index=usadas_treino).sort_values().plot.barh()
# Verificando e avaliando a importancia de cada coluna para o modelo XGB

pd.Series(xgb.feature_importances_, index=usadas_treino).sort_values().plot.barh()

# importando a bilbioteca para plotar o gráfico de Matriz de Confusão
import scikitplot as skplt

# Matriz de Confusão - dados do modelo RandomForest 

skplt.metrics.plot_confusion_matrix(teste['SITUACAO'], rf_pred)
# Matriz de Confusão - Dados do modelo GBM

skplt.metrics.plot_confusion_matrix(teste['SITUACAO'], gbm_pred)

# Matriz de Confusão - Dados do modelo GBM

skplt.metrics.plot_confusion_matrix(teste['SITUACAO'], xbm_pred)





















