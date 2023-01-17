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
#Importando as bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.model_selection import train_test_split  

from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import RandomForestClassifier

import scikitplot as skplt 

from sklearn.model_selection import train_test_split
#Importando a base de dados

base = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')

base.head(10)
base.info()
#Transformando REASON em Dummy
pd.get_dummies(base['REASON']).iloc[:4]
#Transformando JOB em Dummy
pd.get_dummies(base['JOB']).iloc[:5]
base = pd.get_dummies(base, columns=['REASON', 'JOB'])
base.head().T
#Confirmar a transformação da variável BAD
base.info()
#Análise estistica dos dados.
base[base['BAD']==0].drop('BAD', axis=1).describe().style.format("{:.2f}")
#Correlação entre as variáveis
base.corr()
#Analise da variavel BAD
sns.countplot(base['BAD'])
# Histograma com a distribuição das variáveis
base.hist()
plt.show()

base.iloc[:,[0]]
#Retirar os valores nulos

base.dropna(inplace=True)
#Confirmando a retirada dos valores NaN
base.isnull().sum()
#Grafico das variaveis LOAN e BAD com a quantidade de emprestimos e os clientes inadimplente por empréstimo e 
#emprestimos reembolsados, onde 0 = Cliente inadimplente por emprestimos | 1 = Empréstimos reembolsados.

plt.suptitle("Quantidade de emprestimos e tipos de emprestimos")
sns.stripplot(x='BAD', y='LOAN', data=base)
# Definindo as features 
feats = [c for c in base.columns if c not in ['LOAN', 'BAD']]
feats
#Separando o dataframe 

#Importando o train_test_split
from sklearn.model_selection import train_test_split

#Primeiro treino e teste
train, test = train_test_split(base, test_size=0.20, random_state=42)

#Treino e validação
train, valid = train_test_split(train, test_size=0.20, random_state=42)

train.shape, valid.shape, test.shape
#Importando o RandomForest
from sklearn.ensemble import RandomForestClassifier

#Importando o modelo
rf = RandomForestClassifier(n_estimators=200, random_state=42)
# Treinando o modelo

rf.fit(train[feats], train['BAD'])
#Primeiro treino e teste
train, test = train_test_split(base, test_size=0.20, random_state=42)
#Analisando o desempenho do modelo

#Importando metricas

from sklearn.metrics import accuracy_score
#Avaliando os dados de validacao

#Obter as previsões da base de validação
pred = rf.predict(valid[feats])

#Verificar acurácia
accuracy_score(valid['BAD'], pred)
#Avaliando os dados de teste

# obter as previsões dos dados de teste
pred_test = rf.predict(test[feats])

#Verificar a acurácia
accuracy_score(test['BAD'], pred_test)
# Olhar o dataFrame  completo
base['BAD'].value_counts(normalize=True)
base['BAD'].astype('category').cat.categories
# acessando os mapeamentos das categorias
base['BAD'].astype('category').cat.codes
plt.figure(figsize=(10, 9))

#Primeiro modelo criado
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# matriz de confusão
#importar biblioteca de matriz de confusão

import scikitplot as skplt 
# Dados de validação
#comparar onde acertou ou não. Falsos positivos e falsos negativos

skplt.metrics.plot_confusion_matrix(valid['BAD'], pred)
#Dados de teste | Matriz de confusão
skplt.metrics.plot_confusion_matrix(test['BAD'], pred_test)
# XGBoost

# Importar o modelo
from xgboost import XGBClassifier

# Instanciar o modelo
xgb = XGBClassifier(n_estimators=200, n_jobs=-1, random_state=42, learning_rate=0.05)
# Usando o cross validation
scores = cross_val_score(xgb, train[feats], train['BAD'], n_jobs=-1, cv=5)

scores, scores.mean()
# Usando o XGB para treinamento e predição
xgb.fit(train[feats], train['BAD'])
# Fazendo predições
preds = xgb.predict(test[feats])
# Medir o desempenho do modelo
from sklearn.metrics import accuracy_score

accuracy_score(test['BAD'], preds)