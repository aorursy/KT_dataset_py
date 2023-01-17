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
# Aluna: Lídia Medeiros de Lucena Simões
# Turma: Pós-Graduação em Ciências de Dados - IESB - Asa Sul
# Matéria: Data Mining e Machine Learning II
EM = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')

EM.head(5)
print ('EM:' , EM.shape)
EM.info()
EM.sample(10).T
EM.isnull().sum()
EM.dropna(axis=0, how='any', inplace=True)
EM.info()
#Para melhorar a análise da minha base, vou criar uma variável para demonstrar o número de linhas de crédito recente (NINQ) menos o número de linhas de crédito inadimplentes (DELINQ)

EM['NINQ_DELINQ'] = EM['NINQ'] - EM['DELINQ']
EM.info()
#Importando as bibliotecas

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
EM.hist(figsize=(20,10))
ax = sns.countplot(y=x, data=EM).set_title('BAD')
#Separando as colunas para usar no treino

EM_feats = [c for c in EM.columns if c not in ['NINQ_DELINQ']]
EM_feats = EM[EM_feats]
plt.figure(figsize=(18,18))
c = 1
for i in EM_feats.columns:
    if c < len(EM_feats.columns):
        plt.subplot(3,3,c)
        sns.boxplot(x='NINQ_DELINQ' , y= i, data=EM)
        c+=1
    else:
        sns.boxplot(x='NINQ_DELINQ' , y= i, data=EM)
plt.tight_layout() 
#Visualizando o gráfico da variável criada - 'NINQ_DELINQ'
    
sns.set( rc = {'figure.figsize': (10, 10)})
EM_feats = ['NINQ_DELINQ']

for col in EM_feats:
    plt.figure()
    sns.countplot(x=EM[col], data=EM, palette="Set3")
    plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

%matplotlib inline
import seaborn as sns
train, test = train_test_split(EM, random_state=42)

train.shape, test.shape
#Agora identificar a base de Treino com a variável 'NINQ_DELINQ'

train['NINQ_DELINQ'].value_counts(normalize=True)
#Agora identificar a base de Teste com a variável 'NINQ_DELINQ'

test['NINQ_DELINQ'].value_counts(normalize=True)
# Instanciando o RandomForest Classifier

RF = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators=200, random_state=42)
#Treinando o Modelo e Identificando a acurácia do modelo

RF.fit(train[EM_feats], train['NINQ_DELINQ'])
accuracy_score(test['NINQ_DELINQ'], RF.predict(test[EM_feats]))
#Fazendo as previsões usando o Modelo com a variável 'NINQ_DELINQ'

test['NINQ_DELINQ'] = RF.predict(test[EM_feats]).astype(int)
#Verificando os resultados das previsões - TESTE

test['NINQ_DELINQ'].value_counts(normalize=True)
#Verificando os resultados das previsões - TREINO

train['NINQ_DELINQ'].value_counts(normalize=True)