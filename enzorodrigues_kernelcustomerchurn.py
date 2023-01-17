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
#Carregando os dados
df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.shape
# Informações dos dados
df.info()
# Olhando os dados
df.head().T
# Verificando na coluna TotalCharges os valores em branco
df[df['TotalCharges'].str.contains(' ')]
# Trocar de nada para zero  
df['TotalCharges'] = df['TotalCharges'].str.replace(' ', '0').astype(float)
df.info()
# Verificando a coluna gender
df['gender'].value_counts()
# Verificando a coluna InternetService
df['InternetService'].value_counts()
# Vamos fazer uma cópia do dataframe
df2 = df.copy()
# Usando as variáveis dummy para a coluna gender
pd.get_dummies(df['gender'])
# Usando as variáveis dummy para a coluna InternetService
pd.get_dummies(df['InternetService'])
# Converter todas as vaiáveis categóricas em variáveis dummy
df = pd.get_dummies(df, columns=['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',
                                'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                                'StreamingMovies','Contract','PaperlessBilling', 'PaymentMethod'])
df.shape
# Olhando os dados
df.head().T
df.info()
# Separando os dados
# Precisamos de 3 conjuntos de dados: treino, teste e validação

# Importando o train_test_split
from sklearn.model_selection import train_test_split

# Dividindo em treino e teste
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Dividindo em treino e validação
train, valid = train_test_split(train, test_size=0.2, random_state=42)

train.shape, valid.shape, test.shape
# Criar a lista de colunas para treino
# Instanciar o modelo
# Treinar o modelo
# Previsões na base de validação
# Medir a acurácia
# Previsões na base de teste
# Medir a acurácia
# Definindo as feats
feats = [c for c in df.columns if c not in ['customerID', 'Churn']]
# Importar o modelo
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42,min_samples_leaf=2)
# Treinar o modelo
rf.fit(train[feats], train['Churn'])
# Previsões na base de validação
preds_val = rf.predict(valid[feats])

preds_val
# Importar a métrica de acurácia
from sklearn.metrics import accuracy_score

accuracy_score(valid['Churn'], preds_val)
# Previsões na base de teste
preds_test = rf.predict(test[feats])

preds_test
accuracy_score(test['Churn'], preds_test)
# Verificar o desempenho de base
df['Churn'].value_counts(normalize=True)
# Verificar a distribuição da resposta nos dados de treino
train['Churn'].value_counts(normalize=True)
# Importando a biblioteca que permite plotar gráficos com base nos modelos do scikit learn
import scikitplot as skplt
# Plotando a matriz de confusão para os dados de validação
skplt.metrics.plot_confusion_matrix(valid['Churn'], preds_val)
skplt.metrics.plot_confusion_matrix(test['Churn'], preds_test)