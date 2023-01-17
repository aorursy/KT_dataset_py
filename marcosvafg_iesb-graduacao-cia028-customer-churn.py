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
# Carregando os dados

df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



df.shape
# Olhando os dados

df.head().T
# Olhando os dados aleatoriamente

df.sample(5).T
# Verificando os tipos dos dados e os tamanhos

df.info()
# Convertendo a coluna TotalCharges para float

df['TotalCharges'] = df['TotalCharges'].astype(float)
# Vamos identificar os valores em branco na coluna TotalCharges

df[df['TotalCharges'].str.contains(' ')]
# Para corrigir o TotalCharges vamos trocar o espaço em branco

# pelo valor ZERO e forçar novamente a conversão

df['TotalCharges'] = df['TotalCharges'].str.replace(' ', '0').astype(float)



# df['TotalCharges'] = df['TotalCharges'].str.strip().replace('', '0').astype(float)
df.info()
# Copiando o dataframe

df2 = df.copy()
# Criando variaveis dummy para gender

pd.get_dummies(df['gender'])
# Dummy da coluna PaymentMethod

pd.get_dummies(df['PaymentMethod'])
df.info()
# Criando dummys para todas as colunas

df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents',

                                'PhoneService', 'MultipleLines',

                                'InternetService', 'OnlineSecurity',

                                'OnlineBackup', 'DeviceProtection',

                                'TechSupport', 'StreamingTV',

                                'StreamingMovies', 'Contract',

                                'PaperlessBilling', 'PaymentMethod'])
df.shape
# olhando os dados

df.head().T
# Verificando os tipos

df.info()
# Treinamento do modelo



# Separando o dataframe



# Importando o train_test_split

from sklearn.model_selection import train_test_split



# Separando treino e teste

train, test = train_test_split(df, test_size=0.20, random_state=42)



# Separando treino e validação

train, valid = train_test_split(train, test_size=0.20, random_state=42)



train.shape, valid.shape, test.shape
# Definindo colunas de entrada

feats = [c for c in df.columns if c not in ['customerID', 'Churn']]



feats
# Treinar o modelo



# Importando o modelo

from sklearn.ensemble import RandomForestClassifier



# Instanciar o modelo

rf = RandomForestClassifier(n_estimators=200, random_state=42)
# Treinar o modelo

rf.fit(train[feats], train['Churn'])
# Prevendo os dados de validação

preds_val = rf.predict(valid[feats])



preds_val
# Avaliando o desempenho do modelo



# Importando a metrica

from sklearn.metrics import accuracy_score
# Acurácia das previsões de validação

accuracy_score(valid['Churn'], preds_val)
# Medindo a acurácia nos dados de teste

preds_test = rf.predict(test[feats])



accuracy_score(test['Churn'], preds_test)
# Olhando a coluna Churn do dataframe completo

df['Churn'].value_counts(normalize=True)
# Olhando a coluna Churn do dataframe completo

df['Churn'].value_counts()
df2.info()
# Tipo category no pandas

df2['gender'].astype('category').cat.categories
df2['gender'].astype('category').cat.codes
df2['PaymentMethod'].astype('category').cat.codes
df2['PaymentMethod'].astype('category').cat.categories
# Convertendo as colunas categórias em colunas numéricas

for col in df2.columns:

    if df2[col].dtype == 'object':

        df2[col] = df2[col].astype('category').cat.codes
df2.info()
# Separando o dataframe em train, valid e test



# Primeiro, train e test

train, test = train_test_split(df2, test_size=0.2, random_state=42)



# Depois, train e valid

train, valid = train_test_split(train, test_size=0.2, random_state=42)



train.shape, valid.shape, test.shape
# Colunas a serem usadas para treino

feats = [c for c in df2.columns if c not in ['customerID', 'Churn']]
# Instanciando o modelo

rf2 = RandomForestClassifier(n_estimators=200, random_state=42)



# Treinando o modelo

rf2.fit(train[feats], train['Churn'])
# previsões para os dados de validação

preds_val = rf2.predict(valid[feats])



# Verificando a acurácia

accuracy_score(valid['Churn'], preds_val)
# Previsões para os dados de teste

preds_test = rf2.predict(test[feats])



# Verificando a acurácia

accuracy_score(test['Churn'], preds_test)
# Avaliando a importancia de cada coluna (cada variável de entrada)

pd.Series(rf2.feature_importances_, index=feats).sort_values().plot.barh()
# importando a bilbioteca para plotar o gráfico de Matriz de Confusão

import scikitplot as skplt
# Matriz de Confusão - Dados de Validação

skplt.metrics.plot_confusion_matrix(valid['Churn'], preds_val)
preds_val