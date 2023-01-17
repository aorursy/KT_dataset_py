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
#Carregar os dados

df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



df.shape
#Olhar os dados

df.head(3).T
#Visualização dados aleatórios

df.sample(5).T
#Verificar o tipo dos dados e quantidade 

df.info()
# Transformar o campo totalCharges para float



df['TotalCharges'] = df['TotalCharges'].str.replace(' ','0').astype(float)
#Vamos identificar os espaços em branco

#df[df['TotalCharges'].str.contains(' ')]
# Pra corrigir a coluna TotalCharges vamos trocar espaço em branco por -1 e forçar novamente a conversão. 

#df['TotalCharges'] = df['TotalCharges'].str.replace(' ', '-1').astype(float)

# df['TotalCharges'] = df['TotalCharges'].str.strip().replace('', '-1').astype(float)
# Criando variaveis dummy para a coluna gender



pd.get_dummies(df['gender']).iloc[:, 1:]
#criando variaveis dummies da PaymentMethod 

pd.get_dummies(df['PaymentMethod']).iloc[:, 1:]
#Guardar o dataframe original

df2 = df.copy()
# Criando dummy para todas as colunas



df = pd.get_dummies(df, columns=['gender','Partner','Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',

                                'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract','PaperlessBilling',

                                'PaymentMethod'])
df.head().T
# Definindo as features 

feats = [c for c in df.columns if c not in ['customerID','Churn']]
# Separar o dataframe em treino, validação e teste



#Importando o train_test_split

from sklearn.model_selection import train_test_split



#Primeiro treino e teste

train, test = train_test_split(df, test_size=0.20, random_state=42)



#Treino e validação



train, valid = train_test_split(train, test_size=0.20, random_state=42)



train.shape, valid.shape, test.shape

# Treinando o modelo

#Importando o RandomForest 



from sklearn.ensemble import RandomForestClassifier



#Importando o modelo

rf = RandomForestClassifier(n_estimators=200, random_state=42)
# Treinando o modelo



rf.fit(train[feats], train['Churn'])
#Analisando o desempenho do modelo



#Importando metricas



from sklearn.metrics import accuracy_score
#Avaliando os dados de validacao



preds_val = rf.predict(valid[feats])



accuracy_score(valid['Churn'], preds_val)
#Avaliando os dados de teste



preds_test = rf.predict(test[feats])



accuracy_score(test['Churn'], preds_test)
# Olhar o dataFrame  completo

df['Churn'].value_counts(normalize=True)
#Continuando a aula

# Usando a nova base(copia de df)





df2.info()
#Exemplo de 

#Tipo category do pandas

#.cat para acessar as categorias solicitadas de gender





df2['gender'].astype('category').cat.categories
# acessando os mapeamentos das categorias

df2['gender'].astype('category').cat.codes
#mais um caso



df2['PaymentMethod'].astype('category').cat.categories
#convertendo as colunas categoricas para numericas



for col in df2.columns:

    if df2[col].dtypes == 'object':

        df2[col] = df2[col].astype('category').cat.codes
# Separar a base em 3 train2, valid2, test2 



train2, test2 = train_test_split(df2, test_size=0.2, random_state=42)



#treino e validação

train2, valid2 = train_test_split(df2, test_size=0.2, random_state=42)



train2.shape, valid2.shape, test2.shape
#



feats2 = [c for c in df2.columns if c not in['customerID', 'Churn']]
rf2 = RandomForestClassifier(n_estimators=200, random_state=42)



#treino

rf2.fit(train2[feats2], train2['Churn'])
#Obter as previsões da base de validação

preds2 = rf2.predict(valid2[feats2])



#Verificar a acurácia

accuracy_score(valid2['Churn'], preds2)
# obter as previsões dos dados de teste

preds_test2 = rf2.predict(test2[feats2])



#Verificar a acurácia

accuracy_score(test2['Churn'], preds_test2)
#Avaliar a importancia de cada coluna ( variavel)



import matplotlib.pyplot as plt



plt.figure(figsize=(20, 10))



#Primeiro modelo criado



pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# Segundo modelo criado



pd.Series(rf2.feature_importances_, index=feats2).sort_values().plot.barh()
# matriz de confusão



#importar biblioteca de matriz de confusão



import scikitplot as skplt 
# Dados de validação

#comparar onde acertou ou não. Falsos positivos e falsos negativos



skplt.metrics.plot_confusion_matrix(valid['Churn'], preds_val)
#Dados de teste | Matriz de confusão

skplt.metrics.plot_confusion_matrix(test['Churn'], preds_test)
#Testar o limitador de tamanho da árvore

rft = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=9)

rft.fit(train[feats], train['Churn'])

pred_teste = rft.predict(valid[feats])

accuracy_score(valid['Churn'], pred_teste)
#Testando aumentar o número de estimadores

rft = RandomForestClassifier(n_estimators=300, random_state=42)

rft.fit(train[feats], train['Churn'])

pred_teste = rft.predict(valid[feats])

accuracy_score(valid['Churn'], pred_teste)
#Testando limitar o número de registros num nó para  splitar

rft = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_split= 1000)

rft.fit(train[feats], train['Churn'])

pred_teste = rft.predict(valid[feats])

accuracy_score(valid['Churn'], pred_teste)

#Separando os datasets novamente, dessa vez levando em consideração o desbalanceio, ou seja, estratificando os datasets de teste e validação pela variável alvo



train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])



train, valid = train_test_split(train, test_size=0.2, random_state=42)



train.shape, valid.shape, test.shape
#Testando da maneira básica para comparação. Lembrando que a acurácia foi de 0.7888198757763976 para o de validação e 0.794889992902768 para teste

rft = RandomForestClassifier(n_estimators=200, random_state=42)

rft.fit(train[feats], train['Churn'])

pred_teste = rft.predict(valid[feats])

print(accuracy_score(valid['Churn'], pred_teste))



pred_teste_test = rft.predict(test[feats])

print(accuracy_score(test['Churn'], pred_teste_test))
#Agora com opções

rft= RandomForestClassifier(n_estimators=200, random_state=42, max_depth=9, min_samples_split= 10)

rft.fit(train[feats], train['Churn'])



pred_teste = rft.predict(valid[feats])

print(accuracy_score(valid['Churn'], pred_teste))



pred_teste_test = rft.predict(test[feats])

print(accuracy_score(test['Churn'], pred_teste_test))
df['Churn'].value_counts()

#Testando colocar pesos nas possibilidades de Churn para atacar o desbalanceio

class_weight = dict({'No':1, 'Yes':1.1})

rdf = RandomForestClassifier(bootstrap=True,

            class_weight=class_weight, 

            criterion='gini',

            max_depth=8, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=4, min_samples_split=10,

            min_weight_fraction_leaf=0.0, n_estimators=300,

            oob_score=False,

            random_state=42,

            verbose=0, warm_start=False)



rdf.fit(train[feats], train['Churn'])



pred_teste = rdf.predict(valid[feats])

print(accuracy_score(valid['Churn'], pred_teste))



pred_teste_test = rdf.predict(test[feats])

print(accuracy_score(test['Churn'], pred_teste_test))