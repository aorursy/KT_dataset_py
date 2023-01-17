#importando bibliotecas

import pandas as pd

import numpy as np

import seaborn as sn

from sklearn.ensemble import RandomForestClassifier
#lendo e armazenando os datasets de treino e teste

dt = pd.read_csv('../input/dataset_treino.csv')

dt_teste = pd.read_csv('../input/dataset_teste.csv')

#criando um campo para as features

features = ['num_gestacoes','glicose','pressao_sanguinea','grossura_pele','insulina','bmi','indice_historico','idade']
#verificando valores nulos do dataset de treino

dt.isnull().sum()
#verificando valores nulos do dataset de teste

dt_teste.isnull().sum()
#verificando os tipos de dados do dataset de treino

dt.dtypes
#verificando os tipos de dados do dataset de teste

dt_teste.dtypes
#dividindo o dataset de treino em datasets contendo as features e o outro a classe

X = dt[features]

y = dt.classe
#descrição dos campos

dt.describe()
#analisando os dados graficamente

sn.pairplot(data = X)
#criando a random forest classifier

rf = RandomForestClassifier(n_estimators=100)
#treinando a random forest classifier

rf.fit(X,y)
#score do treino com base no dataset

rf.score(X,y)
#matriz de confusão do teste

from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y, y)

cnf_matrix
#criando o arquivo final

teste = dt_teste[features]

result = rf.predict(teste)

df_result = dt_teste.id

dataframe=pd.DataFrame(result, columns=['classe'])

final = pd.concat([df_result, dataframe], axis=1,ignore_index=False)

#final.to_csv('final_random_forest.csv',index=False)

final.head(5)
#tipo dos dados do arquivo final

final.dtypes