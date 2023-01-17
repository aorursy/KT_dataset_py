# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importando pacotes

import numpy as np

import pandas as pd

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn import linear_model

from math import sqrt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from statistics import mode
# Coletando dados

treino = pd.read_csv("../input/kernel02/dataset_treino.csv")

teste = pd.read_csv("../input/kernel02/dataset_teste.csv")

submission = pd.read_csv("../input/kernel02/sample_submission.csv")

transacoeshist = pd.read_csv("../input/kernel02/transacoes_historicas.csv")
hist = transacoeshist.groupby(["card_id"])

hist= hist["purchase_amount"].size().reset_index()

hist.columns = ["card_id", "hist_transactions"]

treino = pd.merge(treino,hist, on="card_id", how="left")

teste = pd.merge(teste,hist, on="card_id", how="left")



hist = transacoeshist.groupby(["card_id"])

hist = hist["purchase_amount"].agg(['sum','mean','max','min','std']).reset_index()

hist.columns = ['card_id','sum_hist_tran','mean_hist_tran','max_hist_tran','min_hist_tran','std_hist_tran']

treino = pd.merge(treino,hist,on='card_id',how='left')

teste = pd.merge(teste,hist,on='card_id',how='left')
treino.head(2)
teste.head(2)
#Substituindo valor nulo na coluna 'first_active_month' pela data mais frequente

teste['first_active_month'].fillna(mode(treino['first_active_month']), inplace=True)
#Representando numericamente valores da variável "first_active_month"

treino['first_active_month']=pd.to_datetime(treino['first_active_month'])

teste['first_active_month']=pd.to_datetime(teste['first_active_month'])

treino["ano"] = treino["first_active_month"].dt.year

teste["ano"] = teste["first_active_month"].dt.year

treino["mes"] = treino["first_active_month"].dt.month

teste["mes"] = teste["first_active_month"].dt.month
#One Hot Encoding



#treino

dumtreino_feature_1 = pd.get_dummies(treino['feature_1'],prefix = 'f1_')

dumtreino_feature_2 = pd.get_dummies(treino['feature_2'],prefix = 'f2_')

dumtreino_feature_3 = pd.get_dummies(treino['feature_3'],prefix = 'f3_')



#teste

dumteste_feature_1 = pd.get_dummies(teste['feature_1'], prefix = 'f1_')

dumteste_feature_2 = pd.get_dummies(teste['feature_2'], prefix = 'f2_')

dumteste_feature_3 = pd.get_dummies(teste['feature_3'], prefix = 'f3_')



#concatenando dados

treino = pd.concat([treino, dumtreino_feature_1, dumtreino_feature_2, dumtreino_feature_3], axis = 1, sort = False)

teste = pd.concat([teste, dumteste_feature_1, dumteste_feature_2, dumteste_feature_3], axis = 1, sort = False)
treino.head()
teste.head()
fig = plt.figure(figsize=(15,5))

sns.distplot(treino['target'])
#Criando coluna que representa os outliers

treino['outlier'] = 0

treino.loc[treino['target'] < -30, 'outlier'] = 1

treino['outlier'].value_counts()
treino.head()
#Criando dataset com valores "não-outliers"

intreino = treino[treino['outlier'] == 0]
fig = plt.figure(figsize=(15,5))

sns.distplot(intreino['target'])
treino.columns
#Criação do modelo01 - Treinando modelo 01 com dados "não outliers"



#Separando em componentes de input e output

arrayX = intreino[['hist_transactions', 'sum_hist_tran', 'mean_hist_tran',

       'max_hist_tran', 'min_hist_tran', 'std_hist_tran', 'mes', 'ano', 

        'f1__1', 'f1__2', 'f1__3', 'f1__4', 'f1__5', 'f2__1', 'f2__2',

       'f2__3', 'f3__0', 'f3__1']].values



arrayY = intreino[['target']].values



#Divide os dados em treino e teste

X_train, X_test, Y_train, Y_test = train_test_split(arrayX, arrayY, test_size = 0.33, random_state = 5)



#Criando o modelo

modelo01 = LinearRegression()



#Treinando o modelo

modelo01.fit(X_train, Y_train)



#Fazendo previsões

Y_pred = modelo01.predict(X_test)



#Resultado

rmse = sqrt(mean_squared_error(Y_test, Y_pred)) 

print("RMSE:", rmse)
#Previsão nos dados de teste



arrayTeste = teste[['hist_transactions', 'sum_hist_tran', 'mean_hist_tran',

       'max_hist_tran', 'min_hist_tran', 'std_hist_tran', 'mes', 'ano', 

        'f1__1', 'f1__2', 'f1__3', 'f1__4', 'f1__5', 'f2__1', 'f2__2',

       'f2__3', 'f3__0', 'f3__1']].values



y_pred = modelo01.predict(arrayTeste)



submission['target'] = y_pred
submission.head()
#Criação do modelo 02 - Regressão logística para tentar prever outliers nos dados de teste



#Separando  em componentes de input e output

arrayX = treino[['hist_transactions', 'sum_hist_tran', 'mean_hist_tran',

       'max_hist_tran', 'min_hist_tran', 'std_hist_tran', 'mes', 'ano', 

        'f1__1', 'f1__2', 'f1__3', 'f1__4', 'f1__5', 'f2__1', 'f2__2',

       'f2__3', 'f3__0', 'f3__1']].values



arrayY = treino[['outlier']].values



#Divide os dados em treino e teste

X_train, X_test, Y_train, Y_test = train_test_split(arrayX, arrayY, test_size = 0.33, random_state = 5)



#Definindo os valores para o número de folds

num_folds = 10

seed = 7



#Separando os dados em folds

kfold = KFold(num_folds, True, random_state = seed)



#Criando o modelo

modelo02 = LogisticRegression()



#Treinando o modelo com dados de treino

modelo02.fit(X_train, Y_train)



#Cross Validation

resultado = cross_val_score(modelo02, arrayX, arrayY, cv = kfold)



#Print do resultado

print("Acurácia: %.3f" % (resultado.mean() * 100))
#Prevendo outliers nos dados de teste

arrayTeste = teste[['hist_transactions', 'sum_hist_tran', 'mean_hist_tran',

       'max_hist_tran', 'min_hist_tran', 'std_hist_tran', 'mes', 'ano', 

        'f1__1', 'f1__2', 'f1__3', 'f1__4', 'f1__5', 'f2__1', 'f2__2',

       'f2__3', 'f3__0', 'f3__1']].values



#Previsões

out_prev = modelo02.predict(arrayTeste)



#Gerando dataset com valores previstos e imprimindo resultado

submission['outlier'] = out_prev

submission.groupby('outlier').size()
#Substituindo target dos valores outliers por (-33,21)

submission.loc[submission['outlier'] == 1, 'target'] = -33,21
submission.head()
del(submission['outlier'])



submission.to_csv('submission.csv', header = True, index = False)