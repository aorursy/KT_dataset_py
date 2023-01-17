#Importa pacotes

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import os
#Ver arquivos do diretório

print(os.listdir("../input/"))
#Carregando arquivos de base

#Treino

df_train = pd.read_csv("../input/train_data_.csv").set_index("ids")

df_train = df_train[~df_train.default.isnull()]

df_train["default"] = df_train["default"].astype("int")

#Teste

df_test = pd.read_csv("../input/data_no_label.csv").set_index("ids")
#Verificar bases

print((df_train.shape, df_test.shape))
#Entendi que a base de treino tem 1 coluna a mais que a base de testes. 

#((47952, 26), (12014, 25))
#Encontrar qual coluna existe no treino que não existe no teste



train_col = df_train.columns.values

test_col = df_test.columns.values

print([value for value in train_col if value not in test_col])
#Verificando tipos de variáveis e variância dos dados

df_exp = pd.concat([df_train.dtypes,  ((df_train.T.apply(lambda x: x.nunique(), axis=1)/df_train.count())*100),(df_train.T.apply(lambda x: x.nunique(), axis=1))], axis=1,keys=['Type', 'Variance percent','Unique Values'])
#Identificar colunas com pouca variância

df_exp.sort_values(by=['Variance percent'], ascending=False).plot(kind='bar',y='Variance percent',color='red')

df_exp.sort_values(by=['Unique Values'], ascending=False).plot(kind='bar',y='Unique Values',color='blue')
plt.hist(df_exp['Variance percent'])

plt.ylabel('Frequency')

plt.xlabel('Variance percent')

plt.show()



plt.hist(df_exp['Unique Values'])

plt.ylabel('Frequency')

plt.xlabel('Unique Values')

plt.show()