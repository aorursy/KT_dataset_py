# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True)

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn import tree





import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Os três diferentes data sets são importados e atribuídos a variáveis

file_train = '../input/train.csv'

file_valid = '../input/valid.csv'

file_test = '../input/test.csv'



data_train = pd.read_csv(file_train)

data_valid = pd.read_csv(file_valid)

data_test = pd.read_csv(file_test)



print(len(data_train), len(data_valid), len(data_test))
# Insiro no Y o valor desejado para realizar a predição

Y = data_train['sale_price']



# Guarda o ID para preencher no arquivo de submissão

ID_predict_valid = data_valid['sale_id']

ID_predict_test = data_test['sale_id']



# Retiro o sale_price do dataset que será utilizado para treinar os modelos

data_train = data_train.drop('sale_price', axis=1)



# Os dois data_sets (treino e validação) são unidos para uma melhor manipulação dos dados

data_all = data_train.append(data_valid)



# ESSA LINHA FOI ADICIONADA

data_all = data_all.append(data_test)



# Dados dropados:

#     - Ease-ment : Todos os dalos da coluna são nulos,logo é melhorar desconsiderar.

#     - Address: Contém muitos valores únicos, logo não sendo útil para análise

#     - Apartment_number: Maior parte dos valores vazios, sendo ruim de inserir dados por acabar gerando dados "viciados"

#     - sale_id : informação não é útil para treino, pois não contem nenhum valor relevante, somente para indexar linhas

data_all = data_all.drop(['address', 'ease-ment', 'sale_id', 'apartment_number'], axis=1)
print(len(data_train),len(data_valid),len(data_test))

print(len(data_all))
# Verifica como estão os dados



#Overview dos dados

# Número de linhas e colunas

print(data_all.shape, "\n")



# Tipo de dados de cada coluna

print(data_all.dtypes, "\n")



#Descobrir a quantidade de valores nulos

print(data_all.isna().sum(), "\n")

# Converte as colunas para o tipo dummies

cols=[i for i in data_all.columns if i not in ['neighborhood','building_class_category','sale_date','tax_class_at_present','building_class_at_present','building_class_at_time_of_sale', 'tax_class_at_time_of_sale']]

for col in cols:

    data_all[col]=pd.to_numeric(data_all[col], errors='coerce')



#data_all.info()
# Verifica quantas colunas estão com valores nulos

null_columns=data_all.columns[data_all.isnull().any()]

data_all[null_columns].isnull().sum()
# Preenche os dados necessários que estão como nulos

data_all['land_square_feet'].fillna(data_all['land_square_feet'].mode()[0], inplace=True)

data_all['gross_square_feet'].fillna(data_all['gross_square_feet'].mode()[0], inplace=True)



# Converte a data de venda para padrão aceitável para treino

data_all['sale_date'] = pd.to_datetime(data_all.sale_date, format='%m/%d/%y').astype(int)
# Verifica se restou alguma coluna com valor nulo

null_columns=data_all.columns[data_all.isnull().any()]

data_all[null_columns].isnull().sum()
# Converte todos os dados que podem ser categorizados utilizando dummmies

data_all = pd.concat([data_all, pd.get_dummies(data_all['building_class_at_present'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['building_class_category'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['tax_class_at_present'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['tax_class_at_time_of_sale'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['building_class_at_time_of_sale'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['borough'])], axis=1);

data_all = pd.concat([data_all, pd.get_dummies(data_all['neighborhood'])], axis=1);





# Deleta colunas antigas que originaram os "dummies"

data_all = data_all.drop(['neighborhood','borough', 'building_class_category', 'tax_class_at_present', 'building_class_at_present', 

                          'tax_class_at_time_of_sale', 'building_class_at_time_of_sale'], axis=1)
print(len(data_all))

print(len(data_train))

print(len(data_valid))

print(len(data_test))
data_all
# Prepara dados para realizar o treinamento dos modelos. Separa os dados novamente em data_sets distintos após a manipulação como o Thiago explicou

data_train = data_all[:(len(data_train))]

data_valid = data_all[(len(data_train)):(len(data_train)+len(data_valid))]

data_test = data_all[(len(data_train)+len(data_valid)):]



#print((len(data_train)+len(data_valid)))



data_test = data_all[(len(data_train)+len(data_valid)):]



X = data_train

print(len(data_train), len(data_valid), len(data_test))

# Modelo Random Forest

rforest_model = RandomForestRegressor(n_estimators=50, bootstrap=False, n_jobs=-1)

rforest_model.fit(X, Y)

print("RandomForestRegressor Score")

print(rforest_model.score(X,Y))





# Modelo Decision Tree

decision_tree = tree.DecisionTreeRegressor()

decision_tree.fit(X, Y)

print("Arvore de Decisão Score")

print(decision_tree.score(X,Y))
data_to_predict = data_test.append(data_valid)

print(len(data_to_predict))



ID_predict = ID_predict_test.append(ID_predict_valid)

print(len(ID_predict))


FOREST_Y_predict = rforest_model.predict(data_to_predict)

print("RandomForestRegressor Prediction")

print(FOREST_Y_predict)



# Melhor Score

DECISION_TREE_Y_predict = decision_tree.predict(data_to_predict)

print("Decision Tree Prediction")

print(DECISION_TREE_Y_predict)
# Gera o arquivo de saída da árvore de decisão

data_to_submit = pd.DataFrame({

    'sale_id':ID_predict,

    'sale_price':DECISION_TREE_Y_predict

})

data_to_submit.to_csv('csv_to_submit_decision_tree.csv', index = False)





# Gera o arquivo de saída

data_to_submit2 = pd.DataFrame({

    'sale_id':ID_predict,

    'sale_price':FOREST_Y_predict

})

data_to_submit2.to_csv('csv_to_submit_random_forest.csv', index = False)