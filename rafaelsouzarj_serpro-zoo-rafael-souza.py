# Segundo Desafio de Modelagem Preditiva da AISMP

# rafael.souza@serpro.gov.br - (21)7499

# "O objetivo deste conjunto de dados é prever a classe dos animais com base nas características deles."



# Neste exercício, utilizei a abordagem de árvore de decisão, 

# pois creio que o conjunto de características de cada animal 

# é fator crucial para determinar a espécie de cada um deles.
# importando as bibliotecas

import numpy as np

import pandas as pd

import os

from sklearn.tree import DecisionTreeClassifier



# carregando arquivos de dados

train_data_1 = pd.read_csv('../input/zoo-train.csv', index_col='animal_name')

train_data_2 = pd.read_csv('../input/zoo-train2.csv', index_col='animal_name')

test_data = pd.read_csv('../input/zoo-test.csv', index_col='animal_name')
# substituindo os caracteres por números, para permitir a análise do modelo.



train_data_1 = train_data_1.replace(to_replace = ["y", "n"], value = ["1", "0"])

train_data_1 = train_data_1.replace(to_replace = ["y", "n"], value = ["1", "0"])

train_data_2 = train_data_2.replace(to_replace = ["y", "n"], value = ["1", "0"])

train_data_2 = train_data_2.replace(to_replace = ["y", "n"], value = ["1", "0"])

test_data = test_data.replace(to_replace = ["y", "n"], value = ["1", "0"])

test_data = test_data.replace(to_replace = ["y", "n"], value = ["1", "0"])



# Neste exercício, utilizaremos os dois arquivos de treino concatenados, para aumentar a acurácia da análise preditiva.



#train_data = train_data_1 

train_data = pd.concat([train_data_1, train_data_2])
# Exibindo o arquivo de treino concatenado



print ("train_data")

train_data.head()
# Exibindo o arquivo de teste



print ("test_data")

test_data.head()
# definindo os dados de treino de teste



X_train = train_data.drop(['class_type'], axis=1) # dados de treino: todas as colunas, menos a alvo

Y_train = pd.DataFrame(train_data['class_type']) # dados de treino: coluna alvo

X_test = test_data # dados de teste: todas as colunas, pois não possui a coluna alvo



# exibindo a forma dos dados



print('Forma dos dados de treino:', X_train.shape, Y_train.shape)

print('Forma dos dados de teste:', X_test.shape)

print(type(X_train), type(Y_train), type(X_test))
# Selecionando o modelo



model = DecisionTreeClassifier()
# Treinando o modelo



model.fit(X_train, Y_train)
# Realizando a validação cruzada



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

scores = cross_val_score(model, X_train, Y_train, cv=loo)

scores
# Realizando a análise preditiva



Y_pred = model.predict(X_test)

print(Y_pred)
# Gerando o conjunto de dados de saída



submission = pd.DataFrame({'animal_name': X_test.index, 'class_type': Y_pred})

submission.set_index('animal_name', inplace=True)



# Exibindo os dados de saída



submission.head()
# gerando o arquivo CSV de saída

submission.to_csv('../zoo-submission.csv')



# verificando o conteúdo do arquivo gerado

submission = pd.read_csv('../zoo-submission.csv', index_col='animal_name')

print(submission)
# Até o próximo desafio.

# Atenciosamente.

# rafael.souza@serpro.gov.br