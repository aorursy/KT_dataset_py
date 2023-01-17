# importar pacotes necessários

import numpy as np

import pandas as pd
# carregar arquivo de dados de treino

train_data = pd.read_csv('../input/iris-train.csv', index_col='Id')



# mostrar alguns exemplos de registros

train_data.head()
# carregar arquivo de dados de teste

test_data = pd.read_csv('../input/iris-test.csv', index_col='Id')



# mostrar alguns exemplos de registros

test_data.head()
# definir dados de treino



X_train = train_data.drop(['Species'], axis=1) # tudo, exceto a coluna alvo

y_train = train_data['Species'] # apenas a coluna alvo



print('Forma dos dados de treino:', X_train.shape, y_train.shape)
# definir dados de teste



X_test = test_data # tudo, já que não possui a coluna alvo



print('Forma dos dados de teste:', X_test.shape)
# importar os pacotes necessários para os algoritmos de classificação

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
# B) Logistic Regression



model = LogisticRegression()

model.fit(X_train, y_train)



print(model)
# C) Decision Tree

'''

model = DecisionTreeClassifier()

model.fit(X_train, y_train)



print(model)

'''
# executar previsão usando o modelo escolhido

y_pred = model.predict(X_test)



print('Exemplos de previsões:\n', y_pred[:10])
# gerar dados de envio (submissão)



submission = pd.DataFrame({

  'Id': X_test.index,

  'Species': y_pred

})

submission.set_index('Id', inplace=True)



# mostrar dados de exemplo

submission.head(10)
# gerar arquivo CSV para o envio

submission.to_csv('iris-submission.csv')
# verificar conteúdo do arquivo gerado

#!head iris-submission.csv