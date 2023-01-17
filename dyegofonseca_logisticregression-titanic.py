# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
trainDf = pd.read_csv('../input/train_data.csv')

trainDf.head()
trainDf.info()
trainDf.describe()
#Verificando se há dados faltantes

sns.heatmap(trainDf.isnull())
trainDf.groupby('Survived').count()
# Verificando os sobreviventes x mortos de acordo com o sexo

plt.figure(figsize=(15,8))

sns.countplot(x='Survived', data=trainDf, hue='Sex',palette='RdBu')

plt.legend(['Mulhere', 'Homens'])
def consolidarClasse(cols): 

    for i in range(3):

        if cols[i] == 1:

            if i == 0:

                return 1

            elif i == 1:

                return 2

            else:

                return 3



trainDf['Pclass'] = trainDf[['Pclass_1', 'Pclass_2', 'Pclass_3']].apply(consolidarClasse, 

                                                                        axis=1)

            
trainDf['Pclass'].value_counts()
# Verificando os sobreviventes x mortos de acordo com a classe

plt.figure(figsize=(15,8))

sns.countplot(x='Survived', data=trainDf, hue='Pclass')

plt.legend(['Primeira Classe', 'Segunda Classe', 'Terceira Classe'])
trainDf.head()
trainDf['Pclass'].describe()
#Gráfico de distribuição de idade

plt.figure(figsize=(15,8))

#trainDf['Age'] = trainDf['Age'].apply(lambda i: i/100)

sns.distplot(trainDf['Age'])
plt.figure(figsize=(15,8))

trainDf['Fare'].hist()
trainDf.columns
#Removendo informações não relevante do dataFrame

#trainDf.drop(['Unnamed: 0', 'PassengerId', 'Family_size'], axis=1, inplace=True)

#trainDf.drop(['Pclass_1', 'Pclass_2', 'Pclass_3'], axis=1, inplace=True)



trainDf.head()
plt.figure(figsize=(17,8))

sns.heatmap(trainDf.corr(),annot=True)
#Importanto a base de testes

testDf = pd.read_csv('../input/train_data.csv')

testDf.shape
# Chamando a funcao consolidarClasse, que foi criada acima para juntar as classes em uma unica coluna

testDf['Pclass'] = testDf[['Pclass_1', 'Pclass_2', 'Pclass_3']].apply(consolidarClasse, axis=1)

testDf.head()
#Limpando os dados com informações não importantes para o modelo

#testDf.drop(['Unnamed: 0', 'PassengerId', 'Family_size'], axis=1, inplace=True)



#Removendo os dos de Classes de passageiros para evitar multicolinearidade

#testDf.drop(['Pclass_1', 'Pclass_2', 'Pclass_3'], axis=1, inplace=True)

testDf.head()
# Balanciando o modelo (calibrando)

from sklearn.linear_model import LogisticRegression
# Determinando as variáveis de treino

Xtrain = trainDf.drop(['Survived'], axis=1)

yTrain = trainDf['Survived']



# Determinando as veriáveis de teste

Xtest = testDf.drop(['Survived'], axis=1)

yTest = testDf['Survived']



# Verificando o shape das bases de teste e treino

print('Tamanho de Xtrain: ', Xtrain.shape)

print('Tamanho de yTrain: ', yTrain.shape)

print('\nTamanho de Xtest: ', Xtest.shape)

print('Tamanho de yTest: ', yTest.shape)
# Instanciando o objeto para executar a regressão logistica

logModel = LogisticRegression()
# Treinando o modelo

logModel.fit(Xtrain, yTrain)
# Rodando o Prediction

predictions = logModel.predict(Xtest)
#Avaliando o acerto do modelo com a biblioteca classification_report

from sklearn.metrics import classification_report
# Analisando os resultados obtidos. Precisão de acerto do modelo é de 80%

print(classification_report(yTest, predictions))
# Importando Matriz de confusão

from sklearn.metrics import confusion_matrix
# Analisando a matriz de confusão

print(confusion_matrix(yTest, predictions))
#Verificando os coeficientes

logModel.coef_
#Verificando o intercepto

logModel.intercept_