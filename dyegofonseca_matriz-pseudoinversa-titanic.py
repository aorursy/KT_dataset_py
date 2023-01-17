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
trainDf = pd.read_csv('../input/train_data.csv', index_col=0)

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
trainDf.columns
#Removendo informações não relevante do dataFrame

trainDf.drop(['PassengerId', 'Family_size'], axis=1, inplace=True)

trainDf.drop(['Pclass_1', 'Pclass_2', 'Pclass_3'], axis=1, inplace=True)



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

testDf.drop(['Unnamed: 0','PassengerId', 'Family_size'], axis=1, inplace=True)



#Removendo os dos de Classes de passageiros para evitar multicolinearidade

testDf.drop(['Pclass_1', 'Pclass_2', 'Pclass_3'], axis=1, inplace=True)



testDf.head()
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
Xtrain.head()
yTrain.head()
# Testando o conceito de Matriz pseudo-inversa

A = np.array([[1, 0.27], [0, 0.47]])

A
# Criando o vetor resposta

y_ = np.array([0, 1])

y_
# invertendo a matriz A

Ainv = np.linalg.inv(A)
# Multiplicando A por sua inversa para obter a matriz identidade ("prova real")

np.dot(np.linalg.inv(A), A)
np.dot(Ainv,y_)
(A[0] * np.dot(Ainv,y_)).sum()
(A[1] * np.dot(Ainv,y_)).sum()
Xtrain.shape
# Aplicando a Transposta no Xtrain

Xtrain.T.shape
# Inicio do calculo da  Matriz pseudo-inversa, aplicada a base de dados

# Teste: Multiplicando a matriz transposta de Xtrain com a Xtrain

np.dot(Xtrain.T, Xtrain)
# Teste: Invertendo a matriz

np.linalg.inv(

    np.dot(Xtrain.T, Xtrain)

    )
'''

temp = np.array([[1, 2, 3],

         [1, 2, 3],

         [1, 2, 3]])

np.linalg.inv(temp)

'''
# Aplicando a fórmula da pseudoinversa

memoria = np.dot(

            np.linalg.inv(

                np.dot(Xtrain.T, Xtrain)

                ), 

        Xtrain.T)

memoria
# quando a matriz for singular, usamos np.linalg.pinv()

# matriz singular

#coef_pseudo = np.linalg.pinv(Xtrain)

#coef_pseudo
# Determinando o coeficiente

coef = np.dot(memoria, yTrain)

coef
(Xtest.iloc[0] * coef).sum()
Xtrain.head()
yTest.iloc[0]
for i in range(10):

    #round arredonda

    print(np.round((Xtest.iloc[i] * coef).sum()), yTest.iloc[i])
Xtrain.head(1)
Xtest.iloc[0]
coef
(coef * Xtest.iloc[0]).sum()
acerto = 0

erro = 0

for i in range(Xtest.shape[0]):

    if np.round((Xtest.iloc[i] * coef).sum()) == yTest.iloc[i]:

        acerto +=1

    else:

        erro +=1

        
#Printando o acerto do modelo

acerto/Xtest.shape[0]
#Printando o erro do modelo

erro/Xtest.shape[0]
d = {'Acerto': [acerto], 'Erro': [erro]}

grafDf = pd.DataFrame(data=d)

grafDf.plot.bar()