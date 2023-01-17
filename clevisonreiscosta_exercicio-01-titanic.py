# 1) Import all Library that will be used

%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification



from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import linear_model, svm, gaussian_process

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
# 1) Data treatment and cleaning

#Atribuindo ao data frame o arquivo .csv

#df_train_original = pd.read_csv('train.csv')

#df_test_original = pd.read_csv('test.csv')



df_train_original = pd.read_csv('../input/train.csv')

df_test_original = pd.read_csv('../input/test.csv')
#Removendo colunas do dataframe

df_train = df_train_original

df_test = df_test_original



#Dataframe é igual ao mesmo dataframe menos a coluna (Name)

df_train = df_train.drop(['Name'], axis=1)

df_test = df_test.drop(['Name'], axis=1)



df_train = df_train.drop(['Cabin'], axis=1)

df_test = df_test.drop(['Cabin'], axis=1)



df_train = df_train.drop(['Ticket'], axis=1)

df_test = df_test.drop(['Ticket'], axis=1)



df_train = df_train.drop(['SibSp'], axis=1)

df_test = df_test.drop(['SibSp'], axis=1)



df_train = df_train.drop(['Parch'], axis=1)

df_test = df_test.drop(['Parch'], axis=1)
print ('Shape - df_train_original: ', df_train_original.shape)

print ('Shape - df_train: ', df_train.shape)
#concatena os dados do treino e teste, apenas entres os campos "Pclass" e "Embarked"

# Ou seja, o campo "PassengerId" de ambos DFs serão deletados e o campo "Survived" do DF Treino

all_data = pd.concat((df_train.loc[:,'Sex':'Fare'],

                     df_test.loc[:,'Sex':'Fare']))
# Get_Dummies para transformar categoricos em Numéricos

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:



#Cria Matriz X_train utilizando a Matriz com todos os dados all_data: do inicio da matriz (:) até o fim  da matriz df_train.shape[0]

X_train = all_data[:df_train.shape[0]]



#Cria Matriz X_test utilizando a Matriz com todos os dados all_data: a partir do último registro matriz df_train.shape[0], ou seja, todos os registros que não estiverem em df_train

X_test = all_data[df_train.shape[0]:]



# Cria o y, ou seja, o que será previsto, apenas com o campo "Survived"

y = df_train.Survived
# 2) Aplly Gradient Boost Model



from sklearn.ensemble import GradientBoostingRegressor

import statsmodels.formula.api as smf

from sklearn.preprocessing import scale

gbr = GradientBoostingRegressor()



gbr.fit(X_train, y)
yhat_Train = gbr.predict(X_train)

#yhat_Train
yhat_test = gbr.predict(X_test)

#yhat_test
yhat_rounded = [round(x,ndigits=None) for x in yhat_test]

yhat_rounded = [int(x) for x in yhat_rounded]



yhat_gbr = yhat_rounded

print ('# # # # Esse é o yhat com o método Gradiente Descendente # # # #')

print ('# # # # Ou seja, a previsão se Esse é o yhat com o método Gradiente Descendente # # # #')

print (yhat_gbr)
# Gerando um CSV para o resultado obtido com o Gradiente Descendente:

df_test_gbr = df_test

df_test_gbr['Survived'] = yhat_gbr

df_test_gbr = df_test_gbr.drop(['Pclass'], axis=1)

df_test_gbr = df_test_gbr.drop(['Sex'], axis=1)

df_test_gbr = df_test_gbr.drop(['Age'], axis=1)

df_test_gbr = df_test_gbr.drop(['Fare'], axis=1)

df_test_gbr = df_test_gbr.drop(['Embarked'], axis=1)

df_test_gbr.to_csv('Titanic_GBR.csv', index = False)