# Carregando as bibliotecas necessárias

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from pandas.plotting import scatter_matrix

from sklearn.ensemble import GradientBoostingRegressor

import statsmodels.formula.api as smf

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier



# Carregando os dados demográficos

df_train_original = pd.read_csv("titanic-data-6.csv")

df_test_original = pd.read_csv("test.csv")

# df_train_original, df_test_original = train_test_split(df_train_original, test_size=0.2)

print(len(df_train_original))

print(len(df_test_original))

#df_test_originalSurv = df_test_original.copy()

#df_test_original = df_test_original.drop(['Survived'], axis=1)

df_train_original.info()
df_test_original.info()
df_train_original.nunique()
df_train = df_train_original

df_test = df_test_original



# Criando um novo campo Sex_int que atribui para os valores 0 quando o campo Sex for female e 1 caso male.

df_train['Sex_Int'] = df_train.apply(lambda df_train: 1 if df_train['Sex']=='male' else 0, axis=1)

df_test['Sex_Int'] = df_test.apply(lambda df_test: 1 if df_test['Sex']=='male' else 0, axis=1)

# Criando um novo campo Embarked_Int que atribui para os valores 0 quando o campo Embarked for S (Southampton), 

# 1 caso C (Cherbourg) e 2 caso Q (Queenstown).

df_train['Embarked_Int'] = df_train.apply(lambda df_train: 0 if df_train['Embarked']=='S' else 1 if df_train['Embarked']=='C' else 2, axis=1)

df_test['Embarked_Int'] = df_test.apply(lambda df_test: 0 if df_test['Embarked']=='S' else 1 if df_test['Embarked']=='C' else 2, axis=1)

# Criando um novo campo Child que atribui para os valores 1 quando o campo Age for menor que 13 e 0 caso contrário.

df_train['Child'] = df_train.apply(lambda df_train: 1 if df_train['Age']<13 else 0, axis=1)

df_test['Child'] = df_test.apply(lambda df_test: 1 if df_test['Age']<13 else 0, axis=1)



# Limpeza de dados, vamos atribuir a média de idade para os dados faltantes no campo Age.

# Os campos Embarked e Cabin não serão corrigidos pois não tem relevância para a análise.

mean_age = df_train['Age'].mean()

df_train['Age'].fillna(mean_age, inplace=True)

mean_age = df_test['Age'].mean()

df_test['Age'].fillna(mean_age, inplace=True)

fare_age = df_test['Fare'].mean()

df_test['Fare'].fillna(fare_age, inplace=True)





df_train = df_train.drop(['Name', 'Cabin', 'Ticket', 'SibSp', 'Parch', 'Fare', 'Age', 'Embarked_Int'], axis=1)

df_test = df_test.drop(['Name', 'Cabin', 'Ticket', 'SibSp', 'Parch', 'Fare', 'Age', 'Embarked_Int'], axis=1)



df_train = pd.get_dummies(df_train)

df_test = pd.get_dummies(df_test)



df_train = df_train.drop(['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'], axis=1)

df_test = df_test.drop(['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'], axis=1)



df_train.info()

df_test.info()



y = df_train.Survived

X_train = df_train.drop(['Survived'], axis=1)

X_test = df_test



logreg = LogisticRegression()



logreg.fit(X_train, y)

yhat_Train = logreg.predict(X_train)

yhat_test = logreg.predict(X_test)



yhat_rounded = [round(x,ndigits=None) for x in yhat_test]

yhat_rounded = [int(x) for x in yhat_rounded]



yhat_gbr = yhat_rounded

print ('# # # # Esse é o yhat com o método Gradiente Descendente # # # #')

print ('# # # # Ou seja, a previsão se Esse é o yhat com o método Gradiente Descendente # # # #')



df_test_gbr = df_test.loc[:,'PassengerId':'PassengerId'].copy()

df_test_gbr['Survived'] = yhat_gbr

df_test_gbr.to_csv('Titanic_LogReg.csv', index = False)



y = df_train.Survived

X_train = df_train.drop(['Survived'], axis=1)

X_test = df_test



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, y)

yhat_Train = random_forest.predict(X_train)

yhat_test = random_forest.predict(X_test)



yhat_rounded = [round(x,ndigits=None) for x in yhat_test]

yhat_rounded = [int(x) for x in yhat_rounded]



yhat_gbr = yhat_rounded

print ('# # # # Esse é o yhat com o método Gradiente Descendente # # # #')

print ('# # # # Ou seja, a previsão se Esse é o yhat com o método Gradiente Descendente # # # #')



df_test_gbr = df_test.loc[:,'PassengerId':'PassengerId'].copy()

df_test_gbr['Survived'] = yhat_gbr

df_test_gbr.to_csv('Titanic_RFC.csv', index = False)



y = df_train.Survived

X_train = df_train.drop(['Survived'], axis=1)

X_test = df_test



xgb = XGBClassifier()



xgb.fit(X_train, y)

yhat_Train = xgb.predict(X_train)

yhat_test = xgb.predict(X_test)



yhat_rounded = [round(x,ndigits=None) for x in yhat_test]

yhat_rounded = [int(x) for x in yhat_rounded]



yhat_gbr = yhat_rounded

print ('# # # # Esse é o yhat com o método Gradiente Descendente # # # #')

print ('# # # # Ou seja, a previsão se Esse é o yhat com o método Gradiente Descendente # # # #')



df_test_gbr = df_test.loc[:,'PassengerId':'PassengerId'].copy()

df_test_gbr['Survived'] = yhat_gbr

df_test_gbr.to_csv('Titanic_XGB.csv', index = False)



y = df_train.Survived

X_train = df_train.drop(['Survived'], axis=1)

X_test = df_test



dtc = DecisionTreeClassifier()



dtc.fit(X_train, y)

yhat_Train = dtc.predict(X_train)

yhat_test = dtc.predict(X_test)



yhat_rounded = [round(x,ndigits=None) for x in yhat_test]

yhat_rounded = [int(x) for x in yhat_rounded]



yhat_gbr = yhat_rounded

print ('# # # # Esse é o yhat com o método Gradiente Descendente # # # #')

print ('# # # # Ou seja, a previsão se Esse é o yhat com o método Gradiente Descendente # # # #')



df_test_gbr = df_test.loc[:,'PassengerId':'PassengerId'].copy()

df_test_gbr['Survived'] = yhat_gbr

df_test_gbr.to_csv('Titanic_DTC.csv', index = False)



y = df_train.Survived

X_train = df_train.drop(['Survived'], axis=1)

X_test = df_test



gnb = GaussianNB()



gnb.fit(X_train, y)

yhat_Train = gnb.predict(X_train)

yhat_test = gnb.predict(X_test)



yhat_rounded = [round(x,ndigits=None) for x in yhat_test]

yhat_rounded = [int(x) for x in yhat_rounded]



yhat_gbr = yhat_rounded

print ('# # # # Esse é o yhat com o método Gradiente Descendente # # # #')

print ('# # # # Ou seja, a previsão se Esse é o yhat com o método Gradiente Descendente # # # #')



df_test_gbr = df_test.loc[:,'PassengerId':'PassengerId'].copy()

df_test_gbr['Survived'] = yhat_gbr

df_test_gbr.to_csv('Titanic_GNB.csv', index = False)


