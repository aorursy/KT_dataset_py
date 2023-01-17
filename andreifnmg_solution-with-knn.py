# Adicionando bibliotecas necessarias

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns

import matplotlib.pyplot as plt
# Lendo datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Apresentação basica de train

train.head()
# Apresentação basica de test

test.head()
#Obs: axis = 1, serve para remover a coluna inteira e não apenas uma linha

#Obs: inplace = True, serve para indicar q a alteração será no próprio banco

train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
new_train = pd.get_dummies(train)

new_test = pd.get_dummies(test)
new_train.head()
#Verificando quantos valores nulos tem em cada coluna

new_train.isnull().sum().sort_values(ascending=False).head()
new_test.isnull().sum().sort_values(ascending=False).head()
#Colocando nos campos nulos a média das idades dos passageiros

new_train['Age'].fillna(new_train['Age'].mean(),inplace=True)

new_test['Age'].fillna(new_test['Age'].mean(),inplace=True)

#Colocando nos campos nulos a média do preço das passagens

new_test['Fare'].fillna(new_test['Fare'].mean(),inplace=True)
plt.figure(figsize=(9,6))

sns.barplot(x='Sex',y='Survived',data=train)
plt.figure(figsize=(9,6))

sns.barplot(x='Pclass', y='Survived', data=train)
X = new_train.drop('Survived',axis=1)

recursos = new_train.columns.values

y = new_train['Survived']
modelo = KNeighborsClassifier()

modelo.fit(X,y)
parametros = {

            "n_neighbors": [1,5,8,10],

            "algorithm": ['auto','ball_tree','kd_tree','brute'],

            "leaf_size": [20,30,50]

}

grid_search = GridSearchCV(modelo, parametros)

grid_search.fit(X,y)

modelo = grid_search.best_estimator_

print(grid_search.best_params_, grid_search.best_score_)

modelo.fit(X,y)
print(modelo.score(X,y))

predicao = modelo.predict(new_test)
submission = pd.DataFrame()

submission['PassengerId'] = new_test['PassengerId']

submission['Survived'] = predicao

#Convertendo para CSV e sem o index pois não pede

submission.to_csv('submission.csv',index=False)
# Validando modelo usando Cross Validation

modelo_val = cross_val_score(modelo, X, y,scoring='accuracy', cv=5)

print(modelo_val.mean())