%matplotlib inline

import pandas as pd

import numpy as np

import pylab as plt



plt.rc('figure', figsize=(10, 5))

fizsize_with_subplots = (10, 10)

bin_size = 10



# df_train é o nosso dataframe com os dados de treinamento para construção de nosso modelo.

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.head(3)

df_train.describe()
y = df_train.Survived

df_train.drop('Survived', axis=1, inplace=True) #removemos a coluna com a resposta que foi armazenada em y
n_train = df_train.PassengerId.count()

n_test = df_test.PassengerId.count()



df_todos = pd.concat([df_train, df_test]) # concatena os dataframes

n_todos = df_todos.PassengerId.count()



print(n_train, n_test, n_train+n_test, n_todos)
## Tornando os nulos em Embarked igual à moda (valor mais comum)

df_todos.loc[df_todos['Embarked'].isnull(), 'Embarked'] = 'S'
### Aqui fazemos pela média a substituição dos nulos

df_todos.loc[df_todos['Age'].isnull(), 'Age'] = int(df_todos.Age.mean())

df_todos.loc[df_todos['Fare'].isnull(), 'Fare'] = int(df_todos.Fare.mean())
### Código para gerar features polinomiais

from sklearn.preprocessing import PolynomialFeatures



#gera as features polinomiais com os atributos na lista (alterar se quiser)

poly = PolynomialFeatures(2, interaction_only=True)

novas_colunas = poly.fit_transform(df_todos[['Fare', 'Age']])

novas_colunas
## Adiciona as novas colunas polinomiais no dataframe.

for i in range(novas_colunas.shape[1]):

    df_todos['p' + str(i)] = novas_colunas[:,i]

df_todos.head(3)
novas_colunas_ohe_embarked = pd.get_dummies(df_todos['Embarked']) 

df_todos = pd.concat([df_todos,novas_colunas_ohe_embarked], axis=1) # axis = 1 concatena colunas. axis = 0 concatena linhas

df_todos.head(3)
# aqui atribuimos a nova coluna ao dataframe com outro nome.

df_todos['Pclass_int'] = pd.factorize(df_todos['Pclass'])[0]

df_todos['Sex_int'] = pd.factorize(df_todos['Sex'])[0]

df_todos.head(3)
passenger_Id = df_todos['PassengerId'] # precisamos guardar para fazer a submissão para o kaggle



df_todos_final = df_todos.drop(['PassengerId', 'Pclass', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Sex'], 

                               axis=1, inplace=False)

df_todos_final.head(3)
## checando se estamos com a quantidade certa de linhas. Vai lançar uma exceção se for diferente

assert df_todos_final.Age.count()==n_todos
X_train = df_todos_final[:n_train].values

X_test = df_todos_final[n_train:].values

y_train = y.values

passenger_Id_test = passenger_Id[n_train:].values ## só nos interessa os ids dos passageiros do conjunto de teste para submissão

print(X_train.shape, y_train.shape, X_test.shape, passenger_Id_test.shape)
## Agora já podemos embaralhar os dados de treino

from sklearn.model_selection import KFold

nfolds=5

kf = KFold(n_splits=nfolds, shuffle=True)
from sklearn import tree

from sklearn.metrics import accuracy_score ## Essa é a métrica usada na competição do kaggle



y_full_test =[] ##Aqui guardamos as previsões de cada modelo (classificador) em todo o dado de teste

metricas = []

for train, valid in kf.split(X_train, y_train):

    ## Separamos os dados dos folds

    x_train_fold = X_train[train]

    y_train_fold = y_train[train]

    x_valid = X_train[valid]

    y_valid = y_train[valid]

    

    ##Treinamos o classificador, avaliamos nos dados de validação e medimos o desempenho

    clf = tree.DecisionTreeClassifier()

    clf.fit(x_train_fold, y_train_fold)

    y_pred = clf.predict(x_valid)

    metricas.append(accuracy_score(y_valid, y_pred))

    

    ##Aqui realizamos a previsão nos dados de teste. Para cada modelo (fold) vamos gerar as previsões completas

    ##nesses dados

    y_full_test.append(clf.predict(X_test))

    

print(metricas, '\nmédia da acurácia na validação', np.array(metricas).mean())



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score #Esta eh a metrica usada na competicao do Kaggle.



y_full_test =[] ##Aqui guardamos as previsões de cada modelo (classificador) em todo o dado de teste

metricas = []

for train, valid in kf.split(X_train, y_train):

    ## Separamos os dados dos folds

    x_train_fold = X_train[train]

    y_train_fold = y_train[train]

    x_valid = X_train[valid]

    y_valid = y_train[valid]

    

    ##Treinamos o classificador, avaliamos nos dados de validação e medimos o desempenho

    #clf = tree.DecisionTreeClassifier()

    lr = LogisticRegression(solver='liblinear')

    #clf.fit(x_train_fold, y_train_fold)

    #y_pred = clf.predict(x_valid)

    metricas.append(accuracy_score(y_valid, y_pred))

    

    ##Aqui realizamos a previsão nos dados de teste. Para cada modelo (fold) vamos gerar as previsões completas

    ##nesses dados

    y_full_test.append(clf.predict(X_test))

    

print(metricas, '\nmédia da acurácia na validação', np.array(metricas).mean())





## soma as previsões de cada classificador (0 ou 1), que no final pode dar até nfold no total 

##em cada passageiro, se todos votarem 1

total = np.sum(y_full_test, axis=0)

## Agora dividimos pelo numero de folds e arredondamos.

preds_test = np.round(np.divide(total,nfolds))
df_result = pd.DataFrame(passenger_Id_test, columns=['PassengerId'])

df_result['Survived'] = (preds_test.astype('int'))

df_result.head(3)
df_result.to_csv('submittion.csv', index=False) #Index=false remove uma coluna inútil numerada de 0 a n