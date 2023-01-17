#Importing the necessary modules

#Importando as bibliotecas necessarias

import pandas as pd

#Import the necessary modules for the model

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import accuracy_score
#Converting the dataset to dataframe

#Convertendo o dataset para um dataframe



#Local Files

#Arquivos Locais

#train = pd.read_csv("input/train.csv") 

#test = pd.read_csv("input/test.csv")



#Kaggle Kernel Files

#Arquivos do Kernel do Kaggle

train = pd.read_csv("../input/train.csv") 

test = pd.read_csv("../input/test.csv") 
#It shows the first 5 rows of the dataframe

#Mostra as 5 primeiras linhas do dataframe

train.head()
#Look that the 'test' dataframe doesn't have a 'Survived' column, it's what we want to predict

#Observe que o dataframe de "test" nao possui a coluna "Survived", pois isso e o que queremos prever futuramente.

test.head()
#It removes the specific columns, like 'Name', 'Ticket', 'Cabin', in this case

#If you don't want to create a new dataframe you need to set the 'inplace' parameter as 'True'

'''

Remove as colunas especificadas, no caso quando desejar remover mais de uma coluna de uma so vez,

deve-se utilizar uma lista como primeiro argumento.

Poderia alterar/remover diretamente no dataset, sem precisar atribuir novamente a um dataframe.

Deveria utilizar entao: train.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

'''



train = train.drop(["Name", "Ticket", "Cabin"], axis=1)

test = test.drop(["Name", "Ticket", "Cabin"], axis=1)
#Now it shows the new dataframe without the dropped columns

#We need to apply the changes in both 'train' and 'test' datasets

#Imprime novamente o "head", mas agora apos o .drop, nao ha mais as colunas "Name", "Ticket", "Cabin"

#Deve-se sempre executar os "mesmas" manipulacoes de dados para "train" e "test", se nao teriamos um erro

train.head()
test.head()
#Applies the one_hot_encoding for the 'Sex' and 'Embarked' features

#Aplica one_hot_encoding para a feature de "Sex" e "Embarked"

#one_hot_train = pd.get_dummies(train)

#one_hot_test = pd.get_dummies(test)

new_data_train = pd.get_dummies(train)

new_data_test = pd.get_dummies(test)



test.shape
#Now we can see that the 'Sex' and 'Embarked' are now numerical columns.

#Observamos agora as colunas que nao eram numericas, seguindo o one_hot_encoding.

new_data_train.head()
#Checks if there is a NaN value for the training data, 'train' data.

#Verifica e agrupa a quantidade de valores nulos(NaN) para "train" data

new_data_train.isnull().sum().sort_values(ascending=False).head(10)
#We will use the mean 'Age' of the dataset for the NaN values

#Atribui a media da coluna "Age" para os valores nulos(NaN)

new_data_train["Age"].fillna(new_data_train["Age"].mean(), inplace=True)

new_data_test["Age"].fillna(new_data_test["Age"].mean(), inplace=True)

test["Age"].fillna(test["Age"].mean(), inplace=True)
#Checks if there is a NaN value for the testing data, 'test' data

#Verifica e agrupa a quantidade de valores nulos(NaN) para "test" data

new_data_test.isnull().sum().sort_values(ascending=False).head(10)

test.isnull().sum().sort_values(ascending=False).head(10)
#We will use the mean 'Fare' for the NaN values

#Atribui a media da Coluna "Fare" para os valores nulos(NaN)

new_data_test["Fare"].fillna(new_data_test["Fare"].mean(), inplace=True)

test["Fare"].fillna(test["Fare"].mean(), inplace=True)
#Splitting the 'features' and 'targets' for the model, as X and y

#Separando "features" e "targets" para o modelo, X e y respectivamente

X = new_data_train.drop("Survived", axis=1)

y = new_data_train["Survived"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Model

model = RandomForestClassifier()



# use a full grid over all parameters

param_grid = {"max_depth": range(1,25),

              "criterion": ["gini", "entropy"],

              'max_leaf_nodes': [None, 2, 3, 5, 7, 9, 12, 15]

             }



# run grid search

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)

grid_search.fit(X_train, y_train)



grid_search.best_params_
best_params_model = RandomForestClassifier(n_estimators = 300,

                                           max_leaf_nodes = 12,

                                           max_depth = 12,

                                           random_state=0)

                                           

best_params_model.fit(X_train, y_train)
Yprediction = best_params_model.predict(X_test)



best_params_model.score(X_test, y_test)
accuracy_score(y_test, Yprediction)
#Submission

#We create a new dataframe for the submission

submission = pd.DataFrame()



Xsubmission = new_data_test

submission["PassengerId"] = Xsubmission["PassengerId"]



submission["Survived"] = best_params_model.predict(Xsubmission)



#We save the submission as a '.csv' file

submission.to_csv("submission.csv", index=False)