%matplotlib inline

import numpy as np

import pandas as pd

import re as re



from matplotlib import pyplot as plt



# blibliotecas para plotarmos os dados

import seaborn as sns

sns.set_style("whitegrid")
# leitura dos datasets do titanic

# train = leitura do conjunto de dados de treinamento

# test = leitura do conjunto de dados de teste

#genderSubmissionsubmission = vamos utilizar no final da análise

train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

submission = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})



# full_data é uma estrutura que armazena os nossos 2 datasets, isso irá facilitar nosso trabalho mais para frente

full_data = [train, test]
#vamos dar uma olhada nos dados de treinamento

train.head()
#vamos dar uma olhada nos dados de teste

test.head()
# vamos verificar os dados (colunas do nosso dataset)

print (train.info())
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# utilizando o conjunto de Treinamento vamos observar os dados

p = sns.countplot(data=train, x = 'Survived', hue = 'Sex')

plt.title("Distribuição de sobreviventes de acordo com o sexo")

plt.show()



# variáveis para exibir

total_survived_females = train[train.Sex == "female"]["Survived"].sum()

total_survived_males = train[train.Sex == "male"]["Survived"].sum()



print("Total de sobreviventes: " + str((total_survived_females + total_survived_males)))

print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
# Mas e o tamanho da família, seria importante?

# podemos criar uma nova variável chamada FamilySize de acordo com 

# número de irmãos por conjuge e numero de pais por filhos abordo



# criamos uma nova variável para cada dataset em full_data chamada FamilySize

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



# informações obtidas do conjunto de treinamento

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
# dataset 0 de full_data = dataset de treinamento (train)

full_data[0].head()
# dataset 1 de full_data = dataset de teste (test)

full_data[1].head()
# dentro de cada dataset (train e teste) corrigimos os valores

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    

print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
# observando a média de idade no conjunto de Treinamento

media = train['Age'].mean()

desvio = train['Age'].std()

print("Média da idade:",media)

print("Desvio padrão da idade:",desvio)
# dentro de cada dataset (train e teste) corrigimos o campo Age (idade)

for dataset in full_data:

    age_avg = dataset['Age'].mean() #retorna a média da idade

    age_std = dataset['Age'].std()  # retorna o desvio padrão

    age_null_count = dataset['Age'].isnull().sum() #conta a quantidade de campos nulos

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

        

train['CategoricalAge'] = pd.cut(train['Age'], 5)

print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
# dentro de cada dataset (train e teste) transformamos nossas variáveis em valores numéricos

for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int) 

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4



# Seleção de variáveis que não iremos utilizar

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch','Fare']



train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge'], axis = 1)



test  = test.drop(drop_elements, axis = 1)



print (train.head(10))



train = train.values

test  = test.values
from sklearn.svm import SVC

from sklearn import metrics

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
classifiers = [SVC(probability=True)]

candidate_classifier = SVC()
candidate_classifier.fit(train[0::, 1::], train[0::, 0])
def acuracia(clf,X,y):

    scores = cross_val_score(clf, X, y, cv=5)

    resultados = cross_val_predict(clf, X, y, cv=5)

    print("Cross-validated Scores: ",scores)

    acuracia_media = metrics.accuracy_score(y,resultados)

    print("Acurácia média: ", acuracia_media)

    return None
# armazenamos os resutados esperados

classes = candidate_classifier.predict(train[0::,1::])
# executamos a função acuracia

acuracia(candidate_classifier,train,classes)
result = candidate_classifier.predict(test)
print (result)
submission.head()
final = pd.DataFrame({

        # dados armazenados em submission

        "PassengerId": submission["PassengerId"],

        "Pclass": submission["Pclass"],

        "Pclass": submission["Name"],

        "Sex": submission["Sex"],

        "Age": submission["Age"],

        "FamilySize": submission['SibSp'] + submission['Parch'] + 1,

        # dados armazenados em result

        "Survived": result

    })
final.to_csv("titanic.csv", index=False)

print(final.shape)
final
# utilizando o resultado que obtivemos vamos observar os dados

p = sns.countplot(data=final, x = 'Survived', hue = 'Sex')

plt.title("Distribuição de sobreviventes de acordo com o sexo")

plt.show()
print (final[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
genderSubmission = pd.DataFrame({

        # dados armazenados em submission

        "PassengerId": submission["PassengerId"],

        # dados armazenados em result

        "Survived": result

    })



genderSubmission.to_csv("gender_submission.csv", index=False)



print(genderSubmission)
