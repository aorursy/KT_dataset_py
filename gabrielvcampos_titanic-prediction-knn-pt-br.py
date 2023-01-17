# bibliotecas de análise dos dados

import numpy as np

import pandas as pd



# bibliotecas de visualização

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



# ignorar warnings

import warnings

warnings.filterwarnings('ignore')

treino = pd.read_csv('/kaggle/input/titanic/train.csv')

teste = pd.read_csv('/kaggle/input/titanic/test.csv')
treino.info()

# Descrição breve do nosso dataset

treino.describe(include = 'all')
# Passageiros que sobreviveram separados pelo sexo

sns.barplot(x='Sex', y='Survived', data=treino)

print("Porcentagem de mulheres que sobreviveram: {}".format( treino["Survived"][treino["Sex"] == 'female'].value_counts(normalize=True)[1]*100))

print("Porcentagem de homens que sobreviveram: {}".format( treino["Survived"][treino["Sex"] == 'male'].value_counts(normalize=True)[1]*100))
# gráfico de barras para o Número de SibSp vs Survival

sns.barplot(x='SibSp', y='Survived', data=treino)



print("porcentagem de SibSp = 0 que sobreviveram: {}".format(treino["Survived"][treino["SibSp"] == 0].value_counts(normalize = True)[1]*100))

print("porcentagem de SibSp = 1 que sobreviveram: {}".format(treino["Survived"][treino["SibSp"] == 1].value_counts(normalize = True)[1]*100))

print("porcentagem de SibSp = 2 que sobreviveram: {}".format(treino["Survived"][treino["SibSp"] == 2].value_counts(normalize = True)[1]*100))

print("porcentagem de SibSp = 3 que sobreviveram: {}".format(treino["Survived"][treino["SibSp"] == 3].value_counts(normalize = True)[1]*100))

print("porcentagem de SibSp = 4 que sobreviveram: {}".format(treino["Survived"][treino["SibSp"] == 4].value_counts(normalize = True)[1]*100))
sns.barplot(x="Parch", y="Survived", data=treino)
sns.barplot(x="Pclass", y="Survived", data=treino)

print("Porcentagem de pessoas da 1ª classe que sobreviveram: {}".format(treino['Survived'][treino['Pclass'] == 1].value_counts(normalize=True)[1]*100))

print("Porcentagem de pessoas da 2ª classe que sobreviveram: {}".format(treino['Survived'][treino['Pclass'] == 2].value_counts(normalize=True)[1]*100))

print("Porcentagem de pessoas da 3ª classe que sobreviveram: {}".format(treino['Survived'][treino['Pclass'] == 3].value_counts(normalize=True)[1]*100))

      
treino['Title'] = treino.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

teste['Title'] = teste.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
titulos_dict = {

    'Capt': 'Officer',

    'Col': 'Officer',

    'Major': 'Officer',

    'Jonkheer': 'Royalty',

    'Don': 'Royalty',

    'Sir': 'Royalty',

    'Dr': 'Officer',

    'Rev': 'Officer',

    'the Countess': 'Royalty',

    'Mme': 'Mrs',

    'Mile': 'Miss',

    'Ms': 'Mrs',

    'Mr': 'Mr',

    'Mrs': 'Mrs',

    'Miss': 'Miss',

    'Master': 'Master',

    'Lady': 'Royalty'

}



treino['Title'] = treino['Title'].map(titulos_dict)

teste['Title'] = teste['Title'].map(titulos_dict)

sns.barplot(x='Title', y='Survived', data=treino)
faixas_idade = [0, 5, 12, 18, 24, 60, np.inf]

rotulos = ['Bebê', 'Criança', 'Adolescente', 'Jovem Adulto', 'Adulto', 'Idoso']

treino['AgeGroup'] = pd.cut(treino["Age"], faixas_idade, labels = rotulos)

teste['AgeGroup'] = pd.cut(teste["Age"], faixas_idade, labels = rotulos)

ax = sns.barplot(x="AgeGroup", y="Survived", data=treino)



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")



plt.show()
treino['HasCabin'] = treino['Cabin'].notnull().astype('int')

teste['HasCabin'] = teste['Cabin'].notnull().astype('int')



#gráfico de barras dos passageiros que tinham ou não cabine e sobreviveram

sns.barplot(x='HasCabin', y="Survived", data=treino)



print("Porcentagem de pessoas com cabine que sobreviveram: {}".format(treino['Survived'][treino['HasCabin'] == 1].value_counts(normalize=True)[1]*100))

print("Porcentagem de pessoas sem cabine que sobreviveram: {}".format(treino['Survived'][treino['HasCabin'] == 0].value_counts(normalize=True)[1]*100))
teste.describe(include="all")
treino = treino.drop(['Cabin'], axis = 1)

teste = teste.drop(['Cabin'], axis = 1)
treino = treino.drop(['Ticket'], axis = 1)

teste = teste.drop(['Ticket'], axis = 1)
print("Pessoas que embarcaram em Southampton:")

S = treino[treino["Embarked"] == "S"].shape[0]

print(S)



print("Pessoas que embarcaram em Cherbourg:")

C = treino[treino["Embarked"] == "C"].shape[0]

print(C)



print("Pessoas que embarcaram em Queenstown:")

Q = treino[treino["Embarked"] == "Q"].shape[0]

print(Q)
treino = treino.fillna({"Embarked": "S"})

treino["Embarked"].value_counts()
embarked_map_dict = {

    "S": 1,

    "C": 2,

    "Q": 3

}



treino["Embarked"] = treino["Embarked"].map(embarked_map_dict)

teste['Embarked'] = teste['Embarked'].map(embarked_map_dict)



treino.head()
treino_agrupado = treino.groupby(['Sex','Pclass','Title']).median()

treino_agrupado = treino_agrupado.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
treino_agrupado.head()
def preencher_idade(linha):

    condicao = (

        (treino_agrupado['Sex'] == linha['Sex']) &

        (treino_agrupado['Title'] == linha['Title']) &

        (treino_agrupado['Pclass'] == linha['Pclass'])

    )



    if np.isnan(treino_agrupado[condicao]['Age'].values[0]):

        condicao = (

            (treino_agrupado['Sex'] == linha['Sex']) &

            (treino_agrupado['Pclass'] == linha['Pclass'])

        )

    

    return treino_agrupado[condicao]['Age'].values[0]





treino['Age'] = treino.apply(lambda linha: preencher_idade(linha) if np.isnan(linha['Age']) else linha['Age'], axis=1)

teste['Age'] = teste.apply(lambda linha: preencher_idade(linha) if np.isnan(linha['Age']) else linha['Age'], axis=1)
treino['AgeGroup'] = pd.cut(treino["Age"], faixas_idade, labels = rotulos)

teste['AgeGroup'] = pd.cut(teste["Age"], faixas_idade, labels = rotulos)
agegroup_map_dict = {

    'Adulto': 1,

    'Jovem Adulto': 2,

    'Adolescente': 3,

    'Bebê': 4,

    'Criança': 5,

    'Idoso': 6

}



treino['AgeGroup'] = treino['AgeGroup'].map(agegroup_map_dict)

teste['AgeGroup'] = teste['AgeGroup'].map(agegroup_map_dict)



treino.drop('Age', axis = 1, inplace=True)

teste.drop('Age', axis = 1, inplace=True)
titulos_map_dict = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Officer': 5, 'Royalty': 6}

treino['Title'] = treino['Title'].map(titulos_map_dict)

teste['Title'] = teste['Title'].map(titulos_map_dict)



treino['Title'] = treino['Title'].fillna(0)

teste['Title'] = teste['Title'].fillna(0)
treino.drop('Name', axis=1, inplace=True)

teste.drop('Name', axis=1, inplace=True)
sexo_map_dict = {'female': 0, 'male': 1}

treino['Sex'] = treino['Sex'].map(sexo_map_dict)

teste['Sex'] = teste['Sex'].map(sexo_map_dict)
treino.head()
treino['Fare'] = pd.Series.round(treino['Fare'], 4)

teste['Fare'] = pd.Series.round(teste['Fare'], 4)

        

treino['FareGroup'] = pd.qcut(treino['Fare'], 4, labels = [1, 2, 3, 4])

teste['FareGroup'] = pd.qcut(teste['Fare'], 4, labels = [1, 2, 3, 4])

teste['FareGroup'] = teste['FareGroup'].fillna(1)



treino.drop('Fare', axis = 1, inplace = True)

teste.drop('Fare', axis = 1, inplace = True)
treino.head()

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split



predictors = treino.drop(['Survived', 'PassengerId'], axis=1)

target = treino["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)
perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)
decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk]})

models.sort_values(by='Score', ascending=False)
ids = teste['PassengerId']

predictions = knn.predict(teste.drop('PassengerId', axis=1))



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)