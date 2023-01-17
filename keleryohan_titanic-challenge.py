#importando bibliotecas

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor

from sklearn.model_selection import cross_val_score, StratifiedKFold

import re

import operator

from sklearn.feature_selection import SelectKBest, f_classif
#lendo o banco de dados



train = pd.read_csv("../input/titanic/train.csv", dtype={"Age": np.float64})

test = pd.read_csv("../input/titanic/test.csv", dtype={"Age": np.float64})



#isolando a coluna alvo

target = train["Survived"].values

#juntando os dois datasets para aplicarmos o pré-processamento a ambos

full = pd.concat([train, test],sort=True) 
print(full.head())
print(full.describe())
print(full.info())
#pegando o sobrenome e o título dos passageiros

full['surname'] = full["Name"].apply(lambda x: x.split(',')[0].lower())



full["Title"] = full["Name"].apply(lambda x: re.search(' ([A-Za-z]+)\.',x).group(1))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 2, "Mme": 3,"Don": 9,"Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

full["TitleCat"] = full.loc[:,'Title'].map(title_mapping)
#verificando o tamanho da família e separando em grupos

full["FamilySize"] = full["SibSp"] + full["Parch"] + 1

full["FamilySize"] = pd.cut(full["FamilySize"], bins=[0,1,4,20], labels=[0,1,2])
#Lidando com as colunas de string e valores na 



#criando a coluna de tamanho do nome para substituir a de nome

full["NameLength"] = full["Name"].apply(lambda x: len(x))



#transformando a coluna embarked em pandas categorical

full["Embarked"] = pd.Categorical(full.Embarked).codes



#convertendo a coluna Sex usando dummies

full = pd.concat([full,pd.get_dummies(full['Sex'])],axis=1)



#transformando a coluna cabincat para valores de int categóricos e preenchendo seus na com 0

full['CabinCat'] = pd.Categorical(full.Cabin.fillna('0').apply(lambda x: x[0])).codes



#preenchendo o(1) valor na de Fare com a média da coluna

full["Fare"] = full["Fare"].fillna(np.mean(full["Fare"]))





#convertendo a coluna de cabine

#função que verifica se a cabine tem digitos pares, ímpares ou na 

def get_type_cabine(cabine):

    # Use a regular expression to search for a title. 

    cabine_search = re.search('\d+', cabine)

    # If the title exists, extract and return it.

    if cabine_search:

        num = cabine_search.group(0)

        if np.float64(num) % 2 == 0:

            return '2'

        else:

            return '1'

    return '0'

#preenchendo os nas com string vazia

full["Cabin"] = full["Cabin"].fillna(" ")



full["CabinType"] = full["Cabin"].apply(get_type_cabine)
#separando passageiros em crianças(pessoas com menos de 18 anos), mulheres e homens

child_age = 18

def get_person(passenger):

    age, sex = passenger

    if (age < child_age):

        return 'child'

    elif (sex == 'female'):

        return 'female_adult'

    else:

        return 'male_adult'

full = pd.concat([full, pd.DataFrame(full[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])],axis=1)



#convertendo a coluna person usando dummies

full = pd.concat([full,pd.get_dummies(full['person'])],axis=1)
#colunas baseadas no ingresso

table_ticket = pd.DataFrame(full["Ticket"].value_counts())

table_ticket.rename(columns={'Ticket':'Ticket_Members'}, inplace=True)



table_ticket['Ticket_perishing_women'] = full.Ticket[(full.female_adult == 1.0) 

                                    & (full.Survived == 0.0) 

                                    & ((full.Parch > 0) | (full.SibSp > 0))].value_counts()

table_ticket['Ticket_perishing_women'] = table_ticket['Ticket_perishing_women'].fillna(0)

table_ticket['Ticket_perishing_women'][table_ticket['Ticket_perishing_women'] > 0] = 1.0 



table_ticket['Ticket_surviving_men'] = full.Ticket[(full.male_adult == 1.0) 

                                    & (full.Survived == 1.0) 

                                    & ((full.Parch > 0) | (full.SibSp > 0))].value_counts()

table_ticket['Ticket_surviving_men'] = table_ticket['Ticket_surviving_men'].fillna(0)

table_ticket['Ticket_surviving_men'][table_ticket['Ticket_surviving_men'] > 0] = 1.0 



table_ticket["Ticket_Id"]= pd.Categorical(table_ticket.index).codes



table_ticket["Ticket_Id"][table_ticket["Ticket_Members"] < 3 ] = -1

table_ticket["Ticket_Members"] = pd.cut(table_ticket["Ticket_Members"], bins=[0,1,4,20], labels=[0,1,2])



full = pd.merge(full, table_ticket, left_on="Ticket",right_index=True,how='left', sort=False)
#colunas baseada no sobrenome

table_surname = pd.DataFrame(full["surname"].value_counts())

table_surname.rename(columns={'surname':'Surname_Members'}, inplace=True)



table_surname['Surname_perishing_women'] = full.surname[(full.female_adult == 1.0) 

                                    & (full.Survived == 0.0) 

                                    & ((full.Parch > 0) | (full.SibSp > 0))].value_counts()

table_surname['Surname_perishing_women'] = table_surname['Surname_perishing_women'].fillna(0)

table_surname['Surname_perishing_women'][table_surname['Surname_perishing_women'] > 0] = 1.0 



table_surname['Surname_surviving_men'] = full.surname[(full.male_adult == 1.0) 

                                    & (full.Survived == 1.0) 

                                    & ((full.Parch > 0) | (full.SibSp > 0))].value_counts()

table_surname['Surname_surviving_men'] = table_surname['Surname_surviving_men'].fillna(0)

table_surname['Surname_surviving_men'][table_surname['Surname_surviving_men'] > 0] = 1.0 



table_surname["Surname_Id"]= pd.Categorical(table_surname.index).codes





table_surname["Surname_Id"][table_surname["Surname_Members"] < 3 ] = -1



table_surname["Surname_Members"] = pd.cut(table_surname["Surname_Members"], bins=[0,1,4,20], labels=[0,1,2])



full = pd.merge(full, table_surname, left_on="surname",right_index=True,how='left', sort=False)
#coluna de idade

classers = ['Fare','Parch','Pclass','SibSp','TitleCat', 'CabinCat','female','male', 'Embarked', 'FamilySize', 'NameLength','Ticket_Members','Ticket_Id']

etr = ExtraTreesRegressor(n_estimators=200)

X_train = full[classers][full['Age'].notnull()]

Y_train = full['Age'][full['Age'].notnull()]

X_test = full[classers][full['Age'].isnull()]

etr.fit(X_train,np.ravel(Y_train))

age_preds = etr.predict(X_test)

full['Age'][full['Age'].isnull()] = age_preds
#colunas disponíveis

features = ['female','male','Age','male_adult','female_adult', 'child','TitleCat', 'Pclass',

'Pclass','Ticket_Id','NameLength','CabinType','CabinCat', 'SibSp', 'Parch',

'Fare','Embarked','Surname_Members','Ticket_Members','FamilySize',

'Ticket_perishing_women','Ticket_surviving_men',

'Surname_perishing_women','Surname_surviving_men']



#separando os dados em treino e teste de novo

train = full[0:891].copy()

test = full[891:].copy()



#usando o selectorKBest para conseguir as melhores colunas para o modelo

selector = SelectKBest(f_classif, k=len(features))

selector.fit(train[features], target)

scores = -np.log10(selector.pvalues_)

indices = np.argsort(scores)[::-1]
#definindo o modelo

rfc = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0:0.745,1:0.255})
# testando (cross validation)

kf = StratifiedKFold(n_splits=10,random_state=42)



scores = cross_val_score(rfc, train[features], target, cv=kf)

print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean()*100, scores.std()*100, 'RFC Cross Validation'))

rfc.fit(train[features], target)

score = rfc.score(train[features], target)

print("Accuracy: %0.3f            [%s]" % (score*100, 'RFC full test'))

importances = rfc.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(len(features)):

    print("%d. feature %d (%f) %s" % (f + 1, indices[f]+1, importances[indices[f]]*100, features[indices[f]]))

#predicting

rfc.fit(train[features], target)

predictions = rfc.predict(test[features])
#prediction file

PassengerId =np.array(test["PassengerId"]).astype(int)

my_prediction = pd.DataFrame(predictions, PassengerId, columns = ["Survived"])



my_prediction.to_csv("my_prediction.csv", index_label = ["PassengerId"])