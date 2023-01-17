import numpy as np

import pandas as pd

import re

import operator

from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.feature_selection import SelectKBest, f_classif
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



target = train["Survived"].values
full = pd.concat([train, test], sort=False)
title_mapping = {

    "Mr": 1, 

    "Miss": 2,

    "Ms": 2,

    "Mlle": 2, 

    "Mrs": 3,

    "Mme": 3, 

    "Master": 4, 

    "Dr": 5, 

    "Rev": 6, 

    "Major": 7, 

    "Col": 7,

    "Capt": 7, 

    "Sir": 9,

    "Don": 9,

    "Dona": 9, 

    "Lady": 10, 

    "Countess": 10, 

    "Jonkheer": 10 

}



full["Title"] = full["Name"].apply(lambda x: re.search(' ([A-Za-z]+)\.', x).group(1))



full["TitleCat"] = full['Title'].map(title_mapping)
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1

full["FamilySize"] = pd.cut(full["FamilySize"], bins=[0, 1, 4, 20], labels=[0, 1, 2])
full["NameLength"] = full["Name"].apply(lambda x: len(x))
full["Embarked"] = pd.Categorical(full.Embarked).codes
full["Fare"] = full["Fare"].fillna(full["Fare"].mean())
full = pd.concat([full, pd.get_dummies(full['Sex'])], axis=1)
def get_type_cabine(cabine):

    cabine_search = re.search('\d+', cabine)



    if cabine_search:

        num = cabine_search.group(0)

        if np.float64(num) % 2 == 0:

            return '2'

        else:

            return '1'

    return '0'

    

full['CabinCat'] = pd.Categorical(full.Cabin.fillna('0').apply(lambda x: x[0])).codes



full["Cabin"] = full["Cabin"].fillna(" ")



full["CabinType"] = full["Cabin"].apply(get_type_cabine)
child_age = 18

def get_person(passenger):

    age, sex = passenger

    if (age < child_age):

        return 'child'

    elif (sex == 'female'):

        return 'female_adult'

    else:

        return 'male_adult'



full = pd.concat([full, pd.DataFrame(full[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])], axis=1)

full = pd.concat([full, pd.get_dummies(full['person'])], axis=1)
table_ticket = pd.DataFrame(full["Ticket"].value_counts())

table_ticket.rename(columns={'Ticket':'Ticket_Members'}, inplace=True)



# Women

table_ticket['Ticket_perishing_women'] = full.Ticket[(full.female_adult == 1.0) & (full.Survived == 0.0) &

                                                    ((full.Parch > 0) | (full.SibSp > 0))].value_counts()

table_ticket['Ticket_perishing_women'] = table_ticket['Ticket_perishing_women'].fillna(0)

table_ticket['Ticket_perishing_women'][table_ticket['Ticket_perishing_women'] > 0] = 1.0 



# Men

table_ticket['Ticket_surviving_men'] = full.Ticket[(full.male_adult == 1.0)  & (full.Survived == 1.0) & 

                                                    ((full.Parch > 0) | (full.SibSp > 0))].value_counts()

table_ticket['Ticket_surviving_men'] = table_ticket['Ticket_surviving_men'].fillna(0)

table_ticket['Ticket_surviving_men'][table_ticket['Ticket_surviving_men'] > 0] = 1.0 



table_ticket["Ticket_Id"]= pd.Categorical(table_ticket.index).codes

table_ticket["Ticket_Id"][table_ticket["Ticket_Members"] < 3 ] = -1

table_ticket["Ticket_Members"] = pd.cut(table_ticket["Ticket_Members"], bins=[0, 1, 4, 20], labels=[0, 1, 2])



# Join full and table_ticket

full = pd.merge(full, table_ticket, left_on="Ticket", right_index=True, how='left', sort=False)
# Capture surname

full['surname'] = full["Name"].apply(lambda x: x.split(',')[0].lower())



table_surname = pd.DataFrame(full["surname"].value_counts())

table_surname.rename(columns={'surname':'Surname_Members'}, inplace=True)



# Women

table_surname['Surname_perishing_women'] = full.surname[ (full.female_adult == 1.0) & (full.Survived == 0.0) & 

                                                         ((full.Parch > 0) | (full.SibSp > 0)) ].value_counts()

table_surname['Surname_perishing_women'] = table_surname['Surname_perishing_women'].fillna(0)

table_surname['Surname_perishing_women'][table_surname['Surname_perishing_women'] > 0] = 1.0 



# Men

table_surname['Surname_surviving_men'] = full.surname[ (full.male_adult == 1.0) & (full.Survived == 1.0) & 

                                                      ( (full.Parch > 0) | (full.SibSp > 0) ) ].value_counts()

table_surname['Surname_surviving_men'] = table_surname['Surname_surviving_men'].fillna(0)

table_surname['Surname_surviving_men'][table_surname['Surname_surviving_men'] > 0] = 1.0 





table_surname["Surname_Id"] = pd.Categorical(table_surname.index).codes

table_surname["Surname_Id"][table_surname["Surname_Members"] < 3 ] = -1

table_surname["Surname_Members"] = pd.cut(table_surname["Surname_Members"], bins=[0, 1, 4, 20], labels=[0, 1, 2])



# Join full and table_surname

full = pd.merge(full, table_surname, left_on="surname", right_index=True, how='left', sort=False)
classers = ['Fare','Parch','Pclass','SibSp','TitleCat', 

            'CabinCat','female','male', 'Embarked', 'FamilySize', 

            'NameLength','Ticket_Members','Ticket_Id']



etr = ExtraTreesRegressor(n_estimators=200)



X_train = full[classers][full['Age'].notnull()]

Y_train = full['Age'][full['Age'].notnull()]

X_test = full[classers][full['Age'].isnull()]



etr.fit(X_train, np.ravel(Y_train))

age_preds = etr.predict(X_test)



full['Age'][full['Age'].isnull()] = age_preds
features = ['female', 'male', 'Age', 'male_adult', 'female_adult', 'child', 'TitleCat', 'Pclass',

            'Pclass', 'Ticket_Id', 'NameLength', 'CabinType', 'CabinCat', 'SibSp', 'Parch',

            'Fare', 'Embarked', 'Surname_Members', 'Ticket_Members', 'FamilySize',

            'Ticket_perishing_women', 'Ticket_surviving_men',

            'Surname_perishing_women', 'Surname_surviving_men']



train = full[0:891].copy()

test = full[891:].copy()
rfc = RandomForestClassifier(n_estimators=3000, min_samples_split=4, class_weight={0: 0.745, 1: 0.255})
kf = StratifiedKFold(n_splits=10, random_state=42)



scores = cross_val_score(rfc, train[features], target, cv=kf)

print("Accuracy: {:.3f} (+/- {:.2f}) [{}]".format(scores.mean()*100, scores.std()*100, 'Cross Validation'))



rfc.fit(train[features], target)

score = rfc.score(train[features], target)

print("Accuracy: {:.3f}            [{}]".format(score*100, 'Simple test'))
rfc.fit(train[features], target)

predictions = rfc.predict(test[features])
submission_df = {

    "PassengerId": test["PassengerId"],

    "Survived": predictions

}

submission = pd.DataFrame(submission_df)



submission.to_csv("submission.csv", index=False)