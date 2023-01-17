import pandas

import re

import operator

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression



titanic = pandas.read_csv('../input/train.csv')

titanic_test = pandas.read_csv('../input/test.csv')



predictors = ["Pclass", "Sex", "Fare", "Title"]



family_id_mapping = {}



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



def get_family_id(row):

    last_name = row["Name"].split(",")[0]

    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    if family_id not in family_id_mapping:

        if len(family_id_mapping) == 0:

            current_id = 1

        else:

            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)

        family_id_mapping[family_id] = current_id

    return family_id_mapping[family_id]



#test

titanic_test['Age'] = titanic_test['Age'].fillna(titanic['Age'].median())

titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0

titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1

titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')

titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 0

titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1

titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 2

titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())

titles = titanic_test["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}

for k,v in title_mapping.items():

    titles[titles == k] = v

titanic_test["Title"] = titles

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

family_ids = titanic_test.apply(get_family_id, axis=1)

family_ids[titanic_test["FamilySize"] < 3] = -1

titanic_test["FamilyId"] = family_ids

titanic_test['NameLength'] = titanic_test["Name"].apply(lambda x: len(x))





#train

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0

titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

titanic['Embarked'] = titanic['Embarked'].fillna('S')

titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0

titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1

titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())

titles = titanic["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}

for k,v in title_mapping.items():

    titles[titles == k] = v

titanic["Title"] = titles

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

family_ids = titanic.apply(get_family_id, axis=1)

family_ids[titanic["FamilySize"] < 3] = -1

titanic["FamilyId"] = family_ids

titanic['NameLength'] = titanic["Name"].apply(lambda x: len(x))



predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



full_predictions = []

for alg, predictors in algorithms:

    alg.fit(titanic[predictors], titanic["Survived"])

    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]

    full_predictions.append(predictions)



predictions = (full_predictions[0] * 2 + full_predictions[1] * 2) / 4

predictions[predictions <= 0.5] = 0

predictions[predictions > 0.5] = 1

predictions = predictions.astype(int)

submission = pandas.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": predictions

    })



submission.to_csv("kaggle.csv", index=False)