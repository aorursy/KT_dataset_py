import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
titanic = pd.read_csv("../input/train.csv")     # read the data into "titanic"

print(titanic.info())
print(titanic.head(3))
dropVariables = ["PassengerId",  "Name",  "Ticket", "Cabin"]

for each in dropVariables:

    titanic = titanic.drop([each], axis = 1)

titanic['Family'] = titanic['SibSp'] + titanic['Parch']

titanic = titanic.drop(['SibSp'], axis = 1)

titanic = titanic.drop(['Parch'], axis = 1)

print(titanic.head(3))
# we want to find the most likely values for the missing data of Age and Embarked

# So I want to know the missing data first.

titanic_missingAge = titanic[titanic['Age'].isnull().values == True]

Age_relevant = ['Pclass', 'Sex', 'Fare', 'Embarked', 'Family']

titanic['FareBand'] = pd.qcut(titanic['Fare'], 4)

Age_relevant = ['Pclass', 'Sex', 'FareBand', 'Embarked', 'Family','Survived']

for each in Age_relevant:

    print(titanic[[each, 'Age']].groupby([each], as_index=False).mean().sort_values(by = each, ascending=True))

    print(titanic[[each, 'Age']].groupby([each], as_index=False).median().sort_values(by = each, ascending=True))

    print(titanic[[each, 'Age']].groupby([each], as_index=False).max().sort_values(by = each, ascending=True))

    print(titanic[[each, 'Age']].groupby([each], as_index=False).min().sort_values(by = each, ascending=True))
male = titanic['Sex'] == 'male'

female = titanic['Sex'] == 'female'

Pclass1 = titanic['Pclass'] == 1

Pclass2 = titanic['Pclass'] == 2

Pclass3 = titanic['Pclass'] == 3

Family1 = (titanic['Family'] == 0)|(titanic['Family'] == 1)|(titanic['Family'] == 2)

Family2 = (titanic['Family'] == 3)|(titanic['Family'] == 4)|(titanic['Family'] == 5)

Family3 = (titanic['Family'] == 6)|(titanic['Family'] == 7)|(titanic['Family'] == 10)

titanic['Age'][(male)&(Pclass1)&Family1] = titanic['Age'][(male)&(Pclass1)&Family1].fillna(34)

titanic['Age'][(male)&(Pclass1)&Family2] = titanic['Age'][(male)&(Pclass1)&Family2].fillna(32)

titanic['Age'][(male)&(Pclass1)&Family3] = titanic['Age'][(male)&(Pclass1)&Family3].fillna(30)

titanic['Age'][(Pclass2)&Family1] = titanic['Age'][(Pclass2)&Family1].fillna(30)

titanic['Age'][(Pclass2)&Family2] = titanic['Age'][(Pclass2)&Family2].fillna(28)

titanic['Age'][(Pclass2)&Family3] = titanic['Age'][(Pclass2)&Family3].fillna(26)

titanic['Age'][(Pclass3)&Family1] = titanic['Age'][(Pclass3)&Family1].fillna(28)

titanic['Age'][(Pclass3)&Family2] = titanic['Age'][(Pclass3)&Family2].fillna(26)

titanic['Age'][(Pclass3)&Family3] = titanic['Age'][(Pclass3)&Family3].fillna(24)

titanic['Age'][(female)&(Pclass1)&Family1] = titanic['Age'][(female)&(Pclass1)&Family1].fillna(31)

titanic['Age'][(female)&(Pclass1)&Family2] = titanic['Age'][(female)&(Pclass1)&Family2].fillna(29)

titanic['Age'][(female)&(Pclass1)&Family3] = titanic['Age'][(female)&(Pclass1)&Family3].fillna(27)

# actually this method is not enough rigorous and for simplicity, we fill Embarked with S because S appears most.

titanic['Embarked'] = titanic['Embarked'].fillna('S')

print(titanic.info())
# after fill the missing data, let's turn string into int

titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0        # turn string into int

titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1      # "female": 1, "male" : 0

titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0   # "S":0, "C":1, "Q":2

titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1

titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
# then the data cleaning ends. Let's use model to predict something.

predictors = ["Pclass", "Sex", "Age", "Family", "Embarked"]  # the features to predict the "survived"

alg = LinearRegression()     # the model we use

kf = KFold(3, random_state=1)  # some method:split the data up into cross-validation folds, total three parts

# Combine the first two parts, train a model, and make predictions on the third.

# Combine the first and third parts, train a model, and make predictions on the second.

# Combine the second and third parts, train a model, and make predictions on the first.

# to avoid overfitting
predictions = []

for train, test in kf.split(titanic):

    train_predictors = (titanic[predictors].iloc[train,:])

    train_target = titanic["Survived"].iloc[train]

    alg.fit(train_predictors, train_target)

    test_predictions = alg.predict(titanic[predictors].iloc[test,:])

    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0

accuracy = len(predictions[predictions == titanic['Survived']])/len(predictions)    # the accuracy of prediction

print(accuracy)
alg2 = LogisticRegression(random_state=1)

scores = cross_val_score(alg2, titanic[predictors], titanic["Survived"], cv=3)

print(scores)

print(scores.mean())
alg3 = SVC()

scores = cross_val_score(alg3, titanic[predictors], titanic["Survived"], cv=3)

print(scores)

print(scores.mean())
alg4 = KNeighborsClassifier(n_neighbors = 3)

scores = cross_val_score(alg4, titanic[predictors], titanic["Survived"], cv=3)

print(scores)

print(scores.mean())
alg5 = GaussianNB()

scores = cross_val_score(alg5, titanic[predictors], titanic["Survived"], cv=3)

print(scores)

print(scores.mean())
alg6 = Perceptron()

scores = cross_val_score(alg6, titanic[predictors], titanic["Survived"], cv=3)

print(scores)

print(scores.mean())
alg7 = LinearSVC()

scores = cross_val_score(alg7, titanic[predictors], titanic["Survived"], cv=3)

print(scores)

print(scores.mean())
alg8 = SGDClassifier()

scores = cross_val_score(alg8, titanic[predictors], titanic["Survived"], cv=3)

print(scores)

print(scores.mean())
alg9 = DecisionTreeClassifier()

scores = cross_val_score(alg9, titanic[predictors], titanic["Survived"], cv=3)

print(scores)

print(scores.mean())
alg10 = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(alg10, titanic[predictors], titanic["Survived"], cv=3)

print(scores)

print(scores.mean())
test = pd.read_csv("../input/test.csv")     # read the test data
test["Age"] = test["Age"].fillna(titanic["Age"].median())   # just for simplicity

test['Family'] = test['SibSp'] + test['Parch']

test.loc[test["Sex"] == "male", "Sex"] = 0 

test.loc[test["Sex"] == "female", "Sex"] = 1

test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0

test.loc[test["Embarked"] == "C", "Embarked"] = 1

test.loc[test["Embarked"] == "Q", "Embarked"] = 2
predictions = alg.predict(test[predictors])



# Create a new dataframe with only the columns Kaggle wants from the data set

gender_submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })
gender_submission[gender_submission['Survived'] > .5] = 1

gender_submission[gender_submission['Survived'] <=.5] = 0

pd.set_option('display.height', 2000)

print(gender_submission)