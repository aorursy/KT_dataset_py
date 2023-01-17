# Loading relevant packages

from pandas import read_csv, concat, DataFrame

from numpy import isnan, array, inf

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

%matplotlib inline
# Loading the train and test files (predicted values are in the 'gender_submission.csv' file)

train = read_csv("../input/train.csv")

train = train.set_index("PassengerId")



test = read_csv("../input/test.csv")

test = test.set_index("PassengerId")



y_test = read_csv('../input/gendermodel.csv').set_index("PassengerId")

test = y_test.join(test)
# Formatting train data



train_set = train.drop(['Name', 'Cabin'], 1)



# Ticket number is split, only the integer is kept in the analysis

ticket_number = []

for ticket in train_set.Ticket:

    elements = ticket.split(" ")

    if len(elements) > 1:

        ticket_number += [int(elements[len(elements) - 1])]

    else:

        if elements[0] == 'LINE': 

            ticket_number += [0]

        else:

            ticket_number += [int(elements[0])]

ticket_number = array(ticket_number)



train_set.Ticket = ticket_number.astype(int)



# Sex values are set to 0 (for "male") and 1 (for "female")

train_set.Sex.loc[train_set.Sex == "male"] = 0

train_set.Sex.loc[train_set.Sex == "female"] = 1



# Keeping only the non nan values of "Embarked" column

train_set = train_set.loc[(train_set.Embarked == "C") | (train_set.Embarked == "Q") | (train_set.Embarked == "S")]



# Giving numerical values to the different classes

train_set.Embarked.loc[train_set.Embarked == "C"] = 0

train_set.Embarked.loc[train_set.Embarked == "Q"] = 1

train_set.Embarked.loc[train_set.Embarked == "S"] = 2



# Extra filtering on age data which are set to nan.

train_set = train_set.drop(train_set.index[isnan(train_set.Age)])

# Preparing the test data set (steps are similar to the trained data set).

test_set = test.drop(['Name', 'Cabin'], 1)



ticket_number = []

for ticket in test_set.Ticket:

    elements = ticket.split(" ")

    if len(elements) > 1:

        ticket_number += [int(elements[len(elements) - 1])]

    else:

        if elements[0] == 'LINE': 

            ticket_number += [0]

        else:

            ticket_number += [int(elements[0])]

ticket_number = array(ticket_number)



test_set.Ticket = ticket_number.astype(int)



test_set.Sex.loc[test_set.Sex == "male"] = 0

test_set.Sex.loc[test_set.Sex == "female"] = 1



test_set.Embarked.loc[test_set.Embarked == "C"] = 0

test_set.Embarked.loc[test_set.Embarked == "Q"] = 1

test_set.Embarked.loc[test_set.Embarked == "S"] = 2





filling_set = concat([train_set, test_set])



y_filling_train = filling_set.loc[~isnan(filling_set.Fare)].Fare.values

X_filling_train = filling_set.loc[~isnan(filling_set.Fare)][["Pclass", "Sex", "SibSp", "Parch", "Ticket", "Embarked"]].values



random_forest = RandomForestRegressor(n_estimators = 1000)



random_forest.fit(X_filling_train, y_filling_train)



X_filling_test = test_set.loc[isnan(test_set.Fare)][["Pclass", "Sex", "SibSp", "Parch", "Ticket", "Embarked"]].values

test_set.Fare.loc[isnan(test_set.Fare)] = random_forest.predict(X_filling_test)



y_filling_train = filling_set.loc[~isnan(filling_set.Age)].Age.values

X_filling_train = filling_set.loc[~isnan(filling_set.Age)][["Pclass", "Sex", "SibSp", "Parch", "Ticket", "Embarked"]].values



random_forest = RandomForestRegressor(n_estimators = 1000)



random_forest.fit(X_filling_train, y_filling_train)



X_filling_test = test_set.loc[isnan(test_set.Age)][["Pclass", "Sex", "SibSp", "Parch", "Ticket", "Embarked"]].values

test_set.Age.loc[isnan(test_set.Age)] = random_forest.predict(X_filling_test)
X = train_set[train_set.columns[1:]].values

y = train_set.Survived.values



X = X.astype(float)



random_forest = RandomForestClassifier(n_estimators = 1000)

random_forest.fit(X, y)



X = test_set[test_set.columns[1:]].values

y = test_set.Survived.values



X = X.astype(float)



print("Score of RF classification:\t" + str(random_forest.score(X, y)) + "\n")

for n in range(len(train_set.columns[1:])):

    print("Importance of " + str(train_set.columns[1 + n]) + ":\t\t" + str(random_forest.feature_importances_[n]))
train_set = train_set.drop(["Embarked"], 1)

test_set = test_set.drop(["Embarked"], 1)



X = train_set[train_set.columns[1:]].values

y = train_set.Survived.values



X = X.astype(float)



random_forest = RandomForestClassifier(n_estimators = 1000)

random_forest.fit(X, y)



X = test_set[test_set.columns[1:]].values

y = test_set.Survived.values



X = X.astype(float)



print("Score of RF classification:\t" + str(random_forest.score(X, y)) + "\n")

for n in range(len(train_set.columns[1:])):

    print("Importance of " + str(train_set.columns[1 + n]) + ":\t\t" + str(random_forest.feature_importances_[n]))
train_set = train_set.drop(["SibSp"], 1)

test_set = test_set.drop(["SibSp"], 1)



X = train_set[train_set.columns[1:]].values

y = train_set.Survived.values



X = X.astype(float)



random_forest = RandomForestClassifier(n_estimators = 1000)

random_forest.fit(X, y)



X = test_set[test_set.columns[1:]].values

y = test_set.Survived.values



X = X.astype(float)



print("Score of RF classification:\t" + str(random_forest.score(X, y)) + "\n")

for n in range(len(train_set.columns[1:])):

    print("Importance of " + str(train_set.columns[1 + n]) + ":\t\t" + str(random_forest.feature_importances_[n]))
train_set = train_set.drop(["Parch"], 1)

test_set = test_set.drop(["Parch"], 1)



X = train_set[train_set.columns[1:]].values

y = train_set.Survived.values



X = X.astype(float)



random_forest = RandomForestClassifier(n_estimators = 1000)

random_forest.fit(X, y)



X = test_set[test_set.columns[1:]].values

y = test_set.Survived.values



X = X.astype(float)



print("Score of RF classification:\t" + str(random_forest.score(X, y)) + "\n")

for n in range(len(train_set.columns[1:])):

    print("Importance of " + str(train_set.columns[1 + n]) + ":\t\t" + str(random_forest.feature_importances_[n]))
train_set = train_set.drop(["Pclass"], 1)

test_set = test_set.drop(["Pclass"], 1)



X = train_set[train_set.columns[1:]].values

y = train_set.Survived.values



X = X.astype(float)



random_forest = RandomForestClassifier(n_estimators = 1000)

random_forest.fit(X, y)



X = test_set[test_set.columns[1:]].values

y = test_set.Survived.values



X = X.astype(float)



print("Score of RF classification:\t" + str(random_forest.score(X, y)) + "\n")

for n in range(len(train_set.columns[1:])):

    print("Importance of " + str(train_set.columns[1 + n]) + ":\t\t" + str(random_forest.feature_importances_[n]))
passengerId = test_set.index

survived = random_forest.predict(X)

survived_real = y



# submission = DataFrame({ 'PassengerId': passengerId, 'Survived': survived, 'Survived_real': y})

submission = DataFrame({ 'PassengerId': passengerId, 'Survived': survived})

submission.to_csv('Titanic.csv', index = False)
submission
