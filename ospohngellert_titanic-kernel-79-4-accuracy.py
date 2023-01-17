import pandas as pd

import random

from sklearn.neural_network import MLPClassifier

import numpy

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import re

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import matplotlib.path as path

import scipy.stats as stats

import math

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.describe()
sum(pd.isnull(train.Age))/len(train)
ageNoNa = train.Age[pd.isnull(train.Age) == False]

plt.hist(ageNoNa)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for known training data.")

plt.show()

print("Mean age: " + str(numpy.mean(ageNoNa)))

print("Median age: " + str(numpy.median(ageNoNa)))

print("Standard debiation: " + str(numpy.std(ageNoNa)))

print("Normality: " + str(stats.normaltest(ageNoNa)))
ageHasSibs = train[(train.SibSp > 0) & (pd.isnull(train.Age) == False)].Age

plt.hist(ageHasSibs)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have siblings.")

plt.show()

print("Mean age: " + str(numpy.mean(ageHasSibs)))

print("Median age: " + str(numpy.median(ageHasSibs)))
ageHasNoSibs = train[(train.SibSp == 0) & (pd.isnull(train.Age) == False)].Age

plt.hist(ageHasNoSibs)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who don't have siblings.")

plt.show()

print("Mean age: " + str(numpy.mean(ageHasNoSibs)))

print("Median age: " + str(numpy.median(ageHasNoSibs)))
ageHasParch = train[(train.Parch > 0) & (pd.isnull(train.Age) == False)].Age

plt.hist(ageHasParch)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have parents/children.")

plt.show()

print("Mean age: " + str(numpy.mean(ageHasParch)))

print("Median age: " + str(numpy.median(ageHasParch)))
ageHasNoParch = train[(train.Parch == 0) & (pd.isnull(train.Age) == False)].Age

plt.hist(ageHasNoParch)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who don't have siblings.")

plt.show()

print("Mean age: " + str(numpy.mean(ageHasNoParch)))

print("Median age: " + str(numpy.median(ageHasNoParch)))
ageOfMaster = train[train.Name.str.contains('Master.') & (pd.isnull(train.Age) == False)].Age

plt.hist(ageOfMaster)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have the title Master.")

plt.show()

print("Mean age: " + str(numpy.mean(ageOfMaster)))

print("Median age: " + str(numpy.median(ageOfMaster)))
ageOfMiss = train[train.Name.str.contains('Miss.') & (pd.isnull(train.Age) == False)].Age

plt.hist(ageOfMiss)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have the title Miss.")

plt.show()

print("Mean age: " + str(numpy.mean(ageOfMiss)))

print("Median age: " + str(numpy.median(ageOfMiss)))
ageOfMr = train[train.Name.str.contains('Mr.') & (pd.isnull(train.Age) == False)].Age

plt.hist(ageOfMr)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have the title Mr.")

plt.show()

print("Mean age: " + str(numpy.mean(ageOfMr)))

print("Median age: " + str(numpy.median(ageOfMr)))
ageOfMrs = train[train.Name.str.contains('Mrs.') & (pd.isnull(train.Age) == False)].Age

plt.hist(ageOfMr)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have the title Mrs.")

plt.show()

print("Mean age: " + str(numpy.mean(ageOfMrs)))

print("Median age: " + str(numpy.median(ageOfMrs)))
ageOfMissWithParch = train[train.Name.str.contains('Miss.') & (train.Parch > 0) & (pd.isnull(train.Age) == False)].Age

plt.hist(ageOfMissWithParch)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have the title Miss and have Parents/Children.")

plt.show()

print("Mean age: " + str(numpy.mean(ageOfMissWithParch)))

print("Median age: " + str(numpy.median(ageOfMissWithParch)))
ageOfMissWithoutParch = train[train.Name.str.contains('Miss.') & (train.Parch == 0) & (pd.isnull(train.Age) == False)].Age

plt.hist(ageOfMissWithoutParch)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have the title Miss and don't have Parents/Children.")

plt.show()

print("Mean age: " + str(numpy.mean(ageOfMissWithoutParch)))

print("Median age: " + str(numpy.median(ageOfMissWithoutParch)))
ageOfMrsWithParch = train[train.Name.str.contains('Mrs.') & (train.Parch > 0) & (pd.isnull(train.Age) == False)].Age

plt.hist(ageOfMrsWithParch)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have the title Mrs and have Parents/Children.")

plt.show()

print("Mean age: " + str(numpy.mean(ageOfMrsWithParch)))

print("Median age: " + str(numpy.median(ageOfMrsWithParch)))
ageOfMrsWithoutParch = train[train.Name.str.contains('Mrs.') & (train.Parch == 0) & (pd.isnull(train.Age) == False)].Age

plt.hist(ageOfMrsWithoutParch)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age distribution for people who have the title Mrs and don't have Parents/Children.")

plt.show()

print("Mean age: " + str(numpy.mean(ageOfMrsWithoutParch)))

print("Median age: " + str(numpy.median(ageOfMrsWithoutParch)))
fare = train.Fare

plt.hist(fare)

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.title("Price distribution for known training data.")

plt.show()

print("Mean fare: " + str(numpy.mean(fare)))

print("Median fare: " + str(numpy.median(fare)))

print("Standard deviation: " + str(numpy.std(fare)))

print("Normality: " + str(stats.normaltest(fare)))
fareOver181 = train[train.Fare > 181]

fareBetween100And181 = train[(train.Fare > 100) & (train.Fare < 181)]

fareUnder181 = train[(train.Fare < 181)].Fare

plt.hist(fareUnder181)

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.title("Price distribution for fares under 181.")

plt.show()

print("Survival rate for those whose fare is > 181: " + str(sum(fareOver181.Survived) / len(fareOver181.Survived)))

print("Survival rate for those whose fare is < 181 and > 100: " + str(sum(fareBetween100And181.Survived) / len(fareBetween100And181.Survived)))



print("Normality: " + str(stats.normaltest(fareUnder181)))

rootFareUnder181 = fareUnder181.map(lambda x: math.sqrt(x))

plt.hist(rootFareUnder181)

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.title("Price distribution for square roots of fares under 181.")

plt.show()

print("Normality: " + str(stats.normaltest(rootFareUnder181)))
logFareUnder181 = fareUnder181.map(lambda x: math.log(x, 10) if x != 0 else x)

plt.hist(rootFareUnder181)

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.title("Price distribution for logs of fares under 181.")

plt.show()

print("Normality: " + str(stats.normaltest(logFareUnder181)))
def ageFunc(x):

    age = x['Age']

    name = x['Name']

    sibs = x['SibSp']

    if math.isnan(age):

        if "Master." in name:

            x['Age'] = 5

        elif "Miss." in name and sibs > 0:

            x['Age'] = 11

        elif "Miss." in name and sibs == 0:

            x['Age'] = 27

        elif "Mr." in name:

            x['Age'] = 32

        elif "Mrs." in name:

            x['Age'] = 36

        else:

            x['Age'] = 29

    return x

    

def embarkedFunc(x):

    vals = {'Q': 1, 'S': 2, 'C': 3}

    return vals.get(x, 0)
def cleanData(frame, isTest):

    averageFare = numpy.mean(frame.Fare)

    frame = frame.apply(ageFunc, axis='columns')

    frame.Fare = frame.Fare.map(lambda x: x if not numpy.isnan(x) else averageFare)

    out = frame.query('Fare < 181').copy() if not isTest else frame.copy()

    out.Sex = out.Sex == 'female'

    out.Fare = out.Fare.map(lambda y: math.log(y, 10) if y != 0 else y)

    out.Ticket = out.Ticket.map(lambda x: not pd.isnull(re.search("[a-zA-Z]", x)))

    out['emS'] = out.Embarked == 'S'

    out['emC'] = out.Embarked == 'C'

    out['class1'] = out.Pclass == 1

    out['class2'] = out.Pclass == 2

    out.Embarked = out.Embarked.map(embarkedFunc)

    out.Cabin = out.Cabin.map(lambda x: not pd.isnull(x))

    return out.iloc[:,out.columns.get_level_values(0).isin({"Survived", "PassengerId", "Fare", 'class1', 'class2', "Sex", "Age", "SibSp", "Parch", "Cabin", "Ticket", 'emS', 'emC'})]



train = cleanData(train, False)

test = cleanData(test, True)
train.corr()
ageLessEq15 = train[(train.Age <= 15)].Survived

ageBetween16and35 = train[(train.Age > 15) & (train.Age <= 35)].Survived

ageBetween36and60 = train[(train.Age > 35) & (train.Age <= 60)].Survived

ageGreater60 = train[(train.Age > 60)].Survived

print("<= 15 survival: " + str(sum(ageLessEq15) / len(ageLessEq15)))

print("16 - 35 survival: " + str(sum(ageBetween16and35) / len(ageBetween16and35)))

print("36 - 60 survival: " + str(sum(ageBetween36and60) / len(ageBetween36and60)))

print("> 60 survival: " + str(sum(ageGreater60) / len(ageGreater60)))
train['isChild'] = train.Age <= 15

train['isSenior'] = train.Age > 60

train.corr()
print("isChild Size: {}".format(len(ageLessEq15)))

print("isSenior Size: {}".format(len(ageGreater60)))
train = train.drop("isSenior", axis=1, errors='ignore')

test["isChild"] = test.Age <= 15
plt.hist(train.SibSp)

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.title("Distribution of Siblings")

plt.show()



plt.hist(train.Parch)

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.title("Distributions of Parch")

plt.show()
train["FamilySize"] = train.SibSp + train.Parch

plt.hist(train.FamilySize)

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.title("Distributions of Parch")

plt.show()

train.corr()
train = train.drop("FamilySize", axis=1, errors='ignore')
def createSplitNoColinear(t):

    train, cv = train_test_split(t, test_size = 0.2)

    X_train = preprocessing.scale(numpy.transpose([train.isChild, train.Sex, train.Parch, train.Fare, train.emS, train.emC, train.Ticket]))

    Y_train = numpy.transpose(train.Survived)

    X_cv = preprocessing.scale(numpy.transpose([cv.isChild, cv.Sex, cv.Parch, cv.Fare, cv.emS, cv.emC, cv.Ticket]))

    Y_cv = numpy.transpose(cv.Survived) 

    return {

        "X_train": X_train,

        "Y_train": Y_train,

        "X_cv": X_cv,

        "Y_cv": Y_cv

    }



def createSplitColinear(t):

    train, cv = train_test_split(t, test_size = 0.2)

    X_train = preprocessing.scale(numpy.transpose([train.isChild, train.Sex, train.Parch, train.Fare, train.emS, train.emC, train.Ticket, train.class1, train.class2, train.Age, train.SibSp, train.Cabin]))

    Y_train = numpy.transpose(train.Survived)

    X_cv = preprocessing.scale(numpy.transpose([cv.isChild, cv.Sex, cv.Parch, cv.Fare, cv.emS, cv.emC, cv.Ticket, cv.class1, cv.class2, cv.Age, cv.SibSp, cv.Cabin]))

    Y_cv = numpy.transpose(cv.Survived) 

    return {

        "X_train": X_train,

        "Y_train": Y_train,

        "X_cv": X_cv,

        "Y_cv": Y_cv

    }





def testData(split, x):

    layer_sizes = [len(split['X_train'][0]) * 3, len(split['X_train'][0])]

    neural =  MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(layer_sizes[0], layer_sizes[1], 1), random_state=1)

    forest = RandomForestClassifier(max_depth=5, random_state=0)

    machine = svm.SVC(kernel='rbf')

    neural.fit(split['X_train'], split['Y_train'])

    forest.fit(split['X_train'], split['Y_train'])

    machine.fit(split['X_train'], split['Y_train'])

    nSurvived = neural.predict(split['X_cv'])

    fSurvived = forest.predict(split['X_cv'])

    sSurvived = machine.predict(split['X_cv'])

    vSurvived = list(map(lambda x: int(sum(x) > 2), zip(nSurvived, fSurvived, sSurvived)))

    x['neural'] = sum(nSurvived == split['Y_cv'])/len(nSurvived)

    x['forest'] = sum(fSurvived == split['Y_cv'])/len(fSurvived)

    x['svm'] = sum(sSurvived == split['Y_cv'])/len(sSurvived)

    x['vote'] = sum(vSurvived == split['Y_cv'])/len(vSurvived)

    return x
TRIALS = 50

predictions = pd.DataFrame({'neural': range(TRIALS), 'forest': range(TRIALS), 'svm': range(TRIALS), 'vote': range(TRIALS)}, dtype=float)

predictions = predictions.apply(lambda x: testData(createSplitNoColinear(train), x), axis=1)

predictions.describe()
TRIALS = 50

predictions = pd.DataFrame({'neural': range(TRIALS), 'forest': range(TRIALS), 'svm': range(TRIALS), 'vote': range(TRIALS)}, dtype=float)

predictions = predictions.apply(lambda x: testData(createSplitColinear(train), x), axis=1)

predictions.describe()
def testCVNeural(split, x, reg):

    layer_sizes = [len(split['X_train'][0]) * 3, len(split['X_train'][0])]

    for a in reg:

        neural =  MLPClassifier(solver='lbfgs', alpha=a, hidden_layer_sizes=(layer_sizes[0], layer_sizes[1], 1), random_state=1)

        neural.fit(split['X_train'], split['Y_train'])

        nSurvived = neural.predict(split['X_cv'])

        x[str(a)] = sum(nSurvived == split['Y_cv'])/len(nSurvived)

    return x
TRIALS = 50

reg = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100]

regDict = {}

for a in reg:

    regDict[str(a)] = range(TRIALS)

predictions = pd.DataFrame(regDict, dtype=float)

predictions = predictions.apply(lambda x: testCVNeural(createSplitColinear(train), x, reg), axis=1)

predictions.describe()
def testCVSVM(split, x, reg):

    layer_sizes = [len(split['X_train'][0]) * 3, len(split['X_train'][0])]

    for a in reg:

        machine = svm.SVC(kernel='rbf', C=a)

        machine.fit(split['X_train'], split['Y_train'])

        sSurvived = machine.predict(split['X_cv'])

        x[str(a)] = sum(sSurvived == split['Y_cv'])/len(sSurvived)

    return x
TRIALS = 50

reg = [1e-3, 3e-3, 1e-1, 3e-1, 1, 3, 10]

regDict = {}

for a in reg:

    regDict[str(a)] = range(TRIALS)

predictions = pd.DataFrame(regDict, dtype=float)

predictions = predictions.apply(lambda x: testCVSVM(createSplitColinear(train), x, reg), axis=1)

predictions.describe()
X_train = preprocessing.scale(numpy.transpose([train.isChild, train.Sex, train.Parch, train.Fare, train.emS, train.emC, train.Ticket, train.class1, train.class2, train.Age, train.SibSp, train.Cabin]))

Y_train = numpy.transpose(train.Survived)

X_test = preprocessing.scale(numpy.transpose([test.isChild, test.Sex, test.Parch, test.Fare, test.emS, test.emC, test.Ticket, test.class1, test.class2, test.Age, test.SibSp, test.Cabin]))

layer_sizes = [len(X_train[0]) * 3, len(X_train[0])]

neural =  MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(layer_sizes[0], layer_sizes[1]), random_state=1)    

forest = RandomForestClassifier(max_depth=5, random_state=0)

machine = svm.SVC(kernel='rbf', C=0.3)

neural.fit(X_train, Y_train)

forest.fit(X_train, Y_train)

machine.fit(X_train, Y_train)

nSurvived = neural.predict(X_test)

fSurvived = forest.predict(X_test)

sSurvived = machine.predict(X_test)

vSurvived = list(map(lambda x: int(sum(x) > 2), zip(nSurvived, fSurvived, sSurvived)))



pd.DataFrame({"PassengerId": test.PassengerId, "Survived": nSurvived}).to_csv("neural.csv", index=False)

pd.DataFrame({"PassengerId": test.PassengerId, "Survived": fSurvived}).to_csv("forest.csv", index=False)

pd.DataFrame({"PassengerId": test.PassengerId, "Survived": sSurvived}).to_csv("svm.csv", index=False)

pd.DataFrame({"PassengerId": test.PassengerId, "Survived": vSurvived}).to_csv("vote.csv", index=False)