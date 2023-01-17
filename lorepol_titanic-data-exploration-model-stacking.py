# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# import the useful libraries in the project  

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline
# Import the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# print a list of statistics for each variable 

print(train.describe())
print(train.head())
print("The shape of the training set is: ", train.shape)

print("\n The features with null values are:\n ", train.isnull().sum())
# split the set in target variable and features 

targetVariable = train["Survived"]

featuresTrain = train.drop(columns="Survived", axis = 1)

print("The number of features is: ", featuresTrain.shape[1])

print("The number of observations in the training set is: ", featuresTrain.shape[0])
# plot the countplot

plt.figure(figsize=(10,5))

plt.title("Number of Survived/Dead passengers")

plt.xlabel("Survived")

plt.ylabel("Counts")

sns.countplot(targetVariable)
plt.figure(figsize=(10,5))

plt.title("Distribution of survived")

plt.xlabel("Passenger Class")

plt.ylabel("Counts")

sns.countplot(featuresTrain["Pclass"], hue=targetVariable)
plt.figure(figsize=(10,5))

plt.title("Distribution of Survived by Embarking Region")

plt.xlabel("Embarkment Region")

plt.ylabel("Counts")

sns.countplot(featuresTrain["Embarked"], hue=targetVariable)
plt.figure(figsize=(10,5))

plt.title("Distribution of Embarked")

plt.xlabel("Embarked")

plt.ylabel("Counts")

sns.countplot(featuresTrain["Embarked"], hue=featuresTrain["Sex"])
plt.figure(figsize=(10,5))

plt.title("Distribution of survived")

plt.xlabel("Sex")

plt.ylabel("Counts")

sns.countplot(featuresTrain["Sex"], hue=targetVariable)
plt.figure(figsize=(10,5))

plt.title("Distribution of survived")

plt.xlabel("Fare")

plt.ylabel("Distribution")

sns.distplot(featuresTrain["Fare"])
plt.figure(figsize=(10,5))

plt.title("Distribution of survived")

plt.xlabel("Survived")

plt.ylabel("SibSp")

sns.countplot(featuresTrain["SibSp"], hue = targetVariable)
#store the id for the passengers

passengerId = featuresTrain["PassengerId"]

#drop that column from the training set 

featuresTrain.drop(columns = "PassengerId", inplace = True)
print("There number of occurrences of S is: ",featuresTrain[featuresTrain["Embarked"] == "S"]["Embarked"].count())

print("There number of occurrences of C is: ",featuresTrain[featuresTrain["Embarked"] == "C"]["Embarked"].count())

print("There number of occurrences of Q is: ",featuresTrain[featuresTrain["Embarked"] == "Q"]["Embarked"].count())
featuresTrain["Embarked"].fillna("S", inplace = True)

print("The number of missing values for Embarked is: ", featuresTrain["Embarked"].isnull().sum())
agenotMissing = featuresTrain[featuresTrain["Age"].isnull() == False]["Age"]

print("The median of the Age variable is: ", np.median(agenotMissing))

featuresTrain["Age"].fillna(np.median(agenotMissing), inplace = True)
print("The number of missing values for Age is now: ", featuresTrain["Age"].isnull().sum())
featuresTrain.drop(columns = ["Cabin", "Ticket"], inplace = True)
featuresTrain.loc[featuresTrain["Age"] <= 6,"AgeType"] = "Baby"

featuresTrain.loc[(featuresTrain["Age"] > 6) & (featuresTrain["Age"] <= 18),"AgeType"] = "Child"

featuresTrain.loc[(featuresTrain["Age"] > 18) & (featuresTrain["Age"] <= 40),"AgeType"] = "YoungAdult"

featuresTrain.loc[(featuresTrain["Age"] > 40) & (featuresTrain["Age"] <= 60), "AgeType"] = "Adult"

featuresTrain.loc[featuresTrain["Age"] > 60, "AgeType"] = "Old"
plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.title("Class of Passengers")

plt.xlabel("Types")

plt.ylabel("Distribution")

sns.countplot(featuresTrain["AgeType"], hue=featuresTrain["Pclass"])

plt.subplot(1,2,2)

plt.title("Survival Rate")

plt.xlabel("Types")

plt.ylabel("Distribution")

sns.countplot(featuresTrain["AgeType"], hue=targetVariable)



print(featuresTrain["Name"])
def getTitlePasseger(data):

    titles = []

    for i in data:

        titlePassenger = i.split(" ")[1]

        titles.append(titlePassenger.strip(titlePassenger[-1]))

    return titles

# we call the function to fill the new column of the database

featuresTrain["TitlePassenger"] = getTitlePasseger(featuresTrain["Name"])

plt.figure(figsize=(20,10))

plt.title("Distribution of Titles")

plt.xlabel("Titles")

plt.xticks(rotation = 90)

plt.ylabel("Distribution")

sns.countplot(featuresTrain["TitlePassenger"], hue = targetVariable)
featuresTrain.drop(columns = ["Name","Embarked"], inplace = True)
from sklearn.preprocessing import RobustScaler

featuresTrain["Fare"] = np.log1p(featuresTrain["Fare"])

featuresTrain["Age"] = np.log1p(featuresTrain["Age"])

scaler = RobustScaler()

scaler.fit(featuresTrain[["Age", "SibSp", "Fare", "Pclass", "Parch"]])

featuresTrain.loc[:,["Age", "SibSp", "Fare", "Pclass", "Parch"]] = scaler.transform(featuresTrain[["Age", "SibSp", "Fare", "Pclass", "Parch"]])
sns.distplot(featuresTrain["Fare"])
sns.distplot(featuresTrain["Age"])
from pandas import get_dummies

featuresTrain = get_dummies(featuresTrain, drop_first=True, )

featuresTrain.drop(columns = ["TitlePassenger_Mulder", "TitlePassenger_Velde", "TitlePassenger_Walle", "TitlePassenger_Steen", "TitlePassenger_Cruyssen", "TitlePassenger_Shawah", "TitlePassenger_Impe", "TitlePassenger_Capt", "TitlePassenger_Jonkheer", "TitlePassenger_Melkebeke", "TitlePassenger_Mme", "TitlePassenger_Gordon", "TitlePassenger_Don", "TitlePassenger_Mlle", 

                              "TitlePassenger_de", "TitlePassenger_Major", "TitlePassenger_Pelsmaeker", "TitlePassenger_th"])

print("The shape of the training data is: ", featuresTrain.shape)
plt.figure(figsize=(20,10))

plt.title("Correlation Distribution")

plt.xlabel("Features")

plt.ylabel("Features")

plt.xticks(rotation = 90)

allData = pd.concat([featuresTrain, targetVariable], axis = 1)

sns.heatmap(allData.corr(), annot=True)
from sklearn.model_selection import KFold 

from sklearn.metrics import accuracy_score



def crossValidation(data,target, model, cv = 5):

    ksplits = KFold(n_splits=cv)

    errors = []

    for j, (indexTrain, indexTest) in enumerate(ksplits.split(data, target)):

        print("This is fold ", j + 1, "of the cross validation")

        #define the training set

        x_train, y_train = data.iloc[indexTrain, :], target.iloc[indexTrain]

        #define the test set

        x_test, y_test = data.iloc[indexTest, :], target.iloc[indexTest]

        print("Fitting the model")

        #here we have the option of including a neural network

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        errorFold = accuracy_score(y_test, predictions)

        errors.append(errorFold)

    print("The mean accuracy score over the folds is: ", np.mean(errors))

    return
from sklearn.linear_model import LogisticRegression

modelLogistic = LogisticRegression(solver="liblinear")

crossValidation(featuresTrain, targetVariable, modelLogistic)
import xgboost

modelXGboost = xgboost.XGBClassifier()

crossValidation(featuresTrain, targetVariable, modelXGboost)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

modelDiscriminant = LinearDiscriminantAnalysis()

crossValidation(featuresTrain, targetVariable, modelDiscriminant)
from sklearn.ensemble import RandomForestClassifier

modelForest = RandomForestClassifier()

crossValidation(featuresTrain, targetVariable, modelForest)
class stackerModel(object):

    def __init__(self, models,metalearner, ksplits = 5):

        #get the models that we want to stack 

        self.models = models

        self.ksplits = ksplits

        self.metalearner = metalearner



    def stackingActive(self, x_train, y_train, x_test):

        #define the outoffold predictions arrays

        kfolds = KFold(n_splits=self.ksplits, random_state=0, shuffle=True)

        oof_train = np.zeros(x_train.shape[0])

        oof_test = np.zeros(x_test.shape[0])

        oof_test_skf = np.empty((self.ksplits, x_test.shape[0]))

        oof_trains_ensemble = []

        oof_tests_ensemble = []

        #declare the array for the different models 



        for j,model in enumerate(self.models):

            print("The model that we are using now is: ", model)

            #performs also cross validation 

            for i, (indexTrain, indexTest) in enumerate(kfolds.split(x_train, y_train)):

                x_tr = x_train.iloc[indexTrain, :]

                y_tr = y_train.iloc[indexTrain]

                x_te = x_train.iloc[indexTest, :]

                #take into account the neural network

                model.fit(x_tr, y_tr)

                oof_train[indexTest] = model.predict(x_te).ravel()

                oof_test_skf[i,:] = model.predict(x_test).ravel()

            oof_test[:] = oof_test_skf.mean(axis = 0)

            #we are creating a list of arrays that we use later on 

            oof_trains_ensemble.append(oof_train.reshape(-1,1))

            oof_tests_ensemble.append(oof_test.reshape(-1,1))

            

        #here we need to use the metalearner

        trainMeta = oof_trains_ensemble[0]

        testMeta = oof_tests_ensemble[0]

        #here we concatenate the new features (oof predictions)

        for i in range(1,len(self.models)):

            trainMeta = np.concatenate((trainMeta, oof_trains_ensemble[i]), axis = 1)

            testMeta = np.concatenate((testMeta, oof_tests_ensemble[i]), axis = 1)

        print("\n I am now fitting the metalearner")

        self.metalearner.fit(trainMeta, y_train)

        predictions = self.metalearner.predict(testMeta)

        return predictions
def crossValMixed(X, y, model,cv = 5):

    n_folds = KFold(n_splits = cv,shuffle=True)

    counter = 0

    errorMetr = []

    #here you define the permutation importance

    for indexTrain, indexTest in n_folds.split(X,y):

        print("This is fold: ", counter, "of the cross validation")

        X_train, y_train = X.iloc[indexTrain, :], y.iloc[indexTrain]

        X_test, y_test = X.iloc[indexTest, :], y.iloc[indexTest]

        print("Fitting the model")

        predictions = model.stackingActive(X_train, y_train, X_test)

        print("The metric in the fold ", counter, "is: ", accuracy_score(y_test,predictions))

        counter += 1

        errorMetr.append(accuracy_score(y_test,predictions))

    print("The mean absolute error over the ", cv, "folds is: ", np.mean(errorMetr))
from sklearn.neighbors import KNeighborsClassifier

modelK = KNeighborsClassifier()

modelsStacked = [modelXGboost, modelK, modelXGboost]

modelMeta = modelLogistic

modelStackedClass = stackerModel(modelsStacked, modelMeta)

crossValMixed(featuresTrain, targetVariable, modelStackedClass) 