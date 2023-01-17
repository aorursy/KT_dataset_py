# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn import linear_model

from enum import Enum

from subprocess import check_output

from statsmodels.graphics.mosaicplot import mosaic

import sklearn

# Load data

trainingData = pd.read_csv("../input/train.csv")

testData = pd.read_csv("../input/test.csv")
# Used to convert strings into ordinals ASSUMES WELL FORMATED STRING

def SexSort(sexOfPassenger):

    sex = 2; # Rather then catch the error I will simply return a 2, for now

    if(sexOfPassenger.find("female") != -1):

        sex = 1;

    elif(sexOfPassenger.find("male") != -1):

        sex = 0; # If survival is not based on sex, then the correlation will tend towards zero

    return sex



# Check which categories are missing data

print(trainingData.isnull().sum())

print(testData.isnull().sum())

# Drop categories which have alot of missing or useless data

trainingData = trainingData.drop(['Name','PassengerId','Cabin'], axis=1)

testData = testData.drop(['Name','Cabin'], axis=1)



trainingData['Gender'] = trainingData['Sex'].apply(SexSort)

testData['Gender'] = testData['Sex'].apply(SexSort)

print(trainingData.info())
colormap = plt.cm.viridis

plt.figure(figsize=(12, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(trainingData.drop("Sex",axis = 1).drop("Ticket",axis=1).drop("Embarked",axis = 1).astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',annot=True)
plt.figure()

sns.countplot(x='Gender', data=trainingData, order=[0,1])

plt.title("Distribution by gender")

plt.xlabel("Gender (0 = male, 1 = female)")



plt.figure()

plt.title("Survival by gender")

sns.countplot(x='Gender', hue="Survived", data=trainingData, order=[0,1])

plt.ylabel("Number of people")

plt.xlabel("Gender (0 = male, 1 = female)")

plt.legend(["Dead","Survived"])



plt.figure()

plt.title("Survival percentage by gender")

sns.barplot(x='Gender',y="Survived", data=trainingData, estimator=lambda x: sum(x==1)/len(x)*100)

plt.ylabel("Percentage survived")

plt.xlabel("Gender (0 = male, 1 = female)")

# NTS: Consider using a binomial(Bernouli) distribution
sns.violinplot(x="Gender", y="Age", hue="Survived", data=trainingData, split=True)

plt.hlines([0,16], xmin=-1, xmax=3, linestyles="dotted")

plt.hlines([60,80], xmin=-1, xmax=3, linestyles="dotted")
plt.figure()

sibsp = pd.crosstab(trainingData['SibSp'], trainingData['Sex'])

sibsp.div(sibsp.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Siblings')

plt.ylabel('Percent survived')



plt.figure()

parch = pd.crosstab(trainingData['Parch'], trainingData['Sex'])

parch.div(parch.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Number of parents or children')

plt.ylabel('Percent survived')
def FamilySize(famTuple):

    sib,par = famTuple

    return sib+par



plt.figure()

sns.pairplot(data = trainingData[['SibSp','Parch','Survived']].dropna(), size=1.5,hue='Survived')

trainingData['FamilySize'] = trainingData[['SibSp','Parch']].apply(FamilySize,axis = 1)



plt.figure()

sns.barplot(x='FamilySize',y="Survived", data=trainingData, estimator=lambda x: sum(x==1)/len(x)*100)
plt.figure()

sns.countplot(x='Pclass', data=trainingData)

plt.title("Total for each class")

plt.xlabel("Class")



plt.figure()

plt.title("Survival by class")

sns.countplot(x='Pclass', hue="Survived", data=trainingData)

plt.ylabel("Number of people")

plt.xlabel("Class")

plt.legend(["Dead","Survived"])



plt.figure()

plt.title("Survival percentage by class")

sns.barplot(x='Pclass',y="Survived", data=trainingData, estimator=lambda x: sum(x==1)/len(x)*100)

plt.ylabel("Percentage survived")

plt.xlabel("Class")



# This graph is not particularly oin



plt.figure()

plt.title("Survival as a function of gender and class")

sns.barplot(x='Pclass',y="Survived", hue = 'Gender',data=trainingData, estimator=lambda x: sum(x==1)/len(x)*100)

plt.ylabel("Percentage of survivors")

plt.xlabel("Class")

plt.legend(["Male","Female"])



plt.figure()

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=trainingData, split=True)

plt.title("Survival as a function of class and age")
sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=trainingData)
print(trainingData['Ticket'].nunique(),trainingData['Ticket'].count()) #identifies unique tickets
# TAKEN FROM PYTANIC KERNEL, ALL CREDIT GOES TO HEADS OR TAILS

grouped = trainingData.groupby('Ticket')

k = 0

for name, group in grouped:

    if (len(grouped.get_group(name)) > 1):

        print(group.loc[:,['Survived','Name', 'Fare']])

        k += 1

    if (k>20):

        break

#trainingData['SharesATicket'] = trainingData['Ticket'].duplicated(keep = 'first').apply(or trainingData['Ticket'].duplicated(keep = 'last'))

#print(trainingData["Ticket"])

#print(trainingData["Ticket"].sum)
# Functions

import math

CLASS_MAX_VAL = 3

BOX_SIZE = 10

# Creates a ordinal ranking based on class and gender (maps gender)

def ClassGenderFunction(cgTuple):

    pclass,gender = cgTuple

    return ((CLASS_MAX_VAL)*gender + ((CLASS_MAX_VAL+1)-pclass)) #creates a unique mapping



# Returns a "child" boolean ASSUMES INTEGER INPUT

def IsAChild(ageOfPassenger):

    if(math.isnan(ageOfPassenger)):

        return 0

    else:

        return ageOfPassenger <= 16;



# Returns a "family" boolean ASSUMES INTEGER INPUT

def HasAFamily(familyInfo):

    siblingSpouse,parentChild = familyInfo

    return (siblingSpouse != 0 or parentChild != 0)



def IsAWomenWithAFamily(personalInfo):

    sex,siblingSpouse,parentChild = personalInfo

    return (SexSort(sex) == 1 and HasAFamily([siblingSpouse,parentChild]))

def NormFamilySize(famSize):

    return np.ceil(famSize/BOX_SIZE)



trainingData['HasAFamily'] = trainingData[['SibSp','Parch']].apply(HasAFamily,axis = 1)

trainingData['ClassGenderCoeff'] = trainingData[['Pclass','Gender']].apply(ClassGenderFunction,axis = 1)

trainingData['IsAChild'] = trainingData['Age'].apply(IsAChild)

trainingData['IsAWomenWithAFamily'] = trainingData[['Sex','SibSp','Parch']].apply(IsAWomenWithAFamily,axis = 1)

trainingData['NormedFamilySize'] = trainingData['FamilySize'].apply(NormFamilySize)



testData['HasAFamily'] = testData[['SibSp','Parch']].apply(HasAFamily,axis = 1)

testData['ClassGenderCoeff'] = testData[['Pclass','Gender']].apply(ClassGenderFunction,axis = 1)

testData['IsAChild'] = testData['Age'].apply(IsAChild)

testData['IsAWomenWithAFamily'] = testData[['Sex','SibSp','Parch']].apply(IsAWomenWithAFamily,axis = 1)



 
plt.figure(figsize=(12, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(trainingData.drop("Sex",axis = 1).drop("Ticket",axis = 1).drop("Embarked",axis = 1).astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',annot=True)
plt.figure()

mosaic(trainingData,["Survived","HasAFamily","Pclass"]);
from sklearn.model_selection import cross_val_score



retainedClassifiers = ['ClassGenderCoeff','HasAFamily','IsAChild']

retainedClassifiersOutputs = np.append(['Survived'],retainedClassifiers)



tempDataFrame = trainingData.loc[:,retainedClassifiersOutputs].dropna()

inputs = tempDataFrame.loc[:,retainedClassifiers]

outputs = np.ravel(tempDataFrame.loc[:,['Survived']])



testDataFrame = testData.loc[:,retainedClassifiersOutputs]

testInputs = testDataFrame.loc[:,retainedClassifiers]



trainingDataLR = linear_model.LogisticRegression()

trainingDataLR = trainingDataLR.fit(inputs,outputs)

logRegScore = cross_val_score(trainingDataLR, inputs, outputs, cv=5).mean()

print(logRegScore)
from sklearn.ensemble import RandomForestClassifier

rfClassifier = RandomForestClassifier(n_estimators = 1000,max_depth = None, min_samples_split = 10)

rfClassifier.fit(inputs,outputs)

scoreRandomForest = cross_val_score(rfClassifier, inputs, outputs, cv=5).mean()

print(scoreRandomForest)





from sklearn import tree



treeClassifier = tree.DecisionTreeClassifier(

    class_weight="balanced",\

    min_weight_fraction_leaf=0.01\

    )

treeClassifier = treeClassifier.fit(inputs,outputs)

treeScore = cross_val_score(treeClassifier, inputs, outputs, cv=5).mean()

print(treeScore)



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

LDAClassifier = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', priors=None, n_components=None, store_covariance=False, tol=0.0001)

LDAClassifier.fit(inputs,outputs)

QDAClassifier = QuadraticDiscriminantAnalysis()

QDAClassifier.fit(inputs,outputs)

LDAScore = cross_val_score(LDAClassifier, inputs, outputs, cv=5).mean()

print(treeScore)

QDAScore = cross_val_score(QDAClassifier, inputs, outputs, cv=5).mean()

print(treeScore)


surivivalPredictions = rfClassifier.predict(testInputs)



submission = pd.DataFrame({'PassengerId' : testData.loc[:,'PassengerId'],

                       'Survived': surivivalPredictions.T})



print(submission)

submission.to_csv("../working/submit.csv", index=False)