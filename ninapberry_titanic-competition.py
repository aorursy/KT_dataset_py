import numpy as np 

import pandas as pd 

import statistics

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



# Read in training and testing data

trainFull = pd.read_csv("/kaggle/input/titanic/train.csv")

testFull = pd.read_csv("/kaggle/input/titanic/test.csv")



# Example rows of the dataset

trainFull.sample(5)
# Look for variables with a large number of missing values in training data

trainFull.describe(include = "all")
# Equivalent for testing data

testFull.describe(include="all")
# Drop 'Cabin' due to high number of missing values, but note whether this data was collected

trainFull['CabinData'] = trainFull.Cabin.notnull().astype('int')

testFull['CabinData'] = testFull.Cabin.notnull().astype('int')

trainFull.drop(['Cabin'],axis=1, inplace=True)

testFull.drop(['Cabin'], axis=1,inplace=True)



# Drop 'Ticket' as seems irrelevant data

testFull.drop(['Ticket'], axis=1, inplace=True)

trainFull.drop(['Ticket'], axis=1, inplace=True)



# Replace missing values in Embarked in training data with most common departure port

mostCommonOrigin = statistics.mode(trainFull.Embarked)

trainFull.Embarked[trainFull.Embarked.isna()] = mostCommonOrigin



# Convert 'Name' to more meaningful 'Title' data

allData = [trainFull, testFull]

for dataset in allData:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.concat([trainFull,testFull],axis=0).groupby('Title').size())
for dataset in allData:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dona', 'Jonkheer', 'Major', 'Rev'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Ms','Mlle'], 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].replace(['Sir','Lady','Countess'],'Royal')

    dataset.drop(['Name'], axis=1,inplace=True)



# See effect of title on survival in training data    

sns.barplot(x="Title", y="Survived", data=trainFull)  



# See where ages are missing

print("In training data:\n",trainFull[trainFull.Age.isna()].groupby('Title').size())

print("In testing data:\n",testFull[testFull.Age.isna()].groupby('Title').size())
# Find mean ages for titles with missing ages

combined = pd.concat([trainFull,testFull],axis=0)

avgMaleDrAge = np.mean(combined.Age[(combined.Title == 'Dr') & ~(combined.Age.isna()) & (combined.Sex == 'male')]) # Dr with missing age is male

avgMasAge = np.mean(combined.Age[(combined.Title == 'Master') & ~(combined.Age.isna())])

avgMrAge = np.mean(combined.Age[(combined.Title == 'Mr') & ~(combined.Age.isna())])

avgMsAge = np.mean(combined.Age[(combined.Title == 'Miss') & ~(combined.Age.isna())])

avgMrsAge = np.mean(combined.Age[(combined.Title == 'Mrs') & ~(combined.Age.isna())])



# Replace missing ages with title mean ages

trainFull.Age[(trainFull.Title == 'Dr') & (trainFull.Age.isna())] = avgMaleDrAge

for dataset in allData:

    dataset.Age[(dataset.Title == 'Master') & (dataset.Age.isna())] = avgMasAge

    dataset.Age[(dataset.Title == 'Mr') & (dataset.Age.isna())] = avgMrAge

    dataset.Age[(dataset.Title == 'Miss') & (dataset.Age.isna())] = avgMsAge

    dataset.Age[(dataset.Title == 'Mrs') & (dataset.Age.isna())] = avgMrsAge



# Replace single missing fare in test data based on Pclass

avgFareForClass = np.mean(combined.Fare[~(combined.Fare.isna()) & (combined.Pclass == 3)])

testFull.Fare[(testFull.Fare.isna())] = avgFareForClass



# Low cardinality categorical variables replaced by one hot encoding

lowCardCols = ['Sex', 'Embarked','Title']

OHencoder = OneHotEncoder(handle_unknown='ignore',sparse = False)

OHcols = pd.DataFrame(OHencoder.fit_transform(trainFull[lowCardCols]))

testOHcols = pd.DataFrame(OHencoder.transform(testFull[lowCardCols]))

OHcols.index = trainFull.index

testOHcols.index = testFull.index

trainFull.drop(lowCardCols, axis=1, inplace=True)

testFull.drop(lowCardCols, axis = 1, inplace= True)

trainFull = pd.concat([trainFull,OHcols],axis=1)

testFull = pd.concat([testFull, testOHcols], axis=1)



# Example data from transformed data set

trainFull.sample(5)
# Split target vector from training data

y = trainFull.Survived

trainFull.drop(['Survived','PassengerId'],axis=1,inplace=True)

indices = testFull['PassengerId']

testFull.drop(['PassengerId'], axis=1, inplace=True)



# Tune hyper parameters for a random forest classifier

bestNestimators = 80

bestMinSplit = 2

bestMinLeaf = 1

bestAccuracy = 0.8

for n in range(10,120,10):

    for m in range(2,15,1):

        for l in range(2,5,1):

        # Use 5-fold cross-validation to approximate accuracy with a random forest classifier

            scores = -1 * cross_val_score(RandomForestClassifier(n_estimators=n,min_samples_split=m, min_samples_leaf=l),trainFull, y,

                                              cv=5,

                                              scoring='neg_mean_absolute_error')

            accuracy = 1-np.mean(scores)

            if accuracy > bestAccuracy:

                bestNestimators = n

                bestMinSplit = m

                bestMinLeaf = l

                bestAccuracy = accuracy

                print("With estimators:",n,"Min split:",m,"Min leaf:",l,"Approximate accuracy:",1-np.mean(scores))
# Train a random forest classifier on whole dataset

randomforest = RandomForestClassifier(n_estimators = bestNestimators, min_samples_split = bestMinSplit, min_samples_leaf = bestMinLeaf)

randomforest.fit(trainFull,y)

testPred = randomforest.predict(testFull)



output = pd.DataFrame({ 'PassengerId' : indices, 'Survived': testPred })

output.to_csv('submission.csv', index=False)