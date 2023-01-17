# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Load packages



import pandas as pd

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

import numpy as np
#Import CSVs

titanicTrain_X = pd.read_csv('../input/train.csv')

titanicTest_X = pd.read_csv('../input/test.csv')



#Split Train into X and y

titanicTrain_y = titanicTrain_X['Survived']

titanicTrain_X = titanicTrain_X.drop('Survived', axis=1)



#Name the dataframes

titanicTrain_y.name = 'titanicTrain_y'

titanicTrain_X.name = 'titanicTrain_X'

titanicTest_X.name = 'titanicTest_X'



#Place datasets into lists

observationSets = [titanicTrain_X, titanicTest_X]



#general statistics on both data sets

for dataset in observationSets:

    print( dataset.describe() ) # results of the print indicate AGE will need NAs cleaned

   
#Sort Passengers into age groups



for dataset in observationSets:

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

    ageBins = [0, 10, 20, 60, 125]

    ageBinValues = [0, 1, 2, 3]

    dataset['Age'] = pd.cut(dataset['Age'], ageBins, labels = ageBinValues)

    
#Change female and male to binary. Female = 0, male = 1



for dataset in observationSets:

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1})

    
#Fix the NA values for Parent and Siblings counts by setting NAs to 0



for dataset in observationSets:

    dataset['Parch'] = dataset['Parch'] .fillna(0)

    dataset['SibSp'] = dataset['SibSp'] .fillna(0)
#Transform Embarked to numeric values. Fill Na embarked to 0. 



for dataset in observationSets:

    dataset['Embarked'] = dataset['Embarked'].fillna(0)

    dataset['Embarked'] = dataset['Embarked'].map({0:0, 'S':1, 'Q':2, 'C':3}).astype(int)
#Remove less useful columns

#  Name: Could potentially extract Mr. Mrs. and Miss. Perhaps a Mrs. is more likely 

#        to survive since they could have kids

#  Ticket: Not in a standard easily parsable format

#  Fare: This data is alread fairly well represented in Pclass

#  Cabin: Contains a lot of nulls and im nut sure what i should do with them without 

#         altering the learning

for dataset in observationSets:

    dataset.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis = 1, inplace = True)
#Titanic Train Data

for dataset in observationSets:

    print(dataset.name)

    print(dataset.head())
#Check for Null Values



print( pd.isnull(titanicTest_X).sum() > 0 )



print( pd.isnull(titanicTrain_X).sum() > 0 )

titanicTest_X.info()

print("----------------------------")

titanicTrain_X.info()
#Starting with Linear Support Vector Classification First since i am performing classification on 

#less than 100k observations



#print(titanicTrain_X.shape)

#print(titanicTrain_y.shape)



linearClassifier = svm.LinearSVC()

linearClassifier = linearClassifier.fit(titanicTrain_X, titanicTrain_y)



linearClassifier.score(titanicTrain_X, titanicTrain_y) 

prediction = linearClassifier.predict(titanicTest_X) 



titanicTest = pd.read_csv('../input/test.csv')

Submission = pd.DataFrame({

        "PassengerId": titanicTest["PassengerId"],

        "Survived": prediction

    })

Submission.to_csv("LinearClassifierSubmissionPrediction.csv", index=False)
# Next i'm going to try out KNearestNeighbors



neighbors = [ 1, 3, 5, 7]



for k in neighbors:

    kNeibrs = KNeighborsClassifier(n_neighbors = k)

    kNeibrs.fit(titanicTrain_X, titanicTrain_y)

    print( kNeibrs.score(titanicTrain_X, titanicTrain_y))

    

#5 nearest neighbors seems to fit the training data the best



kNeigbrs = KNeighborsClassifier(n_neighbors = 5)

prediction = kNeibrs.predict(titanicTest_X)



titanicTest = pd.read_csv('../input/test.csv')

Submission = pd.DataFrame({

        "PassengerId": titanicTest["PassengerId"],

        "Survived": prediction

    })

Submission.to_csv("KNeighborsSubmissionPrediction.csv", index=False)
