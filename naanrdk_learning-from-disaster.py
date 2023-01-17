import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.tree import DecisionTreeClassifier # machine learning algorithm

from sklearn import cross_validation as cv # preprocessing technique

from sklearn.cross_validation import KFold



# Input files 

train =pd.read_csv("../input/train.csv", dtype={"Age": np.float64})

test =pd.read_csv("../input/test.csv", dtype={"Age": np.float64})



# Cleaning the dataset

def clean_data(passenger):

    

    # Filling the missing values with average value

    passenger["Age"] = passenger["Age"].fillna(passenger["Age"].mean())

    passenger["Fare"] = passenger["Fare"].fillna(passenger["Fare"].mean())

    

    passenger["Embarked"] = passenger["Embarked"].fillna("S")

    

    # Preprocessing into binary form for gender

    passenger.loc[passenger["Sex"] == "male", "Sex"] = 1

    passenger.loc[passenger["Sex"] == "female", "Sex"] = 0

    

    # Preprocessing multiclass form for Embarkation

    passenger.loc[passenger["Embarked"] == "S", "Embarked"] = 0

    passenger.loc[passenger["Embarked"] == "C", "Embarked"] = 1

    passenger.loc[passenger["Embarked"] == "Q", "Embarked"] = 2

    

    return passenger





# Cleaned Data assigned to new variables    

train_data = clean_data(train)

test_data  = clean_data(test)



# Merging feature variable

train_data["totalSize"] = train_data["SibSp"]+train_data["Parch"]+1

test_data["totalSize"] = test_data["SibSp"]+test_data["Parch"]+1



# Predictors

predictors = ["Sex", "Age", "Pclass", "totalSize"]



# Fit DecissionTree model

maxScore = 0

depthSize = 0



for n in range(1,100):

    

    dt_scr = 0.

    dt = DecisionTreeClassifier(max_depth=n)

    for train, test in KFold(len(train_data), n_folds=10, shuffle=True):

        dt.fit(train_data[predictors], train_data["Survived"])

        dt_scr += dt.score(train_data[predictors], train_data["Survived"])/10

    

    if dt_scr > maxScore:

        maxScore = dt_scr

        depthSize = n



print(depthSize, maxScore)

dt = DecisionTreeClassifier(max_depth=depthSize)



# Submission function     

def submitResult(dt, train, test, predictors, filename):



    dt.fit(train[predictors], train["Survived"])

    predictions = dt.predict(test[predictors])



    submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })

    

    submission.to_csv(filename, index=False)

    

#Submit result

submitResult(dt, train_data, test_data, predictors, "decissionTreesurvivors.csv")