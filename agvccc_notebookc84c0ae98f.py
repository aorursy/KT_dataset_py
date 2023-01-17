import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
def adjust_data(titanic):

    

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    

    titanic["Embarked"] = titanic["Embarked"].fillna("S")



    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2



    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())



    return titanic
train_data = adjust_data(train)

test_data = adjust_data(test)