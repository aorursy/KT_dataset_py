import numpy as np

import pandas as pd
def make_split_func(tpl):

    def f(x):

        return (1 if x == tpl else 0)

    return f



def split_row(ds, old_row, new_row, tpl):

    ds[new_row] = ds[old_row].map(make_split_func(tpl))



def harmonize_data1(titanic):

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic["Age"].median()

    

    split_row(titanic, "Sex", "IsFemale", "female")

    split_row(titanic, "Sex", "IsMale", "male")

    

    split_row(titanic, "Pclass", "Is1Class", 1)

    split_row(titanic, "Pclass", "Is2Class", 2)

    split_row(titanic, "Pclass", "Is3Class", 3)



    split_row(titanic, "Embarked", "FromS", "S")

    split_row(titanic, "Embarked", "FromC", "C")

    split_row(titanic, "Embarked", "FromQ", "Q")



    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())



    return titanic[ ["IsMale", "IsFemale", "Is1Class", "Is2Class", "Is3Class", "Age", "SibSp", "Parch", "Fare", "FromS", "FromC", "FromQ"] ]



def harmonize_data2(titanic):

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic["Age"].median()



    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    

    titanic["Embarked"] = titanic["Embarked"].fillna("S")



    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2



    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())



    return titanic[ ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] ]
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



train = harmonize_date1(train)

test = harmonize_date1(test)

def squeeze(sq, dataset):

    return sq.transform(dataset)
from sklearn.decomposition import PCA



pca = PCA()

pca.fit(harmonize_data1(train), train["Survived"])