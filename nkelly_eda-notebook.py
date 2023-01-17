import pandas as pd

import numpy as np



activities_test = pd.read_csv("../input/act_test.csv")

activities_train = pd.read_csv("../input/act_train.csv")

people = pd.read_csv("../input/people.csv")



def showTop5(df):

    for col in df.columns:

        print( "======",col,"======")

        print( df[col].value_counts().head(5))

        print()



def getColUniques(df):

    colDictionary = {}

    for col in df.columns:

        colDictionary[col] = set(df[col].unique())

    return colDictionary
showTop5(activities_test)

showTop5(activities_train)



testUniques = getColUniques(activities_test)

trainUniques = getColUniques(activities_train)
set(trainUniques.keys()).symmetric_difference(testUniques.keys())