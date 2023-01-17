import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import math

import random

import numpy

import copy 



import os

print(os.listdir("../input"))



pd.options.mode.chained_assignment = None # need to modify 
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
def kesson_table(df):

    null_val = df.isnull().sum()

    percent = 100*df.isnull().sum()/len(df)

    kesson_table = pd.concat([null_val, percent], axis=1)

    kesson_table_ren_colmuns = kesson_table.rename(columns = {0:"Loss ratio", 1: "%"})

    return kesson_table_ren_colmuns
train["Age"] = train["Age"].fillna(train["Age"].median())

kesson_table(train)
train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1



train.head(10)
test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Fare"] = test["Fare"].fillna(test["Fare"].median())



test.head(10)
class Individual:

    def __init__(self, t, s):

        self.thresholds = t

        self.scores = s   
target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
# Here is the code of the function (1+1)-ES which train thresholds.

# Use only 4 elements, Pclass, Sex, Age, Fare.

# Make 4 thresholds for each element.

# If the condition is satisfied, add 25 to the possibility.

# If the possibility is over 49, the passenger would be defined as survived.

# The definition is correct and add 1 to scores.

# In each generation, one threshold is modified. If the modified individual had good scores, we adopt the individual.



def ES():

    myIndividual = Individual([0, 25, 20, 50], 0)

    

    for features, survived in zip(features_one, target):

            possibility = 0

            survival = 0

            if features[0] == myIndividual.thresholds[0]:

                possibility += 25

            if features[1] == 0:

                possibility += myIndividual.thresholds[1]

            else:

                possibility += 25-myIndividual.thresholds[1]

                

            if features[2] < myIndividual.thresholds[2]:

                possibility += 25

            if features[3] > myIndividual.thresholds[3]:

                possibility += 25

            if possibility > 49:

                survival = 1

                

            if survival == survived:

                myIndividual.scores += 1

    

    generation = 0

    while generation < 10000:

        nextIndividual = copy.deepcopy(myIndividual)

        nextIndividual.scores = 0

        if generation%4 == 0:

            nextIndividual.thresholds[0] = random.randint(1, 3)

        elif generation%4 == 1:

            nextIndividual.thresholds[1] = random.random()*35

        elif generation%4 == 2:

            nextIndividual.thresholds[2] = random.randint(0, 100)

        else:

            nextIndividual.thresholds[3] = random.randint(0, 200)

        for features, survived in zip(features_one, target):

            possibility = 0

            survival = 0

            if features[0] == nextIndividual.thresholds[0]:

                possibility += 25

            if features[1] == 0:

                possibility += nextIndividual.thresholds[1]

            else:

                possibility += 25-nextIndividual.thresholds[1]

                

            if features[2] < nextIndividual.thresholds[2]:

                possibility += 25

            if features[3] > nextIndividual.thresholds[3]:

                possibility += 25

            

            if possibility > 49:

                survival = 1

                

            if survival == survived:

                nextIndividual.scores += 1

        if myIndividual.scores < nextIndividual.scores:

            print("pre score", myIndividual.scores)

            print("new score" ,nextIndividual.scores)

            print("pre thresholds" ,myIndividual.thresholds)

            print("new thresholds" ,nextIndividual.thresholds)

            myIndividual = nextIndividual

        generation+=1

    return myIndividual

# Generate solution using the trained thresholds.

myIndividual = ES()

solution = np.array([], dtype="int64")

for features in test_features:

    survival = 0

    possibility = 0

    if features[0] == myIndividual.thresholds[0]:

                possibility += 25

    if features[1] == 0:

        possibility += myIndividual.thresholds[1]

    else:

        possibility += 25-myIndividual.thresholds[1]

        

    if features[2] < myIndividual.thresholds[2]:

                possibility += 25

    if features[3] > myIndividual.thresholds[3]:

                possibility += 25

    if possibility > 49:

        survival = 1

    solution = np.append(solution, int(survival))
# Make output csv.

PassengerId = np.array(test["PassengerId"]).astype(int)



my_solution = pd.DataFrame(solution, PassengerId, columns = ["Survived"])



my_solution.to_csv("my_es.csv", index_label = ["PassengerId"])