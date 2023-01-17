# Class Imports

import re

import math

import numpy as np  # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Utility Functions

sns_percent = lambda x: sum(x)/len(x)*100
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
# Data mappings

for dataset in [test, train]:

    dataset['Gender']     = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    dataset['CabinClass'] = dataset['Cabin'].astype(str).map(lambda x: re.sub('^(\w)?.*', '\\1', x) if x != "nan" else None )

    dataset['LogFare']    = dataset['Fare'].astype(float).map(lambda x: math.log(x) if x else None)

    dataset['Title']      = dataset['Name'].astype(str).map(lambda x: re.findall('(\w+)\.', x)[0])

train.head() 



#cabin_classes = dataset['Cabin'].astype(str).map(lambda x: re.sub('^(\w)?.*', '\\1', x) if x != "nan" else None ).unique()    

#test.groupby('Title')['Title'].count()   
output_random = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived"   : np.random.randint(0,2, size=len(test)) # random number 0 or 1

})

output_random.to_csv('random.csv', index=False); # score 0.51196 (6993/7071)
train["Survived"].map({0: "dead", 1: "alive"}).value_counts()/len(train)
output_everybody_dead = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived"   : 0

})

output_everybody_dead.to_csv('everybody_dead.csv', index=False) # score 0.62679 (6884/7071)
train["Sex"].value_counts()/len(train)
survivors  = train[train['Survived'] == 1]

casualties = train[train['Survived'] == 0]

pd.DataFrame({

    "survivors":  survivors["Sex"].value_counts()/len(train),

    "casualties": casualties["Sex"].value_counts()/len(train),

})
output_everybody_dead = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived"   : test["Gender"]

})

output_everybody_dead.to_csv('only_women_survive.csv', index=False) # score 0.76555 (5384/7071)
train_with_age = train[ ~np.isnan(train["Age"]) ]

survivalpc_by_age = train_with_age.groupby(["Sex","Age"], as_index = False)["Survived"].mean()

#sns.boxplot("Age", "Survived", survivalpc_by_age)



for gender in ["male", "female"]:

    plt.figure()

    sns.lmplot(data=survivalpc_by_age[survivalpc_by_age["Sex"]==gender], x="Age", y="Survived", order=4)

    plt.title("%s survival by age" % gender)

    plt.xlim(0, 80)

    plt.ylim(0, 1)
output_women_and_children_first = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived"   : ((test["Sex"] == "female") | ((test["Age"] <= 12) | (test["Age"] >= 80))).astype(int)

})

output_women_and_children_first.to_csv('women_and_children_first.csv', index=False) # score 0.77033 (4523/7071)
output_women_and_toddlers_first = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived"   : ((test["Sex"] == "female") | (test["Age"] <= 6)).astype(int)

})

output_women_and_toddlers_first.to_csv('women_and_toddlers_first.csv', index=False) # score 0.75598 (4523/7071)

# Your submission scored 0.75598, which is not an improvement of your best score. Keep trying!
train
train_dummies = pd.get_dummies(train, columns=["Title","CabinClass","Embarked"]).corr()

sns.heatmap(train_dummies.corr(), square=True, annot=False)
train_dummies.corr()['Survived']