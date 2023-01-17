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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
# ID test data in order to split data after transforming

train["Test"] = 0

test["Test"] = 1
# Fill null values for age in train set with median for train

train['Age'].fillna(train['Age'].median(),inplace=True)
# 0 Nulls

train.isnull().sum()
# Fill null values for point of departure with the most common port, 'S'

train['Embarked'].fillna('S',inplace=True)
# 0 Nulls

test.isnull().sum()
# Decided not to use Cabin, redundant variable

test.drop(labels="Cabin",axis=1,inplace=True)

train.drop(labels="Cabin",axis=1,inplace=True)
# Fill test age nulls set with median value

test.fillna(test['Age'].median(),inplace=True)
# Separate targets from train set

# New DataFrame 'survived_train' is PassengerId and Survived (1 or 0)

survived_train = train.drop(labels=["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Ticket","Fare","Embarked","Test"],axis=1)

train.drop(labels='Survived',axis=1,inplace=True)
# Train set contains no nulls

train.isnull().sum() == 0
# Test set contains no nulls

# test.isnull().sum()
# Joins train and test for further data transformation

join = train.append(test)
join.shape
join
# Function to extract prefixes of passengers' names in Name column

def get_titles():



    global join



    join['Title'] = join['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    Title_Dictionary = {

                        "Capt":       "Officer",

                        "Col":        "Officer",

                        "Major":      "Officer",

                        "Jonkheer":   "Royalty",

                        "Don":        "Royalty",

                        "Sir" :       "Royalty",

                        "Dr":         "Officer",

                        "Rev":        "Officer",

                        "the Countess":"Royalty",

                        "Dona":       "Royalty",

                        "Mme":        "Mrs",

                        "Mlle":       "Miss",

                        "Ms":         "Mrs",

                        "Mr" :        "Mr",

                        "Mrs" :       "Mrs",

                        "Miss" :      "Miss",

                        "Master" :    "Master",

                        "Lady" :      "Royalty"



                        }



    join['Title'] = join.Title.map(Title_Dictionary)
get_titles()

join['Title']
# List of nominal/ordinal variables to be transformed

dummies = ["Pclass", "Sex", "Embarked","Title"]
# Gets dummies from previous list

join = pd.get_dummies(data=join,columns=dummies,drop_first=False)
join.head()
join.drop(labels=["Name","Ticket"],axis=1,inplace=True)
join["Test"].value_counts()
join.sort_values(by="Test",axis=0,inplace=True)
train = join.iloc[0:891]

test = join.iloc[891::]
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
train.shape
test