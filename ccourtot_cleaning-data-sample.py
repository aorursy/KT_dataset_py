import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

train = train.drop("Name",1)
test = test.drop("Name",1)
train = train.drop("Ticket",1)
test = test.drop("Ticket",1)
train = train.drop("Cabin",1)
test = test.drop("Cabin",1)

trainS = train.Survived
train = train.drop("Survived",1)

train = train.drop("PassengerId",1)
testId = test.PassengerId
test = test.drop("PassengerId",1)

train = train.drop("Embarked",1)
test = test.drop("Embarked",1)
test
