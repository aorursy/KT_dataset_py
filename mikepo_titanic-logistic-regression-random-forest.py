import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.info()
train.Age[train.Age.isnull()] = train.Age.median()
train = train.drop(['Cabin'],axis=1)
MaxPassEmbarked = train.groupby('Embarked').count()['PassengerId']
train.Embarked[train.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
train.info()
train = train.drop(['Name','Ticket'],axis=1)
train.head()
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
dicts = {}
lbl.fit(train.Sex.drop_duplicates()) 
dicts['Sex'] = list(lbl.classes_)
train.Sex = lbl.transform(train.Sex)
lbl.fit(train.Embarked.drop_duplicates())
dicts['Embarked'] = list(lbl.classes_)
train.Embarked = lbl.transform(train.Embarked)
print("Survivors: depending on the sex of the passenger and socio-economic status //SES")
print(train.groupby(["Sex", "Pclass"])["Survived"].value_counts(normalize=True))
print("Graphical histograms from a subset of data")
print("Survivors: depending on the sex of the passenger")
sns.set(style="darkgrid")

g = sns.FacetGrid(train, row="Sex", col="Survived", margin_titles=True)
bins = np.linspace(0, 891, 2)
g.map(plt.hist, "PassengerId", color="steelblue", bins=bins, lw=0)
test = pd.read_csv('../input/test.csv')
test.Age[test.Age.isnull()] = test.Age.mean()
test.Fare[test.Fare.isnull()] = test.Fare.median()
MaxPassEmbarked = test.groupby('Embarked').count()['PassengerId']
test.Embarked[test.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
result = DataFrame(test.PassengerId)
test = test.drop(['Name','Ticket','Cabin'],axis=1)

lbl.fit(dicts['Sex'])
test.Sex = lbl.transform(test.Sex)

lbl.fit(dicts['Embarked'])
test.Embarked = lbl.transform(test.Embarked)
test.head()
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
target = train.Survived
trainTRN, testTRN, trainTRG, testTRG = train_test_split(train, target, test_size=0.30)
float(len(testTRG))/len(train)
lr = LogisticRegression(random_state=22)
lr.fit(trainTRN, trainTRG)
lr_valid_pred = lr.predict(testTRN)
lr_accuracy = accuracy_score(testTRG, lr_valid_pred)
print(" lr_accuracy = " + str(lr_accuracy))
rf = RandomForestClassifier()

rf.fit(trainTRN, trainTRG)
rf_valid_pred = rf.predict(testTRN)

rf_accuracy = accuracy_score(testTRG, rf_valid_pred)
print(" rf_accuracy = " + str(rf_accuracy))