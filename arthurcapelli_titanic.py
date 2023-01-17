import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
train= pd.read_csv("../input/titanic/train.csv")



titanic = pd.read_csv("../input/titanic/test.csv")
train.info()
train.head()
titanic.info()
titanic.drop("Cabin", axis = 1, inplace= True)

train.drop("Cabin", axis = 1, inplace = True)
womentrain = train.loc[train.Sex=="female"]["Survived"]

rate_women = sum(womentrain)/len(womentrain)

rate_women
# women are more likely to survive
mentrain = train.loc[train.Sex=="male"]["Survived"]

rate_men = sum(mentrain)/len(mentrain)

rate_men
# men really doesn't live that long
train.Sex.value_counts()
train.Parch.value_counts()
parchtrain = train.loc[train.Parch==0]["Survived"]
parchtrain.value_counts()
#and that's why is import to never be alone.
train.Pclass.value_counts()
Pclasstrain3 = train.loc[train.Pclass==3]["Survived"]

Pclasstrain3.value_counts()
Pclasstrain1 = train.loc[train.Pclass==1]["Survived"]

Pclasstrain1.value_counts()
Pclasstrain2 = train.loc[train.Pclass==2]["Survived"]

Pclasstrain2.value_counts()
# Well...looks like people really make some effort to you when you have money... what surprise...
group1 = train.groupby(['Pclass','Sex','Survived'])['Age'].mean()

print(group1)
#beyoung
from sklearn.ensemble import RandomForestClassifier
combine = [train, titanic]
group1 = train.groupby(['Pclass','Sex','Survived'])['Age'].mean()

print(group1)
#beyoung
for datadf in combine:    

    #complete missing age with median

    datadf['Age'].fillna(datadf['Age'].median(), inplace = True)
titanic.info()
y = train["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch","Age"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(titanic[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': titanic.PassengerId, 'Survived': predictions})

output.Survived.value_counts()
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")