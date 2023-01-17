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
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import seaborn as sns

train=pd.read_csv("../input/train.csv")
train.head()
train.info()
plt.figure(figsize=(10,6))

sns.set_style("whitegrid")

sns.countplot(x="Survived",data=train, palette="pastel", saturation=0.8)
plt.figure(figsize=(10,6))

sns.countplot(x="Survived",data=train, hue="Sex", palette="pastel", saturation=0.8)
plt.figure(figsize=(10,6))

sns.countplot(x="Survived",data=train, hue="Pclass", palette="pastel", saturation=0.8)
plt.figure(figsize=(10,6))

sns.countplot(x="SibSp",data=train, palette="pastel", saturation=0.8)
plt.figure(figsize=(10,6))

sns.distplot(train["Fare"], kde=False, bins=30, hist_kws=dict(edgecolor="black", linewidth=1))

plt.xlim(0,)
plt.figure(figsize=(10,6))

sns.distplot(train["Age"].dropna(), kde=False, bins=30, hist_kws=dict(edgecolor="black", linewidth=1))

plt.xlim(0,)
plt.figure(figsize=(10,6))

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
plt.figure(figsize=(10,6))

sns.boxplot(x="Pclass",y="Age",data=train, palette="GnBu_r", saturation = 1)
def complete_age(col):

    age=col[0]

    pclass=col[1]

    if pd.isnull(age):

        if pclass==1:

            return 37

        elif pclass==2:

            return 29

        else:

            return 24

    else:

        return age
train["Age"]=train[["Age", "Pclass"]].apply(complete_age, axis=1)
plt.figure(figsize=(10,6))

sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap="viridis")
train.drop("Cabin", axis=1, inplace=True)
plt.figure(figsize=(10,6))

sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap="viridis")
train.dropna(inplace=True)
sex = pd.get_dummies(train["Sex"], drop_first=True)

embark = pd.get_dummies(train["Embarked"], drop_first=True)
train.drop(["Name", "Sex", "Embarked", "Ticket"], axis=1, inplace=True)

train = pd.concat([train, sex, embark], axis=1)
train.head()
from sklearn.model_selection import train_test_split
X = train.drop("Survived", axis=1)

y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))



print(confusion_matrix(y_test, predictions))
test = pd.read_csv("../input/test.csv")
test.head()
plt.figure(figsize=(10,6))

sns.heatmap(test.isnull(), cbar=False, yticklabels=False, cmap="viridis")
def complete_age(col):

    age=col[0]

    pclass=col[1]

    if pd.isnull(age):

        if pclass==1:

            return 37

        elif pclass==2:

            return 29

        else:

            return 24

    else:

        return age
test["Age"]=test[["Age", "Pclass"]].apply(complete_age, axis=1)
test.drop("Cabin", axis=1, inplace=True)
test[test["Age"].isnull()]
test[test["Fare"].isnull()]
plt.figure(figsize=(10,6))

plt.ylim(0,20)

sns.boxplot(x="Pclass",y="Fare",data=test, palette="GnBu_r", saturation = 1)
def complete_fare(col):

    fare=col[0]

    pclass=col[1]

    if pd.isnull(fare):

        if pclass==1:

            return 60

        elif pclass==2:

            return 15.5

        else:

            return 8

    else:

        return fare
test["Fare"]=test[["Fare", "Pclass"]].apply(complete_fare, axis=1)
plt.figure(figsize=(10,6))

sns.heatmap(test.isnull(), cbar=False, yticklabels=False, cmap="viridis")
sex = pd.get_dummies(test["Sex"], drop_first=True)

embark = pd.get_dummies(test["Embarked"], drop_first=True)
test.drop(["Name", "Sex", "Embarked", "Ticket"], axis=1, inplace=True)

test = pd.concat([test, sex, embark], axis=1)
test.head()

id = pd.DataFrame(test["PassengerId"])

id.head()

id.info()
surv = pd.DataFrame(logmodel.predict(test), columns= ["Survived"])

surv.head()

surv.info()
answer = id.join(surv,)

answer
answer.to_csv("Answer.csv", index=False)
submission = pd.read_csv("../input/gender_submission.csv")
submission.head()
compar = pd.merge(answer, submission, on=["PassengerId"])

compar
compar["Diff"] = compar["Survived_y"] - compar["Survived_x"]
compar
compar["Diff"].value_counts()
type1 = int(compar["Diff"].value_counts()[1])
type2 = int(compar["Diff"].value_counts()[-1])
efficiency = (418-type1-type2)/418

print(efficiency)