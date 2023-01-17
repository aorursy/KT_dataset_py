# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import re



from statistics import mode
df = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv("/kaggle/input/titanic/test.csv")

total = pd.concat([df, test]).reset_index(drop=True)
df.head()
df.isnull().sum()
sns.countplot(x="Survived",data=df)
sns.countplot(x="Survived", hue="Sex", data=df)
sns.countplot(x="Survived", hue="Pclass", data=df)
sns.countplot(x="Survived", hue="Embarked", data=df)
sns.countplot(x="Survived", hue="SibSp", data=df)
sns.countplot(x="Survived", hue="Parch", data=df)
df.describe()
df.loc[:, "Age"].hist(bins=80)
sns.countplot(df.loc[:, "SibSp"])
sns.countplot(df.loc[:, "Parch"])
df.loc[:, "Ticket"].value_counts().head()
df.loc[:, "Fare"].hist(bins=40)
df.loc[:, "Cabin"].value_counts().head()
sns.countplot(df.loc[:, "Embarked"])
sns.heatmap(df.corr(), annot=True)
df.loc[df.loc[:, "Age"].isnull(), "Age"] = df.groupby("Pclass")["Age"].transform('median')



df.loc[:, "Age"].isnull().sum()
# Sex

df.loc[df.loc[:, "Sex"] == "male", "Sex"] = 0

df.loc[df.loc[:, "Sex"] == "female", "Sex"] = 1
df.loc[df.loc[:, "Embarked"] == "S",  "Embarked"] = 0

df.loc[df.loc[:, "Embarked"] == "C",  "Embarked"] = 1

df.loc[df.loc[:, "Embarked"] == "Q",  "Embarked"] = 2
total.loc[:, "Embarked"] = total.loc[:, "Embarked"].fillna(mode(total.loc[:, "Embarked"]))
total.loc[total.loc[:, "Sex"] == "male", "Sex"] = 0

total.loc[total.loc[:, "Sex"] == "female", "Sex"] = 1



total.loc[:, "Sex"] = total.loc[:, "Sex"].astype(int)
total.loc[total.loc[:, "Embarked"] == "S",  "Embarked"] = 0

total.loc[total.loc[:, "Embarked"] == "C",  "Embarked"] = 1

total.loc[total.loc[:, "Embarked"] == "Q",  "Embarked"] = 2



total.loc[:, "Embarked"] = total.loc[:, "Embarked"].astype(int)
sns.heatmap(total.corr(), annot=True)
total.loc[total.loc[:, "Age"].isnull(), "Age"] = total.groupby("Pclass")["Age"].transform("median")
total.loc[total.loc[:, "Fare"].isnull(), "Fare"] = total.groupby("Pclass")["Fare"].transform("median")
total.loc[:, "Cabin"] = total.loc[:, "Cabin"].fillna("U")
total.loc[:, "Cabin"] = total.loc[:, "Cabin"].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
sorted(total.loc[:, "Cabin"].unique().tolist())
cabin_dict = {

    "A": 0,

    "B": 1, 

    "C": 2,

    "D": 3,

    "E": 4,

    "F": 5,

    "G": 6,

    "T": 7,

    "U": 8

}



total.loc[:, "Cabin"] = total.loc[:, "Cabin"].map(cabin_dict)
total.loc[:, "Cabin"].value_counts()
total.loc[:, "Name"] = total.loc[:, "Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)



total.loc[:, "Name"].value_counts()
total.loc[:, "Name"] = total.loc[:, "Name"].replace(["Rev", "Dr", "Col", "Major", "Mlle", "Ms", "Mme", "Lady", "Jonkheer", "Dona", "Capt", "Countess",

                              "Don", "Sir"], "Others")



total.loc[:, "Name"].value_counts()
name_dict = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Others": 4}

total.loc[:, "Name"] = total.loc[:, "Name"].map(name_dict)



total.loc[:, "Name"].value_counts()
total.loc[:, "FamilySize"] = total.loc[:, "SibSp"] + total.loc[:, "Parch"] + 1
train = total.loc[total.loc[:, "Survived"].notnull(), :]

test = total.loc[total.loc[:, "Survived"].isnull(), :]
features = ["Pclass", "Name", "Sex", "Age", "Fare", "Cabin", "Embarked", "FamilySize"]

target = "Survived"
import xgboost as xgb



from sklearn.metrics import precision_score, recall_score, accuracy_score

from sklearn.model_selection import train_test_split
train_train, train_test = train_test_split(train, test_size=0.2, random_state=4)
D_train = xgb.DMatrix(train_train.loc[:, features], train_train.loc[:, target])

D_test = xgb.DMatrix(train_test.loc[:, features], train_test.loc[:, target])
param = { 

    "objective": "binary:logistic",  

}



model = xgb.train(param, D_train)
predictions = model.predict(D_test)

predictions = np.rint(predictions)
test_accuracy = accuracy_score(train_test.loc[:, "Survived"], predictions)

test_precision = precision_score(train_test.loc[:, "Survived"], predictions)

test_recall = recall_score(train_test.loc[:, "Survived"], predictions)



print(f"Accuracy in the test set: {'%.2f'%(test_accuracy * 100)}%")

print(f"Precision in the test set: {'%.2f'%(test_precision * 100)}%")

print(f"Recall in the test set: {'%.2f'%(test_recall * 100)}%")
D_train = xgb.DMatrix(train.loc[:, features], train.loc[:, target])

D_test = xgb.DMatrix(test.loc[:, features], test.loc[:, target])
param = { 

    "objective": "binary:logistic",  

}



model = xgb.train(param, D_train)
predictions = model.predict(D_test)

predictions = np.rint(predictions)
test.loc[:, "Survived"] = predictions
test.head()
test.loc[:, "Survived"] = test.loc[:, "Survived"].astype(int)
submission = test.loc[:, ["PassengerId", "Survived"]]

submission.to_csv("/kaggle/working/submission.csv", index=False)