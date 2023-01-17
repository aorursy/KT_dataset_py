# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train["Set"] = 1 # indicate that the data point is from training set

print(train.info())

train.sample(10)
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test["Set"] = 0 # indicate that the data point is from test set

print(test.info())

test.sample(10)
data = pd.concat([train, test], axis=0, sort=False, ignore_index=True)

print(data.info())

data.describe()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,5))



plt.subplot2grid((1,2),(0,0))

fare = data.Fare

x = range(0,600,50)

plt.hist(data.loc[:, "Fare"], bins=x, rwidth=0.9)

plt.xticks(x)

plt.title("Frequency Plot of Fare")

plt.xlabel("Fare")



data.loc[pd.isna(data["Fare"]), "Fare"] = data["Fare"].median()



plt.subplot2grid((1,2),(0,1))

data.Embarked.value_counts().plot(kind="bar")



data.loc[pd.isnull(data["Embarked"]), "Embarked"] = data["Embarked"].mode()[0]



data.isnull().sum()
fig = plt.figure(figsize=(10,5))



plt.subplot2grid((1,2),(0,0))

age = data.loc[(pd.isna(data.Age)==False) & (pd.isna(data.Survived)==False), ["Age", "Survived"]]

x = range(0, 80, 5)

plt.hist(x=[age.loc[age.Survived==0,"Age"], age.loc[age.Survived==1,"Age"]], bins=x, rwidth=0.9, label=["not survived", "survived"])

plt.xticks(x)

plt.title("Age Distribution in Training Set")

plt.legend()



plt.subplot2grid((1,2),(0,1))

age_train = data.loc[(pd.isna(data.Survived)==False) & (pd.isna(data.Age)==False),"Age"]

age_train_scale = (age_train - age_train.min())/(age_train.max() - age_train.min())

age_test = data.loc[(pd.isna(data.Survived)==True) & (pd.isna(data.Age)==False),"Age"]

age_test_scale = (age_test - age_test.min())/(age_test.max() - age_test.min())

plt.hist(x=[age_train_scale, age_test_scale], rwidth=0.9, label=["train", "test"])

plt.title("Scaled Age Distribution in Training Set and Test Set")

plt.legend()



data.loc[(pd.isna(data.Survived)==False) & (pd.isna(data.Age)==True),"Age"] = data.loc[pd.isna(data.Survived)==False,"Age"].median()

data.loc[(pd.isna(data.Survived)==True) & (pd.isna(data.Age)==True),"Age"] = data.loc[pd.isna(data.Survived)==True,"Age"].median()



data.isnull().sum()
data = data.join(pd.get_dummies(data["Pclass"], prefix = "Pclass"))

data = data.join(pd.get_dummies(data["Sex"], prefix = "Sex"))

data = data.join(pd.get_dummies(data["Embarked"], prefix = "Embarked"))
data["Family"] = data["SibSp"] + data["Parch"] + 1
data["Prefix"] = data["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

print(data["Prefix"].value_counts())

data.loc[(data["Prefix"]!="Mr") & (data["Prefix"]!="Miss") & (data["Prefix"]!="Mrs") & (data["Prefix"]!="Master"), "Prefix"] = "Others"

print(data["Prefix"].value_counts())

data = data.join(pd.get_dummies(data["Prefix"], prefix = "Prefix"))

data.sample(10)
dataexplore = data.loc[pd.isna(data["Survived"])==False,:]

print(dataexplore.groupby("Sex")["Survived"].mean())

print(dataexplore.groupby("Pclass")["Survived"].mean())

print(dataexplore.groupby("Embarked")["Survived"].mean())

print(dataexplore.groupby("Prefix")["Survived"].mean())
fig = plt.figure(figsize=(10,10))



plt.subplot2grid((2,2),(0,0))

x=range(0,80,5)

plt.hist(x=[dataexplore.loc[dataexplore.Survived==0,"Age"], dataexplore.loc[dataexplore.Survived==1,"Age"]], bins=x, rwidth=0.9, label=["not survived", "survived"])

plt.xticks(x)

plt.legend()

plt.xlabel("Age")

plt.title("Survival Distribution against Age")



plt.subplot2grid((2,2),(0,1))

x=range(1,15,1)

plt.hist(x=[dataexplore.loc[dataexplore.Survived==0,"Family"], dataexplore.loc[dataexplore.Survived==1,"Family"]], bins=x, rwidth=0.9, label=["not survived", "survived"])

plt.xticks(x)

plt.legend()

plt.xlabel("Family Size")

plt.title("Survival Distribution against Family Size")



plt.subplot2grid((2,2),(1,0))

x=range(0,600,50)

plt.hist(x=[dataexplore.loc[dataexplore.Survived==0,"Fare"], dataexplore.loc[dataexplore.Survived==1,"Fare"]], bins=x, rwidth=0.9, label=["not survived", "survived"])

plt.xticks(x)

plt.legend()

plt.xlabel("Fare")

plt.title("Survival Distribution against Fare")



plt.subplot2grid((2,2),(1,1))

x=range(0,55,5)

plt.hist(x=[dataexplore.loc[(dataexplore.Survived==0) & (dataexplore.Fare<=50),"Fare"], dataexplore.loc[(dataexplore.Survived==1) & (dataexplore.Fare<=50),"Fare"]], bins=x, rwidth=0.9, label=["not survived", "survived"])

plt.xticks(x)

plt.legend()

plt.xlabel("Fare")

plt.title("Survival Distribution against Fare<=50")
data["Alone"] = 1

data.loc[data.Family>1, "Alone"] = 0

data["Elder"] = 1

data.loc[data.Age<65, "Elder"] = 0

data["Cheapticket"] = 1

data.loc[data.Fare>10, "Cheapticket"] = 0

data.sample(10)
data.columns
from sklearn.ensemble import RandomForestClassifier

predictors = ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 

              'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Family', 'Prefix_Master', 'Prefix_Miss', 

              'Prefix_Mr', 'Prefix_Mrs', 'Prefix_Others', 'Alone', 'Elder', 'Cheapticket']



x_train = data.loc[pd.isna(data.Survived)==False, predictors]

y_train = data.loc[pd.isna(data.Survived)==False, "Survived"]





x_test = data.loc[pd.isna(data.Survived)==True, predictors]



model = RandomForestClassifier(max_depth=5, random_state=0)

model.fit(x_train, y_train)

model.predict(x_test)