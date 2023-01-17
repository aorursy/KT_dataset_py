# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#path of the file I read

test_data_path="/kaggle/input/titanic/test.csv"

#I read the file into a variable test_data

test_data = pd.read_csv(test_data_path)

test_data.info()
test_data.describe()
#path of the file I read

train_data_path="/kaggle/input/titanic/train.csv"

#I read the file into a variable train_data

train_data = pd.read_csv(train_data_path)

train_data.info()
train_data.describe()


fig=plt.figure(figsize=(21,7))

train_data.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("Survived")

fig=plt.figure(figsize=(21,7))

train_data.Sex.value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("Sex")
fig=plt.figure(figsize=(21,7))

train_data.Pclass.value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("Class")
fig=plt.figure(figsize=(21,7))

train_data.Embarked.value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("Embarked")
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
fig=plt.figure(figsize=(21,7))

train_data.Survived[train_data.Sex=="female"].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("Women who Survived")
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
fig=plt.figure(figsize=(21,7))

train_data.Survived[train_data.Sex=="male"].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("Men who Survived")
fig=plt.figure(figsize=(21,7))

train_data.Sex[train_data.Survived==1].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("Sex of Survivors")


women = train_data.loc[(train_data.Sex == 'female')&(train_data.Pclass==1)]["Survived"]

rate_women = sum(women)/len(women)

print("% of 1st class women who survived:", rate_women)



fig=plt.figure(figsize=(18,6))

train_data.Survived[(train_data.Sex=="female")&(train_data.Pclass==1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("1st class Women who Survived")

women = train_data.loc[(train_data.Sex == 'female')&(train_data.Pclass==2)]["Survived"]

rate_women = sum(women)/len(women)

print("% of 2nd class women who survived:", rate_women)



fig=plt.figure(figsize=(18,6))

train_data.Survived[(train_data.Sex=="female")&(train_data.Pclass==2)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("2nd class Women who Survived")
women = train_data.loc[(train_data.Sex == 'female')&(train_data.Pclass==3)]["Survived"]

rate_women = sum(women)/len(women)

print("% of 3rd class women who survived:", rate_women)



fig=plt.figure(figsize=(18,6))

train_data.Survived[(train_data.Sex=="female")&(train_data.Pclass==3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("3rd class Women who Survived")
men = train_data.loc[(train_data.Sex == 'male')&(train_data.Pclass==1)]["Survived"]

rate_men = sum(men)/len(men)

print("% of 1st class men who survived:", rate_men)



fig=plt.figure(figsize=(18,6))

train_data.Survived[(train_data.Sex=="male")&(train_data.Pclass==1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("1st class men who Survived")
men = train_data.loc[(train_data.Sex == 'male')&(train_data.Pclass==2)]["Survived"]

rate_men = sum(men)/len(men)

print("% of 2nd class men who survived:", rate_men)



fig=plt.figure(figsize=(18,6))

train_data.Survived[(train_data.Sex=="male")&(train_data.Pclass==2)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("2nd class men who Survived")
men = train_data.loc[(train_data.Sex == 'male')&(train_data.Pclass==3)]["Survived"]

rate_men = sum(men)/len(men)

print("% of 3rd class men who survived:", rate_men)



fig=plt.figure(figsize=(18,6))

train_data.Survived[(train_data.Sex=="male")&(train_data.Pclass==3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("3rd class men who Survived")
print("Amount of missing values for Age in train_data: ", train_data.Age.isnull().sum())

print("Amount of missing values for Age in test_data:", test_data.Age.isnull().sum())
train_data["Age"]=train_data["Age"].fillna(train_data["Age"].dropna().median())

train_data.info()
test_data["Age"]=test_data["Age"].fillna(test_data["Age"].dropna().median())

test_data.info()
test_data["Fare"]=test_data["Fare"].fillna(test_data["Fare"].dropna().median())

test_data.info()
train_data["Embarked"]=train_data["Embarked"].fillna("S")

train_data.info()
train_data.loc[train_data["Sex"]=="male", "Sex"]=0

train_data.loc[train_data["Sex"]=="female","Sex"]=1

train_data.head()
test_data.loc[test_data["Sex"]=="male", "Sex"]=0

test_data.loc[test_data["Sex"]=="female","Sex"]=1

test_data.head()
train_data.loc[train_data["Embarked"]=="S", "Embarked"]=0

train_data.loc[train_data["Embarked"]=="C", "Embarked"]=1

train_data.loc[train_data["Embarked"]=="Q", "Embarked"]=2

train_data.head()
test_data.loc[test_data["Embarked"]=="S", "Embarked"]=0

test_data.loc[test_data["Embarked"]=="C", "Embarked"]=1

test_data.loc[test_data["Embarked"]=="Q", "Embarked"]=2

test_data.head()
train_data.info()
test_data.info()
train_data["Fare"]=train_data["Fare"].fillna(0)

train_data["Fare"]=train_data["Fare"].astype(int)



test_data["Fare"]=test_data["Fare"].fillna(0)

test_data["Fare"]=test_data["Fare"].astype(int)



train_data["Age"]=train_data["Age"].fillna(0)

train_data["Age"]=train_data["Age"].astype(int)



test_data["Age"]=test_data["Age"].fillna(0)

test_data["Age"]=test_data["Age"].astype(int)

train_data.info()
test_data.info()
train_data["Age"]=train_data["Age"].astype(int)

train_data.loc[train_data["Age"]<=10,"Age"]=0

train_data.loc[(train_data["Age"]>=11) & (train_data["Age"]<=20),"Age"]=1

train_data.loc[(train_data["Age"]>=21) & (train_data["Age"]<=25),"Age"]=2

train_data.loc[(train_data["Age"]>=26) & (train_data["Age"]<=35),"Age"]=3

train_data.loc[(train_data["Age"]>=36) & (train_data["Age"]<=45),"Age"]=4

train_data.loc[(train_data["Age"]>=45) & (train_data["Age"]<=60),"Age"]=5

train_data.loc[(train_data["Age"]>=61) & (train_data["Age"]<=80),"Age"]=6



train_data.head(10)
train_data.tail()
test_data["Age"]=test_data["Age"].astype(int)

test_data.loc[test_data["Age"]<=10,"Age"]=0

test_data.loc[(test_data["Age"]>=11) & (test_data["Age"]<=20),"Age"]=1

test_data.loc[(test_data["Age"]>=21) & (test_data["Age"]<=25),"Age"]=2

test_data.loc[(test_data["Age"]>=26) & (test_data["Age"]<=35),"Age"]=3

test_data.loc[(test_data["Age"]>=36) & (test_data["Age"]<=45),"Age"]=4

test_data.loc[(test_data["Age"]>=45) & (test_data["Age"]<=60),"Age"]=5

test_data.loc[(test_data["Age"]>=61) & (test_data["Age"]<=80),"Age"]=6



test_data.head(10)
train_data["Fare"]=train_data["Fare"].astype(int)

train_data.loc[train_data["Fare"]<=10,"Fare"]=0

train_data.loc[(train_data["Fare"]>=11) & (train_data["Fare"]<=20),"Fare"]=1

train_data.loc[(train_data["Fare"]>=21) & (train_data["Fare"]<=30),"Fare"]=2

train_data.loc[(train_data["Fare"]>=31) & (train_data["Fare"]<=50),"Fare"]=3

train_data.loc[(train_data["Fare"]>=51) & (train_data["Fare"]<=100),"Fare"]=4

train_data.loc[(train_data["Fare"]>=101) & (train_data["Fare"]<=200),"Fare"]=5

train_data.loc[(train_data["Fare"]>=201) & (train_data["Fare"]<=80),"Fare"]=6



train_data.head(10)
test_data["Fare"]=test_data["Fare"].astype(int)

test_data.loc[test_data["Fare"]<=10,"Fare"]=0

test_data.loc[(test_data["Fare"]>=11) & (test_data["Fare"]<=20),"Fare"]=1

test_data.loc[(test_data["Fare"]>=21) & (test_data["Fare"]<=30),"Fare"]=2

test_data.loc[(test_data["Fare"]>=31) & (test_data["Fare"]<=50),"Fare"]=3

test_data.loc[(test_data["Fare"]>=51) & (test_data["Fare"]<=100),"Fare"]=4

test_data.loc[(test_data["Fare"]>=101) & (test_data["Fare"]<=200),"Fare"]=5

test_data.loc[(test_data["Fare"]>=201) & (test_data["Fare"]<=80),"Fare"]=6



test_data.head(10)
train_data.info()
test_data.info()
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch","Age","Fare","Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")