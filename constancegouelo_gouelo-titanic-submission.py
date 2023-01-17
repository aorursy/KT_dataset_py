import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
print(train.shape)

print(test.shape)
print("In the train Dataset, the number of missing values in the columns are \n", train.isnull().sum())

print("\n In the test Dataset, the number of missing values in teh columns are \n", test.isnull().sum())
Survived = train[["Survived"]]

train["source"] = "train"

test["source"] = "test"

dataset = pd.concat([train,test],axis=0)

print(dataset.shape)

print(dataset.head())

print(dataset.tail())
dataset.drop(["Cabin","Ticket","Survived","Name"],axis=1,inplace=True)

print(dataset.shape)

print("\n datatypes:\n")

dataset.info()
dataset["Parch"] = dataset["Parch"].astype("object")

dataset["Pclass"] = dataset["Pclass"].astype("object")

dataset["SibSp"] = dataset["SibSp"].astype("object")

dataset.info()
print(dataset["Age"].describe())

plt.rcParams["figure.figsize"]=12,4

sns.boxplot("Age",data=dataset)

plt.title("Age distribution Boxplot")
print("Missing value in Age:",dataset["Age"].fillna(30,inplace=True))

print(dataset["Age"].isnull().sum())
"Working on the Fares as well as Embarkment categories"
print("\n Embarked value count:\n" ,dataset["Embarked"].value_counts())

print("Missing value in Embarked:",dataset["Embarked"].fillna("S", inplace = True))

print("mean value of Fare is ",dataset["Fare"].mean())

print("Missing value in Fare:",dataset["Fare"].fillna(dataset["Fare"].mean(), inplace = True))

print("Total Missing Values in the dataset is:",dataset.isnull().sum().sum())
train= dataset[dataset["source"]=="train"]

test= dataset[dataset["source"]=="test"]

train.drop("source",axis=1,inplace=True)

test.drop("source",axis=1,inplace=True)

train["Survived"] = Survived["Survived"]
plt.rcParams["figure.figsize"]=13,16



plt.subplot(3,2,1)

sns.countplot(train.Pclass)

plt.title("Passenger Class Frequency Distribution")



plt.subplot(3,2,2)

sns.countplot(train.Sex)

plt.title("Sex Frequency Distribution")



plt.subplot(3,2,3)

sns.distplot(train.Fare)

plt.title("Fare Frequency Distribution")



plt.subplot(3,2,5)

sns.countplot(train.Embarked)

plt.title("Embarked Frequency Distribution")



plt.subplot(3,2,6)

sns.countplot(train.Survived)

plt.title("Survived Frequency Distribution")


