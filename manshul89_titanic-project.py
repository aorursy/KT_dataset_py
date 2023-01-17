# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("..//input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head(10)
train_df.columns
train_df.info()

print("...................")

test_df.info()
train_df.head()
train_df = train_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df    = test_df.drop(['Name','Ticket'], axis=1)
train_df.head(5)
test_df.head(5)
train_df.describe()
train_df["Embarked"].value_counts()
train_df["Embarked"] = train_df["Embarked"].fillna("S")
train_df.info()
sns.factorplot("Embarked","Survived", data = train_df, size = 3 , aspect = 4)
sns.countplot("Embarked", data = train_df)
sns.countplot("Survived", data = train_df)
sns.countplot(x = "Survived", hue = "Embarked", data = train_df)
#Create dummy values for embarked 

embarked_dummy_train = pd.get_dummies(train_df["Embarked"])

embarked_dummy_train.head()
#drop S embarked as it has low rate of survival

embarked_dummy_train.drop("S", axis = 1 , inplace= True)

embarked_dummy_train.head()
train_df.head()
train_df = train_df.join(embarked_dummy_train)
train_df.head()
train_df.drop("Embarked", axis= 1 , inplace= True)
train_df.head()
test_df.head()
embarked_dummy_test = pd.get_dummies(test_df["Embarked"])
embarked_dummy_test.head()
embarked_dummy_test.drop("S", axis = 1 , inplace= True)
embarked_dummy_test.head()
test_df = test_df.join(embarked_dummy_test)
test_df.drop("Embarked", axis=1, inplace= True)

test_df.head()
train_df.info()

print("........")

test_df.info()
#Fill Fare missing Values in test set by median

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
test_df.info()
#Change the type float to int

train_df["Fare"] = train_df["Fare"].astype(int)

test_df["Fare"] = test_df["Fare"].astype(int)
train_df.info()

print("........")

test_df.info()
fare_survived = train_df["Fare"][train_df["Survived"] == 1]

fare_not_survived = train_df["Fare"][train_df["Survived"] == 0]
Average = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])

Average
Stdev = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])

Stdev
sns.distplot(train_df["Fare"], kde= False, bins = 30 , hist = True , color = "green")
Average.plot(kind = "bar" , yerr = Stdev)
train_df.info()

print("........")

test_df.info()
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')

train_df['Age'].dropna().astype(int).hist(bins=70, ax = axis1)

#Age data is also missing and it is in float 

average_age_train = train_df["Age"].mean()

std_age_train = train_df["Age"].std()

count_age_train_null = train_df["Age"].isnull().count()

average_age_test = test_df["Age"].mean()

std_age_test = test_df["Age"].std()

count_age_test_null = test_df["Age"].isnull().count()

#If you want to fill lots of missing values for age then create random values 

rand_train = np.random.randint(average_age_train - std_age_train , average_age_train + std_age_train , size = count_age_train_null)

rand_test = np.random.randint(average_age_test - std_age_test , average_age_test + std_age_test , size = count_age_test_null)

#fill values now 

train_df["Age"][np.isnan(train_df["Age"])] = rand_train

test_df["Age"][np.isnan(test_df["Age"])] = rand_test

#convert the type

train_df["Age"]=train_df["Age"].astype(int)

test_df["Age"]=test_df["Age"].astype(int)

train_df['Age'].hist(bins=70, ax = axis2)
target_1 = train_df["Age"][train_df["Survived"]== 0]

target_2 = train_df["Age"][train_df["Survived"]== 1]
sns.distplot(target_1, hist = False, color = "Blue" , label = "Not Survived")

sns.distplot(target_2, hist = False, color = "green", label= "Survived")
train_df.info()

print("........")

test_df.info()
#Cabin Values are missing. Explore the cabin values. Only 2ö4 values are present out of 891 train samples. Hence drop it 

train_df["Cabin"].describe()
train_df.drop("Cabin", axis = 1 , inplace = True)

test_df.drop("Cabin", axis = 1 , inplace = True)

train_df.info()

print("......................")

test_df.info()

#no missing values are remaining now
train_df.head(10)
train_df.columns
#Family data cleaning 

train_df["Family"] = train_df["SibSp"] + train_df["Parch"]

test_df["Family"] = test_df["SibSp"] + test_df["Parch"]
sns.countplot(train_df["Family"])
sns.countplot(test_df["Family"])
train_df["Family"][train_df["Family"]>0]= 1

train_df["Family"][train_df["Family"]==0]= 0

sns.countplot(train_df["Family"])

test_df["Family"][test_df["Family"]>0]= 1

test_df["Family"][test_df["Family"]==0]= 0

sns.countplot(test_df["Family"])

train_df.head()
#drop SibSp and Parch

train_df.drop(["SibSp","Parch"], axis = 1 , inplace= True)

test_df.drop(["SibSp","Parch"], axis = 1 , inplace= True)
train_df.head()
test_df.head()
sns.countplot(train_df["Family"], hue= train_df["Survived"])
train_df["People"]  =train_df[['Age',"Sex"]].apply(lambda x: x["Sex"] if x["Age"] >= 16 else "Child" , axis = 1) 
train_df["People"].value_counts()
test_df["People"]  =test_df[['Age',"Sex"]].apply(lambda x: x["Sex"] if x["Age"] >= 16 else "Child" , axis = 1) 
test_df["People"].value_counts()
train_df[train_df["People"]== "male"][train_df["Survived"]== 1].count()/(train_df[train_df["People"]== "male"].count())


for cat in ("male", "female","Child"):

    x = (train_df[train_df["People"]== cat][train_df["Survived"]== 1].count())/(train_df[train_df["People"]== cat].count())

    print ( cat + str(x) )
train_df.head()
train_df.drop("Sex", axis = 1 , inplace= True)
train_df.head()
test_df.head()
test_df.drop("Sex", axis = 1 , inplace = True)
test_df.head()
#get dummy values

people_dummy_train = pd.get_dummies(train_df["People"])

people_dummy_train.head()
people_dummy_train.drop("male",axis = 1 , inplace= True)

people_dummy_train.head()
people_dummy_test = pd.get_dummies(test_df["People"])
people_dummy_test.head()
people_dummy_test.drop("male", axis = 1 , inplace= True)
people_dummy_test.head()
#join dummy dataframes with original tables

train_df = train_df.join(people_dummy_train)

test_df = test_df.join(people_dummy_test)
train_df.head()
test_df.head()
train_df.drop("People", axis = 1 , inplace= True)
test_df.drop("People", axis = 1 , inplace = True)
train_df.head()
test_df.head()


train_df["Pclass"].value_counts()
sns.countplot(train_df["Pclass"], hue = train_df["Survived"])
sns.factorplot("Pclass", "Survived", data= train_df)
train_df.head()
class_dummy_train = pd.get_dummies(train_df["Pclass"])

class_dummy_train.columns = ["Class1", "Class2", "Class3"]
class_dummy_train.head()
class_dummy_train.drop("Class3", axis= 1 , inplace= True ) 
class_dummy_train.head()
train_df=train_df.join(class_dummy_train)
train_df.head()
train_df.drop("Pclass", axis = 1 , inplace = True)
train_df.head()
test_df.head()
Class_dummy_test = pd.get_dummies(test_df["Pclass"])

Class_dummy_test.columns = ["Class1","Class2","Class3"]
Class_dummy_test.head()
Class_dummy_test.drop("Class3", axis= 1 , inplace= True)
Class_dummy_test.head()
test_df = test_df.join(Class_dummy_test)
test_df.head()
test_df.drop("Pclass", axis = 1 , inplace= True)
test_df.head()
#define training and testing data 

X_train = train_df.drop("Survived", axis = 1)

Y_train = train_df["Survived"]

X_test = test_df.drop("PassengerId", axis = 1 ).copy()
X_train.head()
Y_train.head()
X_test.head()
#Apply models now 

#logistic regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
logistic = LogisticRegression()

logistic.fit(X_train,Y_train)

prediction = logistic.predict(X_test)

logistic.score(X_train,Y_train)

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train,Y_train)

predict = svc.predict(X_test)

svc.score(X_train,Y_train)
from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier()
random.fit(X_train,Y_train)

predict = random.predict(X_test)

random.score(X_train,Y_train)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)

predict = knn.predict(X_test)

knn.score(X_train,Y_train)