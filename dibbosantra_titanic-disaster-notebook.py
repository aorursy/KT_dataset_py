import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data  = pd.read_csv('/kaggle/input/titanic/test.csv') 
train_data.head()
train_data.info()
fig,ax = plt.subplots(1, 2, figsize = (16,6))

sns.countplot(train_data['Survived'], ax = ax[0])

plt.pie(train_data.Survived.value_counts(), labels = ("Didn't Survive","Survived"), startangle=90)

ax[1].set_title("Survival pie chart")

plt.legend(loc=4)

plt.show()
_,ax = plt.subplots(1,2,figsize= (13,5))



sns.countplot("Pclass", data = train_data, ax= ax[0])

ax[0].set_xlabel("Passenger Class")

ax[0].set_title("Distribution of passengers on basis of Passenger Class")



sns.countplot("Pclass", hue = "Survived", data = train_data)

ax[1].set_xlabel("Passenger Class")

ax[1].set_title("Survial stats for various Passenger Classes")



plt.show()
train_data["Name"] = train_data["Name"].str.extract(r'([A-Za-z]+)\.')

test_data["Name"] = test_data["Name"].str.extract(r'([A-Za-z]+)\.')



print("Unique names are : ", train_data.Name.unique())

print("Unique names are : ", test_data.Name.unique())
train_data.Name = train_data.Name.replace(['Mr', 'Mrs', 'Miss','Mme', 'Ms','Mlle',],0)

train_data.Name = train_data.Name.replace(['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir',

       'Col', 'Capt', 'Countess', 'Jonkheer',"Dona"],1)



test_data.Name = test_data.Name.replace(['Mr', 'Mrs', 'Miss','Mme', 'Ms','Mlle',],0)

test_data.Name = test_data.Name.replace(['Master', 'Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir',

       'Col', 'Capt', 'Countess', 'Jonkheer','Dona'],1)



_,ax = plt.subplots(1,2,figsize= (12,5))



p = sns.countplot("Name", data = train_data, ax= ax[0])

p.set(xticklabels = ["Common Title","Not Common"])

ax[0].set_xlabel("Unique Titles")

ax[0].set_title("Distribution of passengers on basis of Unique Titles")



q = sns.countplot("Name", hue = "Survived", data = train_data)

q.set(xticklabels = ["Common Title","Not Common"])

ax[1].set_xlabel("Unique Title")

ax[1].set_title("Survial stats on basis of Unique Titles")



plt.show()
_,ax = plt.subplots(1,2,figsize = (13,5))

sns.countplot(train_data['Sex'],ax=ax[0])

ax[0].set_title("Distribution of Gender in the ship\n")

sns.countplot(train_data['Sex'],hue = train_data['Survived'],ax=ax[1])

ax[1].set_title("Survival stats for various male and female\n")

plt.show()
sex_survived = train_data.groupby("Sex").Survived.value_counts()

male_s = sex_survived["male"]/sex_survived["male"].sum() * 100

female_s = sex_survived["female"]/sex_survived["female"].sum() * 100



_,ax = plt.subplots(1,3,figsize = (25,6))

ax[0].pie(train_data["Sex"].value_counts(),

          labels = ['male','female'],startangle = 70)



ax[1].pie(sex_survived["male"],labels = ("Didn't Survive :\n {:.2f}%   ".format(male_s[0]), "Survived :\n {:.2f}% ".format(male_s[1])),

          colors=["firebrick","darkcyan"],startangle = 30 )

ax[1].set_title("Percentage of male who survied ")



ax[2].pie(sex_survived["female"],labels = ("Survived :\n {:.2f}%   ".format(female_s[1]), "Didn't Survive :\n {:.2f}%   ".format(female_s[0])),

          colors=["darkcyan", "firebrick"],startangle = 50 )

ax[2].set_title("Percentage of female who survied ")



plt.show()
train_data.Sex.replace({"male":1,"female":0},inplace=True)

test_data.Sex.replace({"male":1,"female":0},inplace=True)
plt.figure(figsize = (8,6))

sns.kdeplot(train_data.Age,shade = True,label = "All passengers")

sns.kdeplot(train_data[train_data.Survived == 1].Age,color ="r",label = "Survived")

sns.kdeplot(train_data[train_data.Survived == 0].Age,color ="black",label = "Didn't SUrvive")

plt.title("Distribution of the age of passengers")

plt.show()
train_data.Age.fillna(np.mean(train_data.Age),inplace=True)

test_data.Age.fillna(np.mean(train_data.Age),inplace=True)



age_p = np.percentile(train_data.Age,[25,50,75])





train_data.loc[train_data.Age <age_p[0],"Age"] = 1

train_data.loc[(train_data.Age >=age_p[0]) & (train_data.Age <age_p[1]),"Age"] = 2

train_data.loc[(train_data.Age >=age_p[1]) & (train_data.Age <age_p[2]),"Age"] = 3

train_data.loc[train_data.Age >=age_p[2] ,"Age"] =4



test_data.loc[test_data.Age <age_p[0],"Age"] = 1

test_data.loc[(test_data.Age >=age_p[0]) & (test_data.Age <age_p[1]),"Age"] = 2

test_data.loc[(test_data.Age >=age_p[1]) & (test_data.Age <age_p[2]),"Age"] = 3

test_data.loc[test_data.Age >=age_p[2] ,"Age"] =4







train_data.Age =train_data.Age.astype("int")

test_data.Age =test_data.Age.astype("int")
_,ax = plt.subplots(1,2,figsize= (16,5))



sns.countplot("Age", data = train_data, ax= ax[0])

ax[0].set_xlabel("Age")

ax[0].set_title("Distribution of passengers on basis of classes of Age")



sns.countplot("Age", hue = "Survived", data = train_data)

ax[1].set_xlabel("Age")

ax[1].set_title("Survial stats for various classes of Age")



plt.show()
_,ax = plt.subplots(1,3,figsize = (22,6))

sns.distplot(train_data[train_data.Survived == 1].Fare,[0,20,40,50,60,80,100,500],ax = ax[0])

ax[0].set_title("Distribution of fare for passengers who survived")

sns.distplot(train_data[train_data.Survived == 0].Fare,[0,20,40,50,60,80,100,500],ax = ax[1])

ax[1].set_title("Distribution of fare for passengers who did not survived")

sns.distplot(train_data.Fare)

ax[2].set_title("Distribution of fare for all passengers")



plt.show()
fare_p = np.percentile(train_data[train_data.Survived == 0].Fare,[25,50,75])



#Transformation for training data in Fare



train_data.loc[train_data.Fare <= fare_p[0], "Fare"]  = 1

train_data.loc[(train_data.Fare > fare_p[0]) & (train_data.Fare <= fare_p[1]), "Fare"]  = 2

train_data.loc[(train_data.Fare > fare_p[1]) & (train_data.Fare <= fare_p[2]), "Fare"]  = 3

train_data.loc[train_data.Fare > fare_p[2], "Fare" ]  = 4



train_data.Fare = train_data.Fare.astype("int")



#Transformation for test data in fare

test_data.Fare.fillna(np.mean(train_data.Fare), inplace=True)



test_data.loc[test_data.Fare <= fare_p[0], "Fare"]  = 1

test_data.loc[(test_data.Fare > fare_p[0]) & (test_data.Fare <= fare_p[1]), "Fare"]  = 2

test_data.loc[(test_data.Fare > fare_p[1]) & (test_data.Fare <= fare_p[2]), "Fare"]  = 3

test_data.loc[test_data.Fare > fare_p[2], "Fare" ]  = 4



test_data.Fare = test_data.Fare.astype("int")
_,ax = plt.subplots(1,2,figsize= (16,5))



sns.countplot("Fare", data = train_data, ax= ax[0])

ax[0].set_xlabel("Fare Price")

ax[0].set_title("Distribution of passengers on basis of classes of Fare Prices")



sns.countplot("Fare", hue = "Survived", data = train_data)

ax[1].set_xlabel("Fare Price")

ax[1].set_title("Survial stats for various classes Fare Prices")



plt.show()
train_data.Cabin  = train_data.Cabin.fillna("O") 

train_data.Cabin  = train_data.Cabin.str.extract(r'([\w\W]{1})')



test_data.Cabin  = test_data.Cabin.fillna("O") 

test_data.Cabin = test_data.Cabin.str.extract(r'([\w\W]{1})')



print(train_data.Cabin.unique())
train_data.Cabin = train_data.Cabin.replace({"O":1,'C':2, 'E':3, 'G':4, 'D':3, 'A':4, 'B':5, 'F':4, 'T':4})

test_data.Cabin = test_data.Cabin.replace({"O":1,'C':2, 'E':3, 'G':4, 'D':3, 'A':4, 'B':5, 'F':4, 'T':4})
_,ax = plt.subplots(1,2,figsize= (16,5))



sns.countplot("Cabin", data = train_data, ax= ax[0])

ax[0].set_xlabel("Cabin")

ax[0].set_title("Distribution of passengers on basis of Cabin")



sns.countplot("Cabin", hue = "Survived", data = train_data)

ax[1].set_xlabel("Cabin")

ax[1].set_title("Survial stats for various Cabins")



plt.show()
train_data.Embarked.fillna("S",inplace=True)

train_data.Embarked = train_data.Embarked.replace({"S":0,"C":1,"Q":2})

train_data.Embarked = train_data.Embarked.astype('int')



test_data.Embarked.fillna("S",inplace=True)

test_data.Embarked = test_data.Embarked.replace({"S":0,"C":1,"Q":2})

test_data.Embarked = test_data.Embarked.astype('int')
train_data.Parch  = np.clip(train_data.Parch,0,1).astype("int")

train_data.SibSp  = np.clip(train_data.SibSp,0,1).astype("int")



test_data.Parch  = np.clip(test_data.Parch,0,1).astype("int")

test_data.SibSp  = np.clip(test_data.SibSp,0,1).astype("int")
train_data.drop(["PassengerId","Ticket"], axis =1, inplace=True)

test_data.drop(["Ticket"], axis =1, inplace=True)
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
y = train_data.Survived

X = train_data.drop("Survived",axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 45)
model = XGBClassifier(n_estimators = 100000, learning_rate = 0.3 ,subsample = 0.9)

model.fit(X_train,y_train,eval_set=[(X_test, y_test)], early_stopping_rounds = 20,verbose=False,eval_metric="error")
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
X_pred  = model.predict(X_train)

predictions1 = [round(value) for value in X_pred]

accuracy1 = accuracy_score(y_train, predictions1)

print("Accuracy: %.2f%%" % (accuracy1 * 100.0))
predictions = model.predict(test_data.iloc[:,1:])
submission = pd.DataFrame({"PassengerId" : test_data.PassengerId, "Survived" : predictions})

submission.to_csv("Submission_Titanic.csv",index=False)