#libraries

import numpy as np 

import pandas as pd 



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Let's be rebels and ignore warnings for now

import warnings





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#uploading our datasets into the notebook ( datasets were already given )

import pandas as pd

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train = pd.read_csv('/kaggle/input/titanic/train.csv')

gender_submission= pd.read_csv("../input/titanic/gender_submission.csv")
#View the train dataset. Head function selects first 5 lines

train.head()
#View the test dataset

test.head()
gender_submission.head()
len(train)
len(test)


#calculating missing value rate

train.isnull().mean().round(4) * 100
missingno.matrix(train, figsize=(30,5))
train["Age"].fillna(train["Age"].mean(), inplace=True)

test["Age"].fillna(test["Age"].mean(), inplace=True)
#Survived people comparison

fig= plt.figure(figsize = (20,1))

#countplot show the total amount of numbers

sns.countplot(y="Survived", data=train)

#distribution by gender 

fig= plt.figure(figsize = (20,1))

sns.countplot(y="Sex", data=train)
#graph

sns.barplot(x="Sex", y="Survived", data=train)



#% of men survived

men = train.loc[train.Sex == 'male']["Survived"]

rate_men = round((sum(men)/len(men))*100,2)

print("% of men who survived:", rate_men)



#% of women survived 

women = train.loc[train.Sex=="female"]["Survived"]

rate_women=round((sum(women)/len(women))*100,2)

print("% of women survived", rate_women)
#graph

sns.barplot(x="Pclass", y="Survived", data=train)

#Survival ratio based on Pclass

print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
sns.barplot(x="SibSp", y="Survived", data=train)



train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.Age.plot.hist()
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
#Bins are like group filters and labels are the names of the groups

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()

#Same thing what we did with Pclass 

sns.barplot(x="Embarked", y="Survived", data=train)

print("Percentage of Embarked = S who survived:", train["Survived"][train["Embarked"] == "S"].value_counts(normalize = True)[1]*100)

print("Percentage of Embarked = Q who survived:", train["Survived"][train["Embarked"] == "Q"].value_counts(normalize = True)[1]*100)

print("Percentage of Embarked = C who survived:", train["Survived"][train["Embarked"] == "C"].value_counts(normalize = True)[1]*100)

g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Fare', bins=20)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['0', '1', '2', '3', '4', '5', '6', '7']

train['AgeGroupNumber'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroupNumber'] = pd.cut(test["Age"], bins, labels = labels)

test.head()


Names=train.Name.str.split(" ").map(lambda x: x[1])

Names_2=test.Name.str.split(" ").map(lambda x: x[1])

# If you want to see all unique values type this - Names.value_counts()

train["Title"]=Names

test["Title"]=Names_2



train["Title"] = train["Title"].replace(['Ms.','Mrs','Mulder,','Capt.','Pelsmaeker,','Cruyssen,','Walle,','Don.','Jonkheer.','Melkebeke,','der',

                                        'Velde,','Messemaeker,','Billiard,','Shawah,','the','Steen,','Carlo,','Gordon,', 

                                        'Mlle.','Major.','Col.', 'Impe,', 'Planke,','y','Rev.', 'Rev.','Mme.'],

                                        'Rare')

test["Title"] = test["Title"].replace(['Ms.','Mrs','Mulder,','Capt.','Pelsmaeker,','Cruyssen,','Walle,','Don.','Jonkheer.','Melkebeke,','der',

                                        'Velde,','Messemaeker,','Billiard,','Shawah,','the','Steen,','Carlo,','Gordon,', 

                                        'Mlle.','Major.','Col.', 'Impe,', 'Planke,','y','Rev.', 'Rev.','Mme.'],

                                        'Rare')

#Changing words into numbers

title_mapping = {"Mr.": 0, "Miss.": 1, "Mrs.": 2,"Rare.": 3,"Master": 4,"Dr.": 5, }

train['TitleGroup'] = train['Title'].map(title_mapping)

test['TitleGroup'] = test['Title'].map(title_mapping)
#Changing words into numbers 

sex_mapping = {"male": 0, "female": 1}

train['SexGroup'] = train['Sex'].map(sex_mapping)

test["SexGroup"] = test["Sex"].map(sex_mapping)

test.head()
#Changing words into numbers 

embarked_mapping = {"S": 0, "C": 1, "Q" : 2}

train["EmbarkedGroup"]=train["Embarked"].map(embarked_mapping)

test["EmbarkedGroup"]=test["Embarked"].map(embarked_mapping)

test.head()
fare=train["Fare"]

max(fare)
bins = [0, 10, 50, 100, 200, 550, np.inf]

labels = ['0', '1', '2', '3', '4', '5']

train['FareGroup'] = pd.cut(train["Fare"], bins, labels = labels)

test['FareGroup'] = pd.cut(test["Fare"], bins, labels = labels)

train=train.drop(["Name",'Sex','Age','Ticket','Fare','Cabin','Embarked','AgeGroup','Title'], axis=1)

test=test.drop(["Name",'Sex','Age','Ticket','Fare','Cabin','Embarked','AgeGroup','Title'], axis=1)

test.head()
train.isnull().mean().round(4)*100
df=train["FareGroup"]

df1=test["FareGroup"]

train["FareGroup"] = df.fillna(method = 'ffill') 

test["FareGroup"] = df1.fillna(method = 'ffill') 



Ef=train["TitleGroup"]

Ef1=test["TitleGroup"]

test["TitleGroup"] = df.fillna(method = 'ffill')

train["TitleGroup"] = df.fillna(method = 'ffill')



Kf=train["EmbarkedGroup"]

Kf1=test["EmbarkedGroup"]

test["EmbarkedGroup"] = df.fillna(method = 'ffill')

train["EmbarkedGroup"] = df.fillna(method = 'ffill')

train.isnull().mean().round(4)*100
x = np.array(train["TitleGroup"])



y=np.array(train["EmbarkedGroup"])

train["TitleGroup"]=x.astype(int)

train["EmbarkedGroup"]=y.astype(int)

train["AgeGroupNumber"]=train["AgeGroupNumber"].astype(int)

train["FareGroup"]=train["FareGroup"].astype(int)



x1 = np.array(test["TitleGroup"])

y1=np.array(test["EmbarkedGroup"])

test["TitleGroup"]=x1.astype(int)

test["EmbarkedGroup"]=y1.astype(int)

test["AgeGroupNumber"]=train["AgeGroupNumber"].astype(int)

test["FareGroup"]=train["FareGroup"].astype(int)





#Splitting our data 

from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)

#Decision Tree Clasifier

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
#logistic regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
#Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
#Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
ids = test['PassengerId']

predictions = logreg.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)