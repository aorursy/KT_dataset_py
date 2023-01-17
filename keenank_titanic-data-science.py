import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



#Machine learning (later use)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('../input/train.csv') #train_data as a pandas dataframe

test_data = pd.read_csv('../input/test.csv')   #train_data as a pandas dataframe

testAndTrain = [test_data, train_data]

#print(testAndTrain)

#print(train_data)

#print(test_data)

print("___Test Data___")

test_data.info()

print('_'*40)

print("___Train Data___")

train_data.info()
#train_data.replace('female',0,inplace=True)

#train_data.replace('male',1,inplace=True)

#print(train_data) Check to make sure it worked, it does.
#svs = sns.barplot(x="Sex", y="Survived", data=train_data)

f_survive=train_data[["Sex", "Survived"]].groupby("Sex")

f_survive.describe()
#Sex

f, (ax1, ax2) = plt.subplots(1,2)

sns.countplot(x="Sex", data=train_data, ax=ax1)

sns.barplot(x="Sex", y="Survived", data=train_data, ax=ax2)



#__________________________________IGNORE___________________________________________________#



#####These were attempts to plot histograms. Useless here but perhaps useful soon#########

#sns.distplot(train_data.Sex)

#females=train_data[train_data['Sex']==0]

#males=train_data[train_data['Sex']==1]

#females["Survived"].hist()



#plt.hist([females.Survived, males.Survived],)

#sns.distplot(males.Survived)

#sns.distplot(females.Survived)

#_________________________________________________________________________________________#
#Count age

sns.countplot(x="Age", data=train_data)
#Survived vs. Age

fig, ax1 = plt.subplots(1,1,figsize=(20,4))

ages = train_data[["Age", "Survived"]].groupby("Age", as_index=False).mean()

sns.barplot(x="Age", y="Survived", data=ages)

###############NEED TO CHANGE SO THAT INCREMENTS DON'T AFFECT AGE VALUES##########

#plt.xticks(np.arange(0,95,2))





#as_index=False was needed in order to allow sns.barplot to recognize "Age" as a column name

#plt.xticks was included to adjust the x-axis of the plot. Before the plot had too many 

#values on the x-axis and was unable to read anything

#sns.axes_style('xtick.major.size':5.0)
#Tickets

fig, ax1 = plt.subplots(1,1,figsize=(20,4))

sns.barplot(x="Ticket", y="Survived", data=train_data)
#Speaking of, Class and Fare



#Any Correlation between Class and Fare?

plt.scatter(x="Pclass", y="Fare", data=train_data)

plt.xlabel("Passenger class")

plt.ylabel("Fare ($)")
#Separate data by class

#Utilized pandas .loc function here instead of hard-coding a Python method

##Could also use .isin pandas function instead of ==

class_1 = train_data.loc[train_data["Pclass"] == 1]

#print(class_1)

class_2 = train_data.loc[train_data["Pclass"] == 2]

class_3 = train_data.loc[train_data["Pclass"] == 3]



#Histogram for 1st class

sns.distplot(class_1["Fare"])

#sns.countplot(x="Fare", data=class_1)



num_class1 = len(class_1.index)

num_class2 = len(class_2.index)

num_class3 = len(class_3.index)



print("Class1:",num_class1,"Class2:",num_class2,"Class3:",num_class3) 

#Histogram for 2nd class

sns.distplot(class_2["Fare"])
#Histogram for 3rd class

sns.distplot(class_3["Fare"])
#Survived vs. Pclass



sns.barplot(x="Pclass", y="Survived", data=train_data)



f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

sns.countplot(class_1["Survived"], ax=ax1)

ax1.title.set_text("1st Class")

sns.countplot(class_2["Survived"], ax=ax2)

ax2.title.set_text("2nd Class")

sns.countplot(class_3["Survived"], ax=ax3)

ax3.title.set_text("3rd Class")



#####################SET THE LABELS TO SEE WHICH PLOT IS WHICH CLASS#######

#####################Side labels instead????????????????
#Embarked

#C = Cherbourg, Q = Queenstown, S = Southampton

sns.countplot(train_data["Embarked"])
sns.barplot(x="Embarked", y="Survived", data=train_data)
Embarked_C = train_data.loc[train_data["Embarked"] == "C"]

Embarked_S = train_data.loc[train_data["Embarked"] == "S"]

Embarked_Q = train_data.loc[train_data["Embarked"] == "Q"]

print(Embarked_C)
sns.countplot(Embarked_C["Survived"])
#Check male/female rate of survival for people embarked from Cherbourg

sns.barplot(x="Sex", y="Survived", data=Embarked_C)
sns.barplot(x="Embarked", y="Pclass", data=train_data)
f, (ax1, ax2, ax3) = plt.subplots(3)

sns.barplot(x="Sex", y="Survived", data=Embarked_C, ax=ax1)

sns.barplot(x="Sex", y="Survived", data=Embarked_S, ax=ax2)

sns.barplot(x="Sex", y="Survived", data=Embarked_Q, ax=ax3)