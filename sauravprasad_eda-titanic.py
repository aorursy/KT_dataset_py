import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

from sklearn.tree import DecisionTreeClassifier



%matplotlib inline
#loading data

test=pd.read_csv("../input/titanic/test.csv")

train=pd.read_csv("../input/titanic/train.csv")
#Combined Graph

sns.set_style("dark")



plt.figure(figsize=(10,5))

sns.scatterplot(x="Fare",y="Age",data=train,hue="Survived")

plt.xlabel("Fare",fontsize=20)

plt.ylabel("Age",fontsize=20)

plt.title(f"(Combined) Fare V/s Age, Correlation: {round(train.Age.corr(train.Fare),2)}",fontsize=20)
#Lets Find the is there any relationship between Age and Fare

plt.figure(figsize=(20,7))

sns.set_style("dark")



#Survived Case

plt.subplot(1,2,1)

x=train[train["Survived"]==1]

sns.scatterplot(x="Fare",y="Age",data=x,palette="muted",color="orange")

plt.xlabel("Fare",fontsize=20)

plt.ylabel("Age",fontsize=20)

plt.title(f"(Survived) Fare V/s Age, Correlation: {round(x.Age.corr(x.Fare),2)}",fontsize=20)



#Not Survived Case

plt.subplot(1,2,2)

x=train[train["Survived"]==0]

sns.scatterplot(x="Fare",y="Age",data=x,palette="muted")

plt.xlabel("Fare",fontsize=20)

plt.ylabel("Age",fontsize=20)

plt.title(f"(Not Survived) Fare V/s Age, Correlation: {round(x.Age.corr(x.Fare),2)}",fontsize=20)



#Extracting Title from the name

def title(x):

    

    """

    This Function take the passenger name as an input

    and extract the title from the name as an output

    """

    

    if "Mrs." in x:

        return "Mrs."

    elif "Mr." in x:

        return "Mr."

    elif "Miss." in x:

        return "Miss."

    elif "Master." in x:

        return "Master."

    else:

        return "Others"



train["Title"]=train["Name"].apply(title)
plt.figure(figsize=(20,7))



#For Gender

plt.subplot(1,2,1)

sns.countplot(x="Sex",data=train,hue="Survived",)

plt.xlabel("Gender",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Frequency Distribution of Gender",fontsize=20)



#For Title

plt.subplot(1,2,2)

sns.countplot(x="Title",hue="Survived",data=train,)

plt.xlabel("Title",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Frequency Distribution of Title",fontsize=20)

plt.figure(figsize=(20,7))



#For Gender

plt.subplot(1,2,1)

sns.countplot(x="Pclass",data=train,hue="Sex",)

plt.xlabel("Passenger Class",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Frequency Distribution of Passenger Class",fontsize=20)



#For Title

plt.subplot(1,2,2)

sns.countplot(x="Pclass",hue="Title",data=train,palette="muted")

plt.xlabel("Passenger Class",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Frequency Distribution of Passenger Class",fontsize=20,)

plt.figure(figsize=(20,7))



#For Gender

plt.subplot(1,2,1)

sns.countplot(x="Pclass",data=train,hue="Survived")

plt.xlabel("Passenger Class",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Frequency Distribution of Passenger Class",fontsize=20)

def Fare_cat(x):

    

    """

    This Function take Fare as an input and 

    classifies it as Free, Low, Medium, or High.

    """

    

    if x==0:

        return "Free"

    elif x>0 and x<14:

        return "Low"

    elif x>=14 and x<32:

        return "Medium"

    else:

        return "High"



#applying the function    

train["Fare_cat"]=train["Fare"].apply(Fare_cat)
plt.figure(figsize=(20,8))



#For Gender

plt.subplot(1,2,1)

sns.countplot(x="Fare_cat",data=train,hue="Survived",)

plt.xlabel("Fare Category",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Frequency Distribution of Price Category",fontsize=20)



#For Title

plt.subplot(1,2,2)

sns.countplot(x="Fare_cat",hue="Title",data=train,palette="bright")

plt.xlabel("Fare Category",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Frequency Distribution of Fare Category",fontsize=20)

#creating family size variable

train["Family_Size"]=train["SibSp"]+train["Parch"]+1



#creating a column of being single or not

def Family(x):

    if x==1:

        return "Alone"

    else:

        return "Not Alone"

train["Status"]=train["Family_Size"].apply(Family)
plt.figure(figsize=(20,7))



#For Gender

plt.subplot(1,2,1)

sns.countplot(x="Status",data=train,hue="Survived",)

plt.xlabel("Status ",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Frequency Distribution of Status Category",fontsize=20)



#For Title

plt.subplot(1,2,2)

sns.countplot(x="Status",hue="Title",data=train,palette="bright")

plt.xlabel("Status ",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Frequency Distribution of Status Category",fontsize=20)