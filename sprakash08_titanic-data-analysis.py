# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Get all the imports

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
sns.set_palette('viridis')
titanic_train = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")
titanic_train.head(10)
titanic_test.head(10)
#Remove Columns
titanic_train.drop(columns=["PassengerId","Name","Ticket"],inplace=True)
test = titanic_test.drop(columns=["PassengerId","Name","Ticket"],inplace=False)
titanic_train.head(5)
male_mar = titanic_train[(titanic_train["Sex"] == "male") & (titanic_train["SibSp"] == 1) & (titanic_train["Parch"] > 0)]
female_mar = titanic_train[(titanic_train["Sex"] == "female") & (titanic_train["SibSp"] == 1) & (titanic_train["Parch"] > 0)]
nan_child = male_mar = titanic_train[(titanic_train["Sex"] == "male") & (titanic_train["SibSp"] == 1) & (titanic_train["Parch"] == 0)]
male_mar_mean = male_mar["Age"].mean()
female_mar_mean = female_mar["Age"].mean()
nan_mean = nan_child["Age"].mean()

age_mean = (male_mar_mean + female_mar_mean + nan_mean) / 3

age_mean
type(titanic_train["Age"].iloc[0])
titanic_train.Age.fillna(age_mean,inplace=True)
test.Age.fillna(age_mean,inplace=True)
titanic_train.describe()
sns.distplot(titanic_train['Age'],bins=30,kde=False)
#Count of men
men = titanic_train[titanic_train["Sex"] == "male"]
#Count of female
women = titanic_train[titanic_train["Sex"] == "female"]

print("Men -->" + str(len(men)))
print("Women -->" + str(len(women)))

sns.countplot(x="Sex",data=titanic_train)
ax = sns.countplot(x="Pclass",data=titanic_train,hue="Sex")
#How many non siblings are on the ship?
non_sibling = titanic_train[titanic_train["SibSp"] == 0]
sns.countplot(x='Sex',data=non_sibling)
#No of male having no siblings
non_sibling[non_sibling['Sex']=="male"]["Sex"].count()
#No of female having no siblings
non_sibling[non_sibling['Sex']=="female"]["Sex"].count()

men_survived = titanic_train[(titanic_train["Sex"] == "male") & (titanic_train["Survived"] == 1)]
men_not_survived = titanic_train[(titanic_train["Sex"] == "male") & (titanic_train["Survived"] == 0)]
women_survived = titanic_train[(titanic_train["Sex"] == "female") & (titanic_train["Survived"] == 1)]
women_not_survived = titanic_train[(titanic_train["Sex"] == "female") & (titanic_train["Survived"] == 0)]
print("Survived Men --> " + str(len(men_survived)))
print("Survived Women --> " + str(len(women_survived)))
print("Not Survived Men --> " + str(len(men_not_survived)))
print("No Survived Women --> " + str(len(women_not_survived)))

sns.countplot(x="Survived",data=titanic_train,hue="Sex")
titanic_train["Embarked"].unique()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
titanic_train["Embarked"].value_counts()

titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")
dict = {"S":0,"C":1,"Q":2}

def convert_e_2_num(x):
    if x in dict:
        return int(dict[x])
    return 0
dictS = {"male":0,"female":1}

def convert_s_2_num(x):
    if x in dictS:
        return int(dictS[x])
titanic_train['Emb'] = titanic_train['Embarked'].apply(convert_e_2_num)
test["Emb"] = test['Embarked'].apply(convert_e_2_num)
titanic_train.head(10)
titanic_train["S"] = pd.to_numeric(titanic_train['Sex'].apply(convert_s_2_num))
test["S"] = pd.to_numeric(test['Sex'].apply(convert_s_2_num))


titanic_train.head(10)

titanic_train.info()


dtree = DecisionTreeClassifier()
#Taken from Github: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823 (Modified duplicate printing)
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    #heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    #heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
x = titanic_train.drop(columns=['Survived','Cabin','Sex','Embarked'],axis=1)
y = titanic_train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=0)
dtree.fit(x_train,y_train)
scores = cross_val_score(dtree,x_train,y_train,cv=10)
scores.mean()
test_1 = test.drop(columns=['Cabin','Sex','Embarked'],axis=1)
test_1.Fare.fillna(0.0,inplace=True)
pred = dtree.predict(x_test)
print(classification_report(y_test,pred))
np = confusion_matrix(y_test,pred)
print_confusion_matrix(np,["Survived","Not Survived"],figsize=(8,5))
#some actual predictions
predictions = dtree.predict(test_1)
len(predictions)
#Not used
#Write to a csv
'''
count=0
preds = str(predictions,encoding="utf-8")
with open("../input/gender_submission.csv","w") as f:
    f.write("PassengerId", "Survived")
    while(count < len(predictions)):
        strObject = str(titanic_test["PassengerId"].iloc[count]) + "," + str(predictions[count])
        f.write(strObject)
        count+=1
    f.close()
'''
randf = RandomForestClassifier()
scoreR = cross_val_score(randf,x_train,y_train)
scoreR.mean()
randf.fit(x_train,y_train)
pred = randf.predict(x_test)
print(classification_report(y_test,pred))
cp = confusion_matrix(y_test,pred)
print_confusion_matrix(cp,["Survived","Not Survived"])

