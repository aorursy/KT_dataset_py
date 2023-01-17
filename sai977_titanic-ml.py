 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
"""importing dataset"""
raw_dataset= pd.read_csv("../input/train.csv")
copy_dataset = raw_dataset

"""cleaning the data"""
#dataset.isnull().sum()
dataset = raw_dataset.drop("Cabin", 1)

"""filling in missing values of embarked column"""
dataset[dataset["Embarked"].isnull() == True]
dataset["Embarked"].fillna(value = "C" , inplace= True )

"""Creating a title column  from the name column"""
dataset["title"]   = dataset["Name"].str.split("," , expand = True)[1].str.split(".", expand = True)[0]

"""FINDING THE AGE AVERAGE BASED ON THE TITLES"""

mr_mean = dataset[dataset["title"] == " Mr" ]["Age"].mean()
mrs_mean = dataset[dataset["title"] == " Mrs" ]["Age"].mean()
miss_mean = dataset[dataset["title"] == " Miss" ]["Age"].mean()
master_mean = dataset[dataset["title"] == " Master" ]["Age"].mean()
other_mean = dataset[dataset["title"] != " Mr" ][dataset["title"] != " Mrs"][dataset["title"] != " Miss"][dataset["title"] != " Master"]["Age"].mean()

"""filling in missing values of nan in age column"""
mr = dataset[((dataset["Age"].isnull()) == True) & ((dataset["title"] == " Mr") == True)]["Age"].index
dataset.loc[mr,"Age"] = mr_mean
 
mrs = dataset[((dataset["Age"].isnull()) == True) & ((dataset["title"] == " Mrs") == True)]["Age"].index
dataset.loc[mrs,"Age"] = mrs_mean       

Miss = dataset[((dataset["Age"].isnull()) == True) & ((dataset["title"] == " Miss") == True)]["Age"].index
dataset.loc[Miss,"Age"] = miss_mean     

master = dataset[((dataset["Age"].isnull()) == True) & ((dataset["title"] == " Master") == True)]["Age"].index
dataset.loc[master,"Age"] = master_mean     

other = dataset[((dataset["Age"].isnull()) == True) & ((dataset["title"] == " Other") == True)]["Age"].index
dataset.loc[other,"Age"] = other_mean     
      
dataset.loc[766, "Age"] = 30   


del mr, mrs, Miss, master, other, mr_mean, mrs_mean, miss_mean,  master_mean, other_mean 

"""Finding out if person came alone"""
i = 0
for i in range (0, dataset.shape[0]):
    a = dataset.loc[i,"SibSp"] + dataset.loc[i,"Parch"]
    if a > 0:
        
        dataset.loc[i,"Alone"] = 0
    else:
        
        dataset.loc[i,"Alone"] = 1
del i

"""Dummy Variables"""
from sklearn.preprocessing import LabelEncoder
label_encoderX = LabelEncoder()
dataset["Sex"] =  label_encoderX.fit_transform(dataset["Sex"])
dataset["Embarked"] =  label_encoderX.fit_transform(dataset["Embarked"])
dataset["title"] =  label_encoderX.fit_transform(dataset["title"])

dataset[["E0","E1","E2"]] = pd.get_dummies(dataset["Embarked"])
dataset[["P0","P1","P2"]] = pd.get_dummies(dataset["Pclass"])
 
"""Feature Selection"""
y = dataset["Survived"]
x = dataset[[ "Sex","P0","P1","P2","Fare","Age"]]

"""train Test Split"""
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=2/3, random_state = 0)



"""Decision tree classifier"""
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = "entropy")
classifier.fit(x_train,y_train)

y_train_pred = classifier.predict(x_train)
y_test_pred = classifier.predict(x_test)

print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_test,y_test_pred))

