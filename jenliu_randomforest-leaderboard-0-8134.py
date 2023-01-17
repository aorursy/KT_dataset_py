# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 22:20:54 2016

@author: ReshamSarkar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

titanic_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
titanic_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

train_set = titanic_train.drop("Survived", axis = 1)
df_combo = pd.concat((train_set, titanic_test), axis = 0, ignore_index = True)

df_combo["Embarked"] = df_combo["Embarked"].fillna("C")

Title_list = pd.DataFrame(index = df_combo.index, columns = ["Title"])
Surname_list = pd.DataFrame(index = df_combo.index, columns = ["Surname"])
Name_list = list(df_combo.Name)
NL_1 = [elem.split("\n") for elem in Name_list]
ctr = 0
for j in NL_1:
    FullName = j[0]
    FullName = FullName.split(",")
    Surname_list.loc[ctr,"Surname"] = FullName[0]
    FullName = FullName.pop(1)
    FullName = FullName.split(".")
    FullName = FullName.pop(0)
    FullName = FullName.replace(" ", "")
    Title_list.loc[ctr, "Title"] = str(FullName)
    ctr = ctr + 1


#Title and Surname Extraction

Title_Dictionary = {
"Capt": "Officer",
"Col": "Officer",
"Major": "Officer",
"Jonkheer": "Sir",
"Don": "Sir",
"Sir" : "Sir",
"Dr": "Dr",
"Rev": "Rev",
"theCountess": "Lady",
"Dona": "Lady",
"Mme": "Mrs",
"Mlle": "Miss",
"Ms": "Mrs",
"Mr" : "Mr",
"Mrs" : "Mrs",
"Miss" : "Miss",
"Master" : "Master",
"Lady" : "Lady"
}    
    
def Title_Label(s):
    return Title_Dictionary[s]

df_combo["Title"] = Title_list["Title"].apply(Title_Label)
    

Surname_Fam = pd.concat([Surname_list, df_combo[["SibSp", "Parch"]]], axis = 1)
Surname_Fam["Fam"] = Surname_Fam.Parch + Surname_Fam.SibSp + 1

Surname_Fam = Surname_Fam.drop(["SibSp", "Parch"], axis = 1)

df_combo = pd.concat([df_combo, Surname_Fam], axis = 1)


Cabin_List = df_combo.loc[:,["Cabin"]]
Cabin_List = Cabin_List.fillna("UNK")
Cabin_Code = []
for j in Cabin_List.Cabin:
    Cabin_Code.append(j[0])
Cabin_Code = pd.DataFrame({"Deck" : Cabin_Code})
df_combo = pd.concat([df_combo, Cabin_Code], axis = 1)


def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0

df_combo["Fam"] = df_combo.loc[:,"Fam"].apply(Fam_label)


def Title_label(s):
    if (s == "Sir") | (s == "Lady"):
        return "Royalty"
    elif (s == "Dr") | (s == "Officer") | (s == "Rev"):
        return "Officer"
    else:
        return s
        
df_combo["Title"] = df_combo.loc[:,"Title"].apply(Title_label)     


def tix_clean(j):
    j = j.replace(".", "")
    j = j.replace("/", "")
    j = j.replace(" ", "")
    return j
    
df_combo[["Ticket"]] = df_combo.loc[:,"Ticket"].apply(tix_clean)

Ticket_count = dict(df_combo.Ticket.value_counts())

def Tix_ct(y):
    return Ticket_count[y]

df_combo["TicketGrp"] = df_combo.Ticket.apply(Tix_ct)
def Tix_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

df_combo["TicketGrp"] = df_combo.loc[:,"TicketGrp"].apply(Tix_label)   

df_combo.drop(["PassengerId", "Name", "Ticket", "Surname", "Cabin", "Parch", "SibSp"], axis=1, inplace = True)

## Filling missing Age data

mask_Age = df_combo.Age.notnull()
Age_Sex_Title_Pclass = df_combo.loc[mask_Age, ["Age", "Title", "Sex", "Pclass"]]
Filler_Ages = Age_Sex_Title_Pclass.groupby(by = ["Title", "Pclass", "Sex"]).median()
Filler_Ages = Filler_Ages.Age.unstack(level = -1).unstack(level = -1)


mask_Age = df_combo.Age.isnull()
Age_Sex_Title_Pclass_missing = df_combo.loc[mask_Age, ["Title", "Sex", "Pclass"]]

def Age_filler(row):
    if row.Sex == "female":
        age = Filler_Ages.female.loc[row["Title"], row["Pclass"]]
        return age
    
    elif row.Sex == "male":
        age = Filler_Ages.male.loc[row["Title"], row["Pclass"]]
        return age
    
Age_Sex_Title_Pclass_missing["Age"]  = Age_Sex_Title_Pclass_missing.apply(Age_filler, axis = 1)   

df_combo["Age"] = pd.concat([Age_Sex_Title_Pclass["Age"], Age_Sex_Title_Pclass_missing["Age"]])    

dumdum = (df_combo.Embarked == "S") & (df_combo.Pclass == 3)
df_combo.fillna(df_combo[dumdum].Fare.median(), inplace = True)


#df_combo["Age"] = (df_combo["Age"] - df_combo["Age"].mean())/df_combo["Age"].std()
#df_combo["Fare"] = (df_combo["Fare"] - df_combo["Fare"].mean())/df_combo["Fare"].std()
    
#### OHE encoding nominal categorical features ###
df_combo = pd.get_dummies(df_combo)


df_train = df_combo.loc[0:len(titanic_train["Survived"])-1]
df_test = df_combo.loc[len(titanic_train["Survived"]):]
total_number_param = len(df_train.columns)
df_target = titanic_train.Survived

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


from sklearn.pipeline import make_pipeline

select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)               
 
#select.fit(df_train, df_target)

pipeline.fit(df_train, df_target)
predictions = pipeline.predict(df_train)
predict_proba = pipeline.predict_proba(df_train)[:,1]

 
cv_score = cross_validation.cross_val_score(pipeline, df_train, df_target, cv= 10)
print("Accuracy : %.4g" % metrics.accuracy_score(df_target.values, predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(df_target, predict_proba))
print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), 
np.min(cv_score),
np.max(cv_score)))

 
final_pred = pipeline.predict(df_test)
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": final_pred })
submission.to_csv("RandomForest_v1.csv", index=False) 
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
titanic_train = pd.read_csv('../input/train.csv', dtype={'Age': np.float64})
titanic_test = pd.read_csv('../input/test.csv', dtype={'Age': np.float64})
train_set = titanic_train.drop("Survived", axis = 1)
df_combo = pd.concat((train_set, titanic_test), axis = 0, ignore_index = True)
df_combo.shape
list(df_combo.columns.values)
df_combo.Sex.value_counts()
len(df_combo.Sex)
843+466
df_combo.Embarked.value_counts()
np.sum(df_combo.Embarked.isnull())
Title_list = pd.DataFrame(index = df_combo.index, columns = ["Title"])
Surname_list = pd.DataFrame(index = df_combo.index, columns = ["Surname"])
Name_list = list(df_combo.Name)
df_combo
df_combo.index
df_combo.columns.values
Title_list
Name_list
type(df_combo.Name)
NL_1 = [elem.split("\n") for elem in Name_list]
type(NL_1)
NL_1[:3]
ctr = 0
for j in NL_1:
    FullName = j[0]
    FullName = FullName.split(",")
    Surname_list.loc[ctr, "Surname"] = FullName[0]
    FullName = FullName.pop(1)
    FullName = FullName.split(".")
    FullName = FullName.pop(0)
    FullName = FullName.replace(" ", "")
    Title_list.loc[ctr, "Title"] = str(FullName)
    ctr = ctr + 1
ctr
test = NL_1
test
test = NL_1[0]
test
test = test.split(",")
j = NL_1[0]
FullName = j[0]
j
FullName
j
FullName
FullName = FullName.split(",")
FullName
FullName[0]
FullName = FullName.pop(1)
FullName
FullName = FullName.split(".")
FullName
FullName = FullName.pop(0)
FullName
t = [1,2,3,4,5]
t.pop(1)
FullName = FullName.replace(" ", "")
FullName
Title_list[:20]
