import numpy as np
import pandas as pd

import pylab as plt
import seaborn as sns



train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train["n"] = 0
test["n"] = 1

global tot

tot = pd.concat([train,test],sort = False)
def check_survive(label):
    global tot
    return tot[["Survived",label]].groupby(label).Survived.mean()

def process_Sex():
    
    global tot
    
    tot.Sex = tot.Sex.replace({"male": 0, "female": 1})
    

def process_Name():
    
    global tot

    
    tot["Title"] = tot.Name.str.extract("([A-Za-z]+)\.")

    
    tot["Title"].replace(['Lady', 'Countess','Sir', 'Jonkheer', 'Dona'], 'Upper',inplace=True)
    tot["Title"].replace(['Don', 'Rev','Capt','Col','Major','Dr','Rev'], 'Officer',inplace=True)
    tot.replace({"Mlle": "Miss", "Ms": "Miss", "Mme":"Mrs"},inplace=True)
    
    temp = check_survive("Title").sort_values()
    temp.iloc[:] = np.arange(len(temp))

    dic = temp.to_dict()
    
    tot.Title.replace(dic, inplace= True)

    
def process_Family():
    
    global tot
    
    tot["FamilySize"] = tot.SibSp + tot.Parch + 1
    
    tot.FamilySize = tot.FamilySize.map(lambda x: 2 if 1< x < 5 else 1 if x == 1 else 0)
    
    
def process_Embarked():
    
    global tot
    

    tot.Embarked.fillna("S",inplace  =True)

    tot.Embarked = tot.Embarked.replace({"S": 0, "Q":1, "C":2})


def fareof(x):
    if x < 7.5: return 0
    if x < 14 : return 1
    if x < 50 : return 2
    if x < 80 : return 3
    return 4
    
def process_Fare():
    
    global tot
    
    tot.Fare.fillna(0, inplace=True)
    
    tot.Fare[tot.Fare.notna()] = tot.Fare[tot.Fare.notna()].map(lambda x: fareof(x))
    

def ageof(x):
    if x < 10: return 0
    if x < 25 : return 1
    if x < 35 : return 2
    return 3
    
def process_Age():
    
    global tot

    tot.Age[tot.Age.notna()] = tot.Age[tot.Age.notna()].map(lambda x: ageof(x))
    
    dic = tot[tot.Age.notna()][["Age","Title"]].groupby("Title").Age.mean().to_dict()

    tot.Age[tot.Age.isna()] = tot.Title[tot.Age.isna()].replace(dic).map(lambda x: int(x))

    
def process_Ticket():

    global tot
    
    tot["RT"] = tot.Ticket.map(lambda x: x.split(" ")[0] if len(x.split(" "))==1 else x.split(" ")[1])
    tot["LT"] = tot.Ticket.map(lambda x: str(x.split(" ")[0])[0] if len(x.split(" ")) >1 else np.nan)

    tot["Tlen"]  = tot.RT.map(lambda x: len(x))

    tot["RT"] = tot.RT.map(lambda x: str(x)[0])


def process_Cabin():
    
    global tot

  

def  process_all():
    
    global tot

    process_Name()
    process_Embarked()
    process_Family()
    process_Fare()
    process_Sex()
    process_Age()
    process_Ticket()
    process_Cabin()

process_all()
lnames = tot.Name.map(lambda x: x.split(",")[0])

tot.Name = lnames

tnum = tot.Ticket.map(lambda x: x.split(" ")[-1])

tot.Ticket = tnum

tot["FamSize"] = tot.SibSp + tot.Parch

nlist = tot.Name.value_counts().index

tot["FamDeath"] = np.nan

for i in range(len(tot)):
    
    if tot.iloc[i,:].FamSize > 0: 
        
        hisname = tot.iloc[i,:].Name
        hisfam = tot.iloc[i,:].FamSize

        temp = pd.concat([tot.iloc[:i,:], tot.iloc[i+1:,:]])

        family = temp[(temp.Name == hisname) * (temp.FamSize == hisfam)]
        
        if len(family) == 0:
            continue

        tot.FamDeath.iloc[i] = family.Survived.mean()

tot.FamDeath.fillna(0.5,inplace=True)
    
 
del  tot["Ticket"], tot["Cabin"], tot["RT"], tot["LT"]
del tot["FamSize"], tot["Name"]
tot.head()
from sklearn.ensemble import RandomForestClassifier

dropped = ["Survived","n"]

clf = RandomForestClassifier(oob_score=True, n_estimators=100, max_depth=5)

clf.fit(tot[tot.n == 0].drop(dropped,axis = 1), tot[tot.n==0].Survived)

features = pd.DataFrame(clf.feature_importances_, index=tot[tot.n == 0].drop(dropped,axis = 1).keys()).sort_values(by = 0, ascending = False)

features.plot.bar(legend = False)

plt.savefig("features.png")
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV

dropped = ["Survived","n","PassengerId","Embarked","Parch","Age","SibSp","Tlen"]

parameters = {
        'n_estimators'      : [100],
        'random_state'      : [1],
        'n_jobs'            : [3],
        'min_samples_split': np.arange(8,12),
        'max_depth'         : np.arange(2,6)
}


clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
clf.fit(tot[tot.n == 0].drop(dropped,axis = 1), tot[tot.n==0].Survived)
 
print(clf.best_estimator_)

from sklearn.model_selection import train_test_split

data = []

clf = clf.best_estimator_

num_trial = 10

for i in range(num_trial):
    
    X_train, X_test, y_train, y_test = train_test_split(tot[tot.n == 0].drop(dropped,axis = 1), tot[tot.n==0].Survived, random_state = i)

    clf.fit(X_train, y_train)
    
    data.append(clf.score(X_test, y_test))
    
plt.scatter(np.arange(num_trial),data)
ser = pd.Series(data)

plt.figure("Validation")
fig = sns.distplot(data,axlabel="Score").get_figure()
plt.savefig("dist.png")
clf.fit(tot[tot.n == 0].drop(dropped,axis = 1), tot[tot.n == 0].Survived)
subm = tot[tot.n == 1].drop(["Survived"], axis = 1).join(pd.Series(clf.predict(tot[tot.n == 1].drop(dropped,axis = 1)),name="Survived"))
subm = subm[["PassengerId","Survived"]].set_index("PassengerId")

subm.Survived = subm.Survived.map(lambda x: int(x))

subm.to_csv("Submission.csv")
