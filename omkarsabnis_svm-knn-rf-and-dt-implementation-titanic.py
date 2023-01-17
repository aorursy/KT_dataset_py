#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#WARNINGS
import warnings
warnings.filterwarnings('ignore') #ALWAYS DO THIS ON COMPLETING THE NOTEBOOK
#LOADING THE DATASET
trainingset=pd.read_csv("../input/train.csv")
testingset=pd.read_csv("../input/test.csv")
#VISUALZING THE DATASET
trainingset.head()
#COLUMN HEADINGS
print(trainingset.columns)
#DATATYPE OF EACH COLUMN
print(trainingset.dtypes)
#DATASET SUMMARY
trainingset.describe(include="all")
#BARPLOT OF THE SURVIVAL RATE vs CLASS OF THE PASSENGER
sns.barplot(x='Pclass',y='Survived',color='yellow',data=trainingset)
#BARPLOT OF SURVIVAL RATE vs SEX OF THE PASSENGER
sns.barplot(x='Sex',y='Survived',color='blue',data=trainingset)
#BARPLOT OF SURVIVAL RATE vs NUMBER OF SIBLINGS/SPOUSE ON BOARD
sns.barplot(x='SibSp',y='Survived',color='Green',data=trainingset)
#NUMBER OF PEOPLE IN EACH CATEGORY
print(trainingset['SibSp'].value_counts())
#BARPLOT OF SURVIVAL RATE VS FAMILY MEMBERS ON BOARD
sns.barplot(x='Parch',y='Survived',color='orange',data=trainingset)
#NUMBER OF PEOPLE IN EACH CATEGORY
print(trainingset['Parch'].value_counts())
#SINCE AGE CAN VARY, WE NEED TO PUT THEM INTO BINS.
trainingset['Age'] = trainingset['Age'].fillna(-0.5)
testingset['Age'] = testingset['Age'].fillna(-0.5)
agebins = [-1,2,8,13,19,25,38,55,np.inf]
labels = ['Missing','Babies','Children','Teenagers','Young Adults','Adults','Seniors','Old']
trainingset['AgeBin'] = pd.cut(trainingset['Age'],agebins,labels=labels)
testingset['AgeBin'] = pd.cut(testingset['Age'],agebins,labels=labels)
#BARPLOT OF SURVIVAL RATE vs AGE OF THE PASSENGER
sns.barplot(x='AgeBin',y='Survived',color='red',data=trainingset)
#NUMBER OF PEOPLE PER CATEGORY
print(trainingset['AgeBin'].value_counts())
#MAKE BINS FOR CABIN AND PLACE THEM IN THE BIN
trainingset["CabinBin"] = (trainingset["Cabin"].notnull().astype('int'))
testingset["CabinBin"] = (testingset["Cabin"].notnull().astype('int'))
#VISUAL ANALYSIS OF THE TESTING DATASET
testingset.describe(include="all")
#DROPPING THE CABIN AND TICKET FEATURES FROM BOTH DATASETS
trainingset = trainingset.drop(['Cabin'],axis=1)
trainingset = trainingset.drop(['Ticket'],axis=1)
testingset = testingset.drop(['Cabin'],axis=1)
testingset = testingset.drop(['Ticket'],axis=1)
testingset.head()
#EMBARKED FEATURE
print("Southampton (S):")
s = trainingset[trainingset["Embarked"] == "S"].shape[0]
print(s)

print("Cherbourg (C):")
c = trainingset[trainingset["Embarked"] == "C"].shape[0]
print(c)

print("Queenstown (Q):")
q = trainingset[trainingset["Embarked"] == "Q"].shape[0]
print(q)
trainingset = trainingset.fillna({'Embarked':'S'})
#COMBINE THE DATASETS
full = [trainingset,testingset]
for i in full:
    i['Title'] = i.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(trainingset['Title'],trainingset['Sex'])
#PLACE THE RARE TITLES INTO THE MORE COMMON TITLES FOR SIMPLIFICATION
for i in full:
    i['Title'] = i['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    i['Title'] = i['Title'].replace(['Countess', 'Lady', 'Sir'], 'Honararies')
    i['Title'] = i['Title'].replace('Mlle', 'Miss')
    i['Title'] = i['Title'].replace('Ms', 'Miss')
    i['Title'] = i['Title'].replace('Mme', 'Mrs')

trainingset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#MAPPING OF EACH GROUP FROM ABOVE INTO NUMERICS FOR EASY MANIPULATION
#map each of the title groups to a numerical value
mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Honararies": 5, "Rare": 6}
for i in full:
    i['Title'] = i['Title'].map(mapping)
    i['Title'] = i['Title'].fillna(0)

trainingset.head()
#PREDICTION OF MISSING VALUES IN AGE BASED ON THE TITLE BY BINNING THEM INTO PREVIOUSLY MADE BINS
mr = trainingset[trainingset["Title"] == 1]["AgeBin"].mode() 
miss = trainingset[trainingset["Title"] == 2]["AgeBin"].mode()
mrs = trainingset[trainingset["Title"] == 3]["AgeBin"].mode()
master= trainingset[trainingset["Title"] == 4]["AgeBin"].mode()
honararies = trainingset[trainingset["Title"] == 5]["AgeBin"].mode()
rare= trainingset[trainingset["Title"] == 6]["AgeBin"].mode() 

agemapping = {1: "Adults", 2: "Young Adults", 3: "Seniors", 4: "Babies", 5: "Seniors", 6: "Seniors"}

for x in range(len(trainingset["AgeBin"])):
    if trainingset["AgeBin"][x] == "Missing":
        trainingset["AgeBin"][x] = agemapping[trainingset["Title"][x]]
        
for x in range(len(testingset["AgeBin"])):
    if testingset["AgeBin"][x] == "Missing":
        testingset["AgeBin"][x] = agemapping[testingset["Title"][x]]
testingset.head()
#MAPPING AGE BINS INTO A NUMERIC VALUE
agemappings = {'Babies': 1, 'Children': 2, 'Teenagers': 3, 'Young Adults': 4, 'Adults': 5, 'Seniors': 6, 'Old': 7}
trainingset['AgeBin'] = trainingset['AgeBin'].map(agemappings)
testingset['AgeBin'] = testingset['AgeBin'].map(agemappings)
trainingset = trainingset.drop(['Age'], axis = 1)
testingset = testingset.drop(['Age'], axis = 1)
testingset.head()
#DROP NAMES BECAUSE THEY ARE OF NO USE ANYMORE
trainingset=trainingset.drop(['Name'],axis=1)
testingset=testingset.drop(['Name'],axis=1)
testingset.head()
#MAPPING SEX INTO A NUMERIC VALUE
sexmapping = {"male": 0, "female": 1}
trainingset['Sex'] = trainingset['Sex'].map(sexmapping)
testingset['Sex'] = testingset['Sex'].map(sexmapping)
testingset.head()
#MAPPING EMBARKED INTO A NUMERIC VALUE
embarkedmapping = {"S": 1, "C": 2, "Q": 3}
trainingset['Embarked'] = trainingset['Embarked'].map(embarkedmapping)
testingset['Embarked'] = testingset['Embarked'].map(embarkedmapping)
testingset.head()
#FILLING MISSING FARE VALUES AND MAPPING THEM INTO NUMERIC VALUES
#MISSING VALUE IS BASED ON THE CLASS OF THE PASSENGER
for x in range(len(testingset["Fare"])):
    if pd.isnull(testingset["Fare"][x]):
        pclass = testingset["Pclass"][x]
        testingset["Fare"][x] = round(trainingset[trainingset["Pclass"] == pclass]["Fare"].mean(), 4)
trainingset['FareBin'] = pd.qcut(trainingset['Fare'], 4, labels = [1, 2, 3, 4])
testingset['FareBin'] = pd.qcut(testingset['Fare'], 4, labels = [1, 2, 3, 4])
trainingset = trainingset.drop(['Fare'], axis = 1)
testingset = testingset.drop(['Fare'], axis = 1)
testingset.head()
from sklearn.model_selection import train_test_split
p = trainingset.drop(['Survived', 'PassengerId'], axis=1)
targetset = trainingset["Survived"]
x_train, x_val, y_train, y_val = train_test_split(p, targetset, test_size = 0.22, random_state = 0)
#USING SUPPORT VECTOR MACHINES
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,fbeta_score,make_scorer
svm = SVC(random_state=59090)
svm.fit(x_train,y_train)
preds=svm.predict(x_val)
accuracysvm = round(accuracy_score(y_val,preds)*100,2)
print(accuracysvm)
#USING DECISION TREE CLASSIFICTION
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_val)
accuracydt = round(accuracy_score(y_pred,y_val)*100,2)
print(accuracydt)
#USING RANDOM FOREST CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
rmfr = RandomForestClassifier()
rmfr.fit(x_train, y_train)
y_pred = rmfr.predict(x_val)
accuracyrf = round(accuracy_score(y_pred, y_val) * 100, 2)
print(accuracyrf)
#USING KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
accuracyknn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(accuracyknn)
ids = testingset['PassengerId']
predrmfr = rmfr.predict(testingset.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predrmfr })
output.to_csv('submission.csv', index=False)