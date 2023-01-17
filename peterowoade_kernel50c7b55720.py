#Loading the needed libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
#Importing the data

train = pd.read_csv("C:/Users/powoade/Downloads/Assignment/train_data.csv")

test = pd.read_csv("C:/Users/powoade/Downloads/Assignment/test_data.csv")
#Copying the train data

train_copy = train.copy()

test_copy = test.copy()
train.head()
test.head()
print("Train Shape :", train.shape, "\nTest Shape :", test.shape)
train.dtypes
train.isnull().sum()
train.describe().transpose()
train.nunique()
train_test = train + test
train_test.dtypes
test.describe().transpose()
#Seaborn stuff (saving kdeplot as peter_plot), so u can call peter_plot with other columns

def peter_plot(whatever):

    anything = sns.FacetGrid(train, hue = "Absent", aspect=5)

    anything.map(sns.kdeplot, whatever, shade = True)

    anything.set(xlim=(0, train[whatever].max()))

    anything.add_legend()
peter_plot("Age")
#Convert the values in Age column to a range(train)

def AgeNew(train):

    if train.Age <=26:

        return "0 - 26"

    elif (train.Age <=33):

        return "27 - 33"

    elif (train.Age <=42):

        return "33 - 42"

    elif (train.Age >42):

        return "Over 42"

train["AgeNew"] = train.apply(lambda train:AgeNew(train), axis=1)
#Convert the values in Age column to a range(test)

def AgeNew(test):

    if test.Age <=26:

        return "0 - 26"

    elif (test.Age <=33):

        return "27 - 33"

    elif (test.Age <=42):

        return "33 - 42"

    elif (test.Age >42):

        return "Over 42"

test["AgeNew"] = test.apply(lambda test:AgeNew(test), axis=1)
train.head(10)
peter_plot("Transportation expense")
def TransportationNew(train):

    if train["Transportation expense"] <=160:

        return "0 - 158"

    elif (train["Transportation expense"] <=190):

        return "159 - 190"

    elif (train["Transportation expense"] <=225):

        return "191 - 225"

    elif (train["Transportation expense"] >225):

        return "Over 225"

train["TransportationNew"] = train.apply(lambda train:TransportationNew(train), axis=1)
def TransportationNew(test):

    if test["Transportation expense"] <=160:

        return "0 - 158"

    elif (test["Transportation expense"] <=190):

        return "159 - 190"

    elif (test["Transportation expense"] <=225):

        return "191 - 225"

    elif (test["Transportation expense"] >225):

        return "Over 225"

test["TransportationNew"] = test.apply(lambda test:TransportationNew(test), axis=1)
train.head()
peter_plot("Distance from Residence to Work")
def DistanceNew(train):

    if train["Distance from Residence to Work"] <=25:

        return "0 - 25"

    elif (train["Distance from Residence to Work"] >25):

        return "Over 25"

train["DistanceNew"] = train.apply(lambda train:DistanceNew(train), axis=1)
def DistanceNew(test):

    if test["Distance from Residence to Work"] <=25:

        return "0 - 25"

    elif (test["Distance from Residence to Work"] >25):

        return "Over 25"

test["DistanceNew"] = test.apply(lambda test:DistanceNew(test), axis=1)
train.head()
peter_plot("Service time")
def ServicetimeNew(train):

    if train["Service time"] <=8:

        return "0 - 8"

    elif (train["Service time"] <=14):

        return "9 - 14"

    elif (train["Service time"] >14):

        return "Over 14"

train["ServicetimeNew"] = train.apply(lambda train:ServicetimeNew(train), axis=1)
def ServicetimeNew(test):

    if test["Service time"] <=8:

        return "0 - 8"

    elif (test["Service time"] <=14):

        return "9 - 14"

    elif (test["Service time"] >14):

        return "Over 14"

test["ServicetimeNew"] = test.apply(lambda test:ServicetimeNew(test), axis=1)
train.head(10)
peter_plot("Work load Average/day ")
def WorkloadNew(train):

    if train["Work load Average/day "] <=225000:

        return "0 - 225000"

    elif (train["Work load Average/day "] <=270000):

        return "225001 - 270000"

    elif (train["Work load Average/day "] >270000):

        return "Over 270000"

train["WorkloadNew"] = train.apply(lambda train:WorkloadNew(train), axis=1)
def WorkloadNew(test):

    if test["Work load Average/day "] <=225000:

        return "0 - 225000"

    elif (test["Work load Average/day "] <=270000):

        return "225001 - 270000"

    elif (test["Work load Average/day "] >270000):

        return "Over 270000"

test["WorkloadNew"] = test.apply(lambda test:WorkloadNew(test), axis=1)
train.head()
peter_plot("Hit target")
def HittargetNew(train):

    if train["Hit target"] <=93:

        return "0 - 93"

    elif (train["Hit target"] >93):

        return "Over 93"

train["HittargetNew"] = train.apply(lambda train:HittargetNew(train), axis=1)
def HittargetNew(test):

    if test["Hit target"] <=93:

        return "0 - 93"

    elif (test["Hit target"] >93):

        return "Over 93"

test["HittargetNew"] = test.apply(lambda test:HittargetNew(test), axis=1)
train.head()
peter_plot("Weight")
def WeightNew(train):

    if train["Weight"] <=58:

        return "0 - 58"

    elif (train["Weight"] <=74):

        return "59 - 74"

    elif (train["Weight"] <=81):

        return "75 - 81"

    elif (train["Weight"] <=91):

        return "82 - 91"

    elif (train["Weight"] >91):

        return "Over 91"

train["WeightNew"] = train.apply(lambda train:WeightNew(train), axis=1)
def WeightNew(test):

    if test["Weight"] <=58:

        return "0 - 58"

    elif (test["Weight"] <=74):

        return "59 - 74"

    elif (test["Weight"] <=81):

        return "75 - 81"

    elif (test["Weight"] <=91):

        return "82 - 91"

    elif (test["Weight"] >91):

        return "Over 91"

test["WeightNew"] = test.apply(lambda test:WeightNew(test), axis=1)
train.head()
peter_plot("Height")
def HeightNew(train):

    if train["Height"] <=168:

        return "0 - 168"

    elif (train["Height"] <=173):

        return "169 - 173"

    elif (train["Height"] >173):

        return "Over 173"

train["HeightNew"] = train.apply(lambda train:HeightNew(train), axis=1)
def HeightNew(test):

    if test["Height"] <=168:

        return "0 - 168"

    elif (test["Height"] <=173):

        return "169 - 173"

    elif (test["Height"] >173):

        return "Over 173"

test["HeightNew"] = test.apply(lambda test:HeightNew(test), axis=1)
peter_plot("Body mass index")
def BodymassNew(train):

    if train["Body mass index"] <=21:

        return "0 - 21"

    elif (train["Body mass index"] <=27):

        return "22 - 27"

    elif (train["Body mass index"] <=30):

        return "28 - 30"

    elif (train["Body mass index"] >30):

        return "Over 30"

train["BodymassNew"] = train.apply(lambda train:BodymassNew(train), axis=1)
def BodymassNew(test):

    if test["Body mass index"] <=21:

        return "0 - 21"

    elif (test["Body mass index"] <=27):

        return "22 - 27"

    elif (test["Body mass index"] <=30):

        return "28 - 30"

    elif (test["Body mass index"] >30):

        return "Over 30"

test["BodymassNew"] = test.apply(lambda test:BodymassNew(test), axis=1)
train.head()
test.head()
train.columns.to_list()
#Find correlation

train.corr()
drop = ["ID", "Transportation expense", "Distance from Residence to Work", "Service time", "Age", "Work load Average/day ", "Hit target", "Weight", "Height", "Body mass index"]

train = train.drop(drop, axis=1)

test = test.drop(drop, axis=1)
train.info()
test.info()
train.nunique()
#To group the nominal and ordinal variables

binary_grp = ["Disciplinary failure", "Social drinker", "Social smoker", "DistanceNew", "HittargetNew", "Education"]

multi_grp = ["Day of the week", "Month of absence", "Seasons", "Pet", "Son", "AgeNew", "TransportationNew", "ServicetimeNew", "WorkloadNew", "WeightNew", "HeightNew", "BodymassNew"]
#To apply Label Encoder (import from sklearn lib) and apply to binary group

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in binary_grp:

    train[i] = le.fit_transform(train[i])

train.dtypes
train = pd.get_dummies(data = train, columns = multi_grp)
train.head()
for i in binary_grp: test[i] = le.fit_transform(test[i]) 
test = pd.get_dummies(data = test, columns = multi_grp)
test.head()
train.columns.tolist()
test.columns.tolist()
#Import Values for SkLearn

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=12)

from sklearn.ensemble import RandomForestClassifier
#Put all other variables in features except Absent

features = train.drop("Absent", axis=1)
features.head()
#Set Absent as target

target = train.Absent
target.head()
#Split data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size= 0.30, random_state=30)
X_test.shape
clf = RandomForestClassifier(n_estimators=30)

scoring = "Accuracy"

score = cross_val_score(clf, features, target, cv=k_fold, n_jobs=1)

print("Model Accuracy is:", score)
round(np.mean(score)*100,2)
clf = RandomForestClassifier(n_estimators=30)

clf.fit(features, target)

prediction = clf.predict(test)
assignment = pd.DataFrame({

    "ID": test_copy["ID"],

    "Absent": prediction

})
assignment.to_csv("assignment.csv", index=False)
assignment2 = pd.DataFrame({

    "ID": test_copy["ID"],

    "Absent": prediction

})
assignment2.to_csv("assignment2.csv", index=True)