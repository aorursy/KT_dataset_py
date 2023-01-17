#loading the needed libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#importing the data

train = pd.read_csv("C:/Users/oakinmade/Desktop/BTU Training/Machine Learning/assignment/train_data.csv") 

test = pd.read_csv("C:/Users/oakinmade/Desktop/BTU Training/Machine Learning/assignment/test_data.csv")
#copying the data

train_copy = train.copy()

test_copy = test.copy()
#xteristics of data set

print ("Train Shape :", train.shape)

print ("\nTest Shape :", test.shape) 
train.dtypes
test.dtypes
#creating train_test

train_test = train + test
train.head(10)
test.head(10)
train.describe().transpose()
test.describe().transpose()
print ("\n missing value in train data\n", train.isnull().sum())

print ("\n missing value in test data\n", test.isnull().sum())
print ("Nunmber of unique values: \n", train.nunique())
train.Son.value_counts()
def tolaplot(feature) :

    facet = sns.FacetGrid(train, hue ='Absent', aspect =4)

    facet.map(sns.kdeplot, feature, shade = True)

    facet.set(xlim=(0, train[feature].max()))

    facet.add_legend()
tolaplot ("Transportation expense")
def Tra(train):

    if train['Transportation expense'] <=160 :

        return '0 - 160'

    elif (train['Transportation expense'] <=225):

        return '161 - 225'

    elif (train['Transportation expense'] <=320):

        return '226 - 320'

    elif (train['Transportation expense'] >320):

        return 'Over 320'

train['Tra'] = train.apply(lambda train:Tra(train), axis=1)
def Tra(test):

    if test['Transportation expense'] <=160 :

        return '0 - 160'

    elif (test['Transportation expense'] <=225):

        return '161 - 225'

    elif (test['Transportation expense'] <=320):

        return '226 - 320'

    elif (test['Transportation expense'] >320):

        return 'Over 320'

test['Tra'] = test.apply(lambda test:Tra(test), axis=1)
tolaplot ("Service time")
def ServiceT(train):

    if train['Service time'] <=7 :

        return '0 - 7'

    elif (train['Service time'] <=14):

        return '8 - 14'

    elif (train['Service time'] >14):

        return 'Over 14'

train['ServiceT'] = train.apply(lambda train:ServiceT(train), axis=1)
def ServiceT(test):

    if test['Service time'] <=7 :

        return '0 - 7'

    elif (test['Service time'] <=14):

        return '8 - 14'

    elif (test['Service time'] >14):

        return 'Over 14'

test['ServiceT'] = test.apply(lambda test:ServiceT(test), axis=1)
tolaplot ("Age")
def AgeG(train):

    if train.Age <=26:

        return '0 - 26'

    elif (train.Age <=32):

        return '27 - 32'

    elif (train.Age <=42):

        return '33 - 42'

    elif train.Age >42:

        return 'Over 42'

train['AgeGroup'] = train.apply(lambda train:AgeG(train), axis=1)
def AgeG(test):

    if test.Age <=26:

        return '0 - 26'

    elif (test.Age <=32):

        return '27 - 32'

    elif (test.Age <=42):

        return '33 - 42'

    elif test.Age >42:

        return 'Over 42'

test['AgeGroup'] = test.apply(lambda test:AgeG(test), axis=1)
tolaplot ("Height")
def HeightG(train):

    if train['Height'] <=165:

        return '0 - 165'

    elif (train['Height'] <=173):

        return '166 - 173'

    elif (train['Height'] >173):

        return 'Over 173'

train['HeightG'] = train.apply(lambda train:HeightG(train), axis=1)
def HeightG(test):

    if test['Height'] <=165:

        return '0 - 165'

    elif (test['Height'] <=173):

        return '166 - 173'

    elif (test['Height'] >173):

        return 'Over 173'

test['HeightG'] = test.apply(lambda test:HeightG(test), axis=1)
tolaplot ("Distance from Residence to Work")
def DistanceToWork(train):

    if train['Distance from Residence to Work'] <=25:

        return '0 - 25'

    elif (train['Distance from Residence to Work'] >25):

        return 'Over 25'

train['DistanceToWork'] = train.apply(lambda train:DistanceToWork(train), axis=1)
def DistanceToWork(test):

    if test['Distance from Residence to Work'] <=25:

        return '0 - 25'

    elif (test['Distance from Residence to Work'] >25):

        return 'Over 25'

test['DistanceToWork'] = test.apply(lambda test:DistanceToWork(test), axis=1)
tolaplot ("Hit target")
def HTNew(train):

    if train['Hit target'] <=95:

        return '0 - 95'

    elif (train['Hit target'] >95):

        return 'Over 95'

train['HTNew'] = train.apply(lambda train:HTNew(train), axis=1)
def HTNew(test):

    if test['Hit target'] <=95:

        return '0 - 95'

    elif (test['Hit target'] >95):

        return 'Over 95'

test['HTNew'] = test.apply(lambda test:HTNew(test), axis=1) 
tolaplot ("Work load Average")
def WorkLoad(train):

    if train ['Work load Average'] <=220000:

        return '0 - 220000'

    elif (train['Work load Average'] <=265000):

        return '220001 - 265000'

    elif (train['Work load Average'] <=335000):

        return '265001 - 335000'

    elif train['Work load Average'] >335000:

        return 'Over 335000'

train['WorkLoad'] = train.apply(lambda train:WorkLoad(train), axis=1)
def WorkLoad(test):

    if test ['Work load Average'] <=220000:

        return '0 - 220000'

    elif (test['Work load Average'] <=265000):

        return '220001 - 265000'

    elif (test['Work load Average'] <=335000):

        return '265001 - 335000'

    elif test['Work load Average'] >335000:

        return 'Over 335000'

test['WorkLoad'] = test.apply(lambda test:WorkLoad(test), axis=1)
tolaplot ("Body mass index")
def Bmi(train):

    if train['Body mass index'] <=21:

        return '0 - 21'

    elif (train['Body mass index'] <=27):

        return '22 - 27'

    elif (train['Body mass index'] <=30):

        return '28 - 30'

    elif (train['Body mass index'] >30):

        return 'Over 30'

train['Bmi'] = train.apply(lambda train:Bmi(train), axis=1)
def Bmi(test):

    if test['Body mass index'] <=21:

        return '0 - 21'

    elif (test['Body mass index'] <=27):

        return '22 - 27'

    elif (test['Body mass index'] <=30):

        return '28 - 30'

    elif (test['Body mass index'] >30):

        return 'Over 30'

test['Bmi'] = test.apply(lambda test:Bmi(test), axis=1)
tolaplot ("Weight")
def WeightW(train):

    if train['Weight'] <=58:

        return '0 - 58'

    elif (train['Weight'] <=73):

        return '59 - 73'

    elif (train['Weight'] <=94):

        return '74 - 94'

    elif (train['Weight'] >94):

        return 'Over 94'

train['WeightW'] = train.apply(lambda train:WeightW(train), axis=1)
def WeightW(test):

    if test['Weight'] <=58:

        return '0 - 58'

    elif (test['Weight'] <=73):

        return '59 - 73'

    elif (test['Weight'] <=94):

        return '74 - 94'

    elif (test['Weight'] >94):

        return 'Over 94'

test['WeightW'] = test.apply(lambda test:WeightW(test), axis=1)
train.head()
test.head()
train.columns.tolist()
train.corr()
test.corr()
drop = ['ID','Weight', 'Height','Body mass index','Transportation expense',

 'Distance from Residence to Work','Service time','Age','Work load Average','Hit target']

train = train.drop(drop, axis=1)

test = test.drop(drop, axis=1)
train.info()
binary_grp = ['Disciplinary failure', 'Social drinker', 'Social smoker', 'Month of absence', 'Education', 'WorkLoad']

multi_group = ['Day of the week', 'Seasons', 'Son', 'Pet','Tra',

               'ServiceT','AgeGroup', 'HeightG', 'DistanceToWork', 'HTNew', 'WeightW', 'Bmi']
print ("Nunmber of unique values: \n", train.nunique())
print ("Nunmber of unique values: \n", test.nunique())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in binary_grp:

    train[i] = le.fit_transform(train[i])

train.dtypes
train = pd.get_dummies(data = train, columns = multi_group)
train.head(10)
for i in binary_grp:test[i] = le.fit_transform(test[i])
test = pd.get_dummies(data = test, columns = multi_group)
test.head(10)
train.info()
test.info()
#Validation Process
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=12)

from sklearn.ensemble import RandomForestClassifier
features = train.drop('Absent', axis=1)
target = train['Absent']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=30)
#70% of the data called Feature, target and the number of variables in each

X_train.shape, y_train.shape
#30% of the data called feature and the number of variables

X_test.shape
#Initiate RandomForest

clf = RandomForestClassifier(n_estimators=100)

scoring = 'accuracy'

#Initiate cross validation

score = cross_val_score(clf, features, target, cv=k_fold, n_jobs=1)

print("Model Accuracy is: ", score)
round(np.mean(score)*100,2)
clf = RandomForestClassifier(n_estimators=100)

clf.fit(features, target)

prediction = clf.predict (test)
Bond = pd.DataFrame({

    "ID": test_copy['ID'],

    "Absent": prediction

})
Bond.to_csv('Bond.csv', index=False)