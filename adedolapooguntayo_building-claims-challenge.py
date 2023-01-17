import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
#importing the data

train = pd.read_csv("C:/Users/aoguntayo/Desktop/DATA SCIENCE ASSIGNMENT/train_data.csv")

test = pd.read_csv("C:/Users/aoguntayo/Desktop/DATA SCIENCE ASSIGNMENT/test_data.csv") 
#copying the train data

train_copy = train.copy()

test_copy = test.copy()
#To show(print) the number of column and no of row in your data 

print ("Train_shape:", train.shape)

print ("Train_shape:", train.shape)
#To show the data types in the train data

train.dtypes
#To show the data types in the test data

test.dtypes
# To statistically decribe the data by showing (mean, median, mode,standard deviation,max.min)

train.describe().transpose()
# To statistically decribe the data by showing (mean, median, mode,standard deviation,max.min)

test.describe().transpose()
#To replace data use train['Absent'] = train[Absent].replace({0:'No', 1:'Yes'})
#To see the headings(number of rows and colums)

train.head()
#Write a code to plot graph with dolaplot

def dolaplot(feature):

        phaset = sns.FacetGrid(train, hue = 'Absent', aspect = 4)

        phaset.map(sns.kdeplot, feature, shade = True)

        phaset.set(xlim=(0, train[feature].max()))

        phaset.add_legend()
dolaplot('Age') 
#Creating a code to Group the Age data in train

def AgeDOLA(train):

    if train.Age <=26:

        return '0 - 26'

    elif (train.Age <=32):

        return '27 - 32'

    elif (train.Age <=42):

        return '33 - 42'

    elif train.Age >42:

        return 'Over 42'

train['Age Grouping'] = train.apply(lambda train:AgeDOLA(train), axis=1)
train.head()
#Creating a code to Group the Age data in test

def AgeDOLA(test):

    if test.Age <=26:

        return '0 - 26'

    elif (test.Age <=32):

        return '27 - 32'

    elif (test.Age <=42):

        return '33 - 42'

    elif test.Age >42:

        return 'Over 42'

test['Age Grouping'] = test.apply(lambda test:AgeDOLA(test), axis=1)
test.head()
#This is to plot the work load average over the day by calling the dolaplot function already created initially

dolaplot('Work load Average/day ') 
def WorkLoad(train):

    if train ['Work load Average/day '] <=22000:

        return '0 - 220000'

    elif (train['Work load Average/day '] <=26,500):

        return '220001 - 265000'

    elif (train['Work load Average/day '] <=33500):

        return '265001 - 335000'

    elif train['Work load Average/day '] >33500:

        return 'Over 335000'

train['WorkLoad'] = train.apply(lambda train:WorkLoad(train), axis=1)
train.head()
def WorkLoad(test):

    if test ['Work load Average/day '] <=22000:

        return '0 - 220000'

    elif (test['Work load Average/day '] <=26,500):

        return '220001 - 265000'

    elif (test['Work load Average/day '] <=33500):

        return '265001 - 335000'

    elif test['Work load Average/day '] >33500:

        return 'Over 335000'

test['WorkLoad'] = test.apply(lambda test:WorkLoad(test), axis=1)
test.head()
dolaplot('Transportation expense') 
def TransportCharge(train):

    if train['Transportation expense'] <=160 :

        return '0 - 160'

    elif (train['Transportation expense'] <=225):

        return '161 - 225'

    elif (train['Transportation expense'] <=320):

        return '226 - 320'

    elif (train['Transportation expense'] >320):

        return 'Over 320'

train['Transport Expense'] = train.apply(lambda train:TransportCharge(train), axis=1)
train.head()
def TransportCharge(test):

    if test['Transportation expense'] <=160 :

        return '0 - 160'

    elif (test['Transportation expense'] <=225):

        return '161 - 225'

    elif (test['Transportation expense'] <=320):

        return '226 - 320'

    elif (test['Transportation expense'] >320):

        return 'Over 320'

test['Transport Expense'] = test.apply(lambda test:TransportCharge(test), axis=1)
test.head()
dolaplot('Service time')
def Service_Time(train):

    if train['Service time'] <=7 :

        return '0 - 7'

    elif (train['Service time'] <=14):

        return '8 - 14'

    elif (train['Service time'] >14):

        return 'Over 14'

train['Service_Time'] = train.apply(lambda train:Service_Time(train), axis=1)
train.head()
def Service_Time(test):

    if test['Service time'] <=7 :

        return '0 - 7'

    elif (test['Service time'] <=14):

        return '8 - 14'

    elif (test['Service time'] >14):

        return 'Over 14'

test['Service_Time'] = test.apply(lambda test:Service_Time(test), axis=1)
test.head()
dolaplot('Height')
def Height_Grouped(train):

    if train['Height'] <=165:

        return '0 - 165'

    elif (train['Height'] <=173):

        return '165 - 173'

    elif (train['Height'] >173):

        return 'Over 173'

train['Height_Grouped'] = train.apply(lambda train:Height_Grouped(train), axis=1)
train.head()
def Height_Grouped(test):

    if test['Height'] <=165:

        return '0 - 165'

    elif (test['Height'] <=173):

        return '165 - 173'

    elif (test['Height'] >173):

        return 'Over 173'

test['Height_Grouped'] = test.apply(lambda test:Height_Grouped(test), axis=1)
test.head()
dolaplot('Weight') 
def WeightGroup(train):

    if train['Weight'] <=58:

        return '0 - 58'

    elif (train['Weight'] <=73):

        return '59 - 73'

    elif (train['Weight'] <=94):

        return '74 - 94'

    elif (train['Weight'] >94):

        return 'Over 94'

train['WeightGroup'] = train.apply(lambda train:WeightGroup(train), axis=1)
train.head()
def WeightGroup(test):

    if test['Weight'] <=58:

        return '0 - 58'

    elif (test['Weight'] <=73):

        return '59 - 73'

    elif (test['Weight'] <=94):

        return '74 - 94'

    elif (test['Weight'] >94):

        return 'Over 94'

test['WeightGroup'] = test.apply(lambda test:WeightGroup(test), axis=1)
test.head()
dolaplot('Hit target') 
def Hit_TargetGroup(train):

    if train['Hit target'] <=95:

        return '0 - 95'

    elif (train['Hit target'] >95):

        return 'Over 25'

train['Hit_TargetGroup'] = train.apply(lambda train:Hit_TargetGroup(train), axis=1)
train.head()
def Hit_TargetGroup(test):

    if test['Hit target'] <=95:

        return '0 - 95'

    elif (test['Hit target'] >95):

        return 'Over 25'

test['Hit_TargetGroup'] = test.apply(lambda test:Hit_TargetGroup(test), axis=1)
test.head()
dolaplot('Body mass index')
def BodymassGroup(train):

    if train['Body mass index'] <=21:

        return '0 - 21'

    elif (train['Body mass index'] <=31):

        return '22 - 31'

    elif (train['Body mass index'] >31):

        return 'Over 31'

train['BodymassGroup'] = train.apply(lambda train:BodymassGroup(train), axis=1)
train.head()
def BodymassGroup(test):

    if test['Body mass index'] <=21:

        return '0 - 21'

    elif (test['Body mass index'] <=31):

        return '22 - 31'

    elif (test['Body mass index'] >31):

        return 'Over 31'

test['BodymassGroup'] = test.apply(lambda test:BodymassGroup(test), axis=1)
test.head()
dolaplot('Distance from Residence to Work')
def Distance_from_ResidenceGroup(train):

    if train['Distance from Residence to Work'] <=25:

        return '0 - 25'

    elif (train['Distance from Residence to Work'] >25):

        return 'Over 25'

train['Distance_from_ResidenceGroup'] = train.apply(lambda train:Distance_from_ResidenceGroup(train), axis=1) 
train.head()
def Distance_from_ResidenceGroup(test):

    if test['Distance from Residence to Work'] <=25:

        return '0 - 25'

    elif (test['Distance from Residence to Work'] >25):

        return 'Over 25'

test['Distance_from_ResidenceGroup'] = test.apply(lambda test:Distance_from_ResidenceGroup(test), axis=1) 
test.head()
#This is to show correlation between the two data (train & test)

train.corr()
#This is to drop the initial heading from the source data in order to use the new grouped data

drop = ['ID', 'Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Hit target', 'Weight', 'Height', 'Body mass index']

train = train.drop(drop, axis=1)

test = test.drop(drop, axis=1)
train.info()
#This is to tell you how many differenet number of ranges or degrees in a  data set

train.nunique()
binary_grp = ['Disciplinary failure', 'Social drinker', 'Social smoker', 'WorkLoad', 'Distance_from_ResidenceGroup', 'Hit_TargetGroup']

multi_group = ['Reason for absence', 'Month of absence', 'Day of the week', 'Seasons', 'Education', 'Son', 'Pet', 'Age Grouping', 'Transport Expense', 'Service_Time', 'Height_Grouped', 'WeightGroup', 'BodymassGroup']
#This is the code that transforms all the data from ranges to integer data type for binary only

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in binary_grp:

    train[i] = le.fit_transform(train[i])

train.dtypes
train.head()
#Since computer only understands only binary variables 0&1 you need to turn other variable dummie.Dummie variables are used to replace a variable that has above two or more sublevels

train = pd.get_dummies(data = train, columns = multi_group)
#for every index in the binary group that we selected transform

for i in binary_grp: test[i] = le.fit_transform(test[i])
test.head()
test = pd.get_dummies(data = test, columns = multi_group)
train.info()
#This is to split the data to 10 

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=12)

from sklearn.ensemble import RandomForestClassifier
features = train.drop('Absent', axis=1)
target = train['Absent']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=30)
X_train.shape, y_train.shape
X_test.shape
clf = RandomForestClassifier(n_estimators=100)

scoring = 'Accuracy'

score = cross_val_score(clf, features, target, cv=k_fold, n_jobs=1)

print('Model Accuracy is: ', score)
round(np.mean(score)*100,2)
clf = RandomForestClassifier(n_estimators=100)

clf.fit(features, target)

prediction = clf.predict(test)
submission = pd.DataFrame({

    'ID': test_copy['ID'],

    'Absent': prediction

})
submission.to_csv('submission.csv', index=False)
submission = pd.DataFrame({

    'ID': test_copy['ID'],

    'Absent': prediction

})