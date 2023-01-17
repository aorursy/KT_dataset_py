import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
train = pd.read_csv('C:/Users/defaultuser0/Downloads/Tammy Enterprise Assignment/train_data.csv')

test= pd.read_csv('C:/Users/defaultuser0/Downloads/Tammy Enterprise Assignment/test_data.csv')
#copying the train data

train_copy = train.copy()

test_copy = test.copy()
train.head()
test.head()
print('Train Shape :', train.shape)

print('\nTest Shape :', test.shape)
test.dtypes
print('Train Shape :', train.shape)

print('\nTest Shape :', test.shape)
test.dtypes
test.isnull().sum()
train.describe().transpose()
print('\n Missing Values in train data\n', train.isnull().sum())

print('\n Missing Values in Test data\n', train.isnull().sum())
print('Number of unique values:\n', train.nunique())
def lolaplot(feature):

    facet = sns.FacetGrid(train, hue = 'Absent', aspect =4)

    facet.map(sns.kdeplot, feature, shade = True)

    facet.set(xlim=(0,train[feature].max()))

    facet.add_legend()
lolaplot('Transportation expense')
def TransportE(train):

    if train['Transportation expense'] <=160 :

        return '0 - 160'

    elif (train['Transportation expense'] <=190) :

        return '161 - 190'

    elif (train['Transportation expense'] <=230) :

        return '191 - 230'

    elif (train['Transportation expense'] <=320) :

        return '231 - 320'

    elif train['Transportation expense'] >320:

        return 'Over 320'

    

train['TransportE'] = train.apply(lambda train:TransportE(train), axis=1)
train.head()
def TransportE(test):

    if test['Transportation expense'] <=160:

        return '0 - 160'

    elif (test['Transportation expense'] <=190):

        return '161 - 190'

    elif (test['Transportation expense'] <=230):

        return '191 - 230'

    elif (test['Transportation expense'] <=320):

        return '231 - 320'

    elif test['Transportation expense'] >320:

        return 'Over 320'

test['TransportE'] = test.apply(lambda test: TransportE(test),axis=1)
test.head()
lolaplot ('Distance from Residence to Work')
def DistanceW(train):

    if train['Distance from Residence to Work'] <=25:

        return '0 -25'

    elif train['Distance from Residence to Work'] >25:

        return 'Over 25'

    

train['DistanceW'] = train.apply(lambda train: DistanceW(train),axis=1)
train.head()
def DistanceW(test):

    if test['Distance from Residence to Work'] <=25:

        return '0 -25'

    elif test['Distance from Residence to Work'] >25:

        return 'Over 25'

    

test['DistanceW'] = test.apply(lambda test: DistanceW(test),axis=1)
test.head()
lolaplot ('Service time')
def ServiceT(train):

    if train['Service time'] <=4:

        return '0 - 4'

    elif (train['Service time'] <=8):

        return '5 - 8'

    elif (train['Service time'] <=14):

        return '9 - 14'

    elif train ['Service time'] >14:

        return 'Over 14'

    

train['ServiceT'] = train.apply(lambda train: ServiceT(train),axis=1)
train.head()
def ServiceT(test):

    if test['Service time'] <=4:

        return '0 - 4'

    elif (test['Service time'] <=8):

        return '5 - 8'

    elif (test['Service time'] <=14):

        return '9 - 14'

    elif test['Service time'] >14:

        return 'Over 14'

       

test['ServiceT']= test.apply(lambda test:ServiceT(test),axis=1)
test.head()
lolaplot ('Age')
def AgeGroup(train):

    if train['Age'] <=26:

        return '0 - 26'

    elif (train['Age'] <=30):

        return '27 - 30'

    elif (train['Age'] <=32):

        return '31 - 32'

    elif (train['Age'] <=43):

        return '33 - 43'

    elif train['Age'] >43:

        return 'Over 43'

    

train['AgeGroup'] = train.apply(lambda train: AgeGroup(train),axis=1)
train.head()
def AgeGroup(test):

    if test['Age'] <=26:

        return '0 - 26'

    elif (test['Age'] <=30):

        return '27 - 30'

    elif (test['Age'] <=32):

        return '31 - 32'

    elif (test['Age'] <=43):

        return '33 - 43'

    elif test['Age'] >43:

        return 'Over 43'

    

    

test['AgeGroup'] = test.apply(lambda test: AgeGroup(test),axis=1)
test.head()
lolaplot ("Body mass index")
def BMI(train):

    if train['Body mass index'] <=21:

        return '0 - 21'

    elif (train['Body mass index'] <=27):

        return '22 - 27'

    elif (train['Body mass index'] <=30):

        return '28 - 30'

    elif train['Body mass index'] >30:

        return 'Over 30'

    

train['BMI'] = train.apply(lambda train: BMI(train),axis=1)
train.head()
def BMI(test):

    if test['Body mass index'] <=21:

        return '0 - 21'

    elif (test['Body mass index'] <=27):

        return '22 - 27'

    elif (test['Body mass index'] <=30):

        return '28 - 30'

    elif test['Body mass index'] >30:

        return 'Over 30'

    

test['BMI'] = test.apply(lambda test: BMI(test),axis=1)
test.head()
lolaplot ('Work load Average/day ')
def WorkLoad(train):

    if train['Work load Average/day '] <=225000:

        return '0 - 225000'

    elif (train['Work load Average/day '] <=275000):

        return '225001 - 275000'

    elif (train['Work load Average/day '] <=340000):

        return '275001 - 340000'

    elif train['Work load Average/day '] >340000:

        return 'Over 340000'

    

train['WorkLoad'] = train.apply(lambda train: WorkLoad(train),axis=1)
train.head()
def WorkLoad(test):

    if test['Work load Average/day '] <=225000:

        return '0 - 225000'

    elif (test['Work load Average/day '] <=275000):

        return '225001 - 275000'

    elif (test['Work load Average/day '] <=340000):

        return '275001 - 340000'

    elif test['Work load Average/day '] >340000:

        return 'Over 340000'

    

test['WorkLoad'] = test.apply(lambda test: WorkLoad(test),axis=1)
test.head()
train.columns.to_list()
train.corr()
drop = ['ID', 'Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Body mass index', 'Work load Average/day ']

train = train.drop(drop, axis=1)

test = test.drop(drop, axis=1)
train.columns.tolist()
train.info()
train.nunique()
binary_grp = ['Disciplinary failure','DistanceW','Social drinker', 'Social smoker', 'Pet', 'Son', 'BMI', 'Education', 'WorkLoad', 'Hit target']

multi_group = ['Seasons','Month of absence','Day of the week','AgeGroup', 'ServiceT', 'TransportE']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in binary_grp:

    train [i] = le.fit_transform(train[i])

train.dtypes
train=pd.get_dummies(data = train, columns = multi_group)
train.head()
for i in binary_grp:

    test [i] = le.fit_transform(test[i])
test = pd.get_dummies(data = test, columns = multi_group)
test.head()
test.info()
train.info()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=12)

from sklearn.ensemble import RandomForestClassifier
Features = train.drop('Absent', axis=1)
target = train['Absent']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Features, target, test_size=0.30, random_state=30)
X_train.shape, y_train.shape
X_test.shape
clf = RandomForestClassifier(n_estimators=100)

scoring = 'accuracy'

score = cross_val_score(clf, Features, target, cv=k_fold, n_jobs=1)

print('Model Accuracy is: ', score)
round(np.mean(score)*100,2)
clf = RandomForestClassifier(n_estimators=100)

clf.fit(Features, target)

prediction = clf.predict(test)
submi = pd.DataFrame({

    'ID':test_copy['ID'],

    'Absent': prediction

})
submi.to_csv('submi.csv', index=False)
submi = pd.DataFrame({

    'ID': test_copy['ID'],

    'Absent': prediction

})