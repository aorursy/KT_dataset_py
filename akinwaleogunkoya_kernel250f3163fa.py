import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
train = pd.read_csv('C:/Users/aogunkoya/Desktop/BTU TRAINING CAMP/Data Analytics/train_data.csv')

test = pd.read_csv('C:/Users/aogunkoya/Desktop/BTU TRAINING CAMP/Data Analytics/test_data.csv')
train_copy = train.copy()

test_copy = test.copy()
test.head(10)
print('Train Shape:', train.shape)

print('Test shape:', test.shape)
train.dtypes
train.describe().transpose()
test.describe().transpose()
#train['Absent'] = train['Absent'].replace({0:'No', 1:'Yes'})
train.head()
def waleplot(feature):

        phaset = sns.FacetGrid(train, hue = 'Absent', aspect = 4)

        phaset.map(sns.kdeplot, feature, shade = True)

        phaset.set(xlim=(0, train[feature].max()))

        phaset.add_legend()
waleplot('Age')
def AgeD(train):

    if train.Age <=26:

        return '0 - 26'

    elif (train.Age <=32):

        return '27 - 32'

    elif (train.Age <=42):

        return '33 - 42'

    elif train.Age >42:

        return 'Over 42'

train['AgeGroup'] = train.apply(lambda train:AgeD(train), axis=1) 
train.head()
def AgeD(test):

    if test.Age <=26:

        return '0 - 26'

    elif (test.Age <=32):

        return '27 - 32'

    elif (test.Age <=42):

        return '33 - 42'

    elif test.Age >42:

        return 'Over 42'

test['AgeGroup'] = test.apply(lambda test:AgeD(test), axis=1) 
test.head()
waleplot('Work load Average/day ')
def WorkLoad(train):

    if train ['Work load Average/day '] <=22000:

        return '0 - 22000'

    elif (train['Work load Average/day '] <=26,500):

        return '22001 - 26500'

    elif (train['Work load Average/day '] <=33500):

        return '26501 - 33500'

    elif train['Work load Average/day '] >33500:

        return 'Over 33500'

train['WorkLoad'] = train.apply(lambda train:WorkLoad(train), axis=1) 
train.head()
def WorkLoad(test):

    if test ['Work load Average/day '] <=22000:

        return '0 - 22000'

    elif (test['Work load Average/day '] <=26500):

        return '22001 - 26500'

    elif (test['Work load Average/day '] <=33500):

        return '26501 - 33500'

    elif test['Work load Average/day '] >33500:

        return 'Over 33500'

test['WorkLoad'] = test.apply(lambda test:WorkLoad(test), axis=1) 
test.head()
waleplot('Transportation expense')
def TransportCharge(train):

    if train['Transportation expense'] <=160 :

        return '0 - 160'

    elif (train['Transportation expense'] <=225):

        return '161 - 225'

    elif (train['Transportation expense'] <=320):

        return '226 - 320'

    elif (train['Transportation expense'] >320):

        return 'Over 320'

train['TransportCharge'] = train.apply(lambda train:TransportCharge(train), axis=1) 
def TransportCharge(test):

    if test['Transportation expense'] <=160 :

        return '0 - 160'

    elif (test['Transportation expense'] <=225):

        return '161 - 225'

    elif (test['Transportation expense'] <=320):

        return '226 - 320'

    elif (test['Transportation expense'] >320):

        return 'Over 320'

test['TransportCharge'] = test.apply(lambda test:TransportCharge(test), axis=1) 
train.head()
waleplot('Service time')
def Service_Time(train):

    if train['Service time'] <=7 :

        return '0 - 7'

    elif (train['Service time'] <=14):

        return '8 - 14'

    elif (train['Service time'] >14):

        return 'Over 14'

train['Service_Time'] = train.apply(lambda train:Service_Time(train), axis=1) 
def Service_Time(test):

    if test['Service time'] <=7:

        return '0 - 7'

    elif (test['Service time'] <=14):

        return '8 - 14'

    elif (test['Service time'] >14):

        return 'Over 14'

test['Service_Time'] = test.apply(lambda test:Service_Time(test), axis=1) 
test.head()
waleplot('Height')
def Height_New(train):

    if train['Height'] <=165:

        return '0 - 165'

    elif (train['Height'] <=173):

        return '165 - 173'

    elif (train['Height'] >173):

        return 'Over 173'

train['Height_New'] = train.apply(lambda train:Height_New(train), axis=1) 
def Height_New(test):

    if test['Height'] <=165:

        return '0 - 165'

    elif (test['Height'] <=173):

        return '166 - 173'

    elif (test['Height'] >173):

        return 'Over 173'

test['Height_New'] = test.apply(lambda test:Height_New(test), axis=1) 
train.head()
waleplot('Weight')
def Weight_New(train):

    if train['Weight'] <=58:

        return '0 - 58'

    elif (train['Weight'] <=73):

        return '59 - 73'

    elif (train['Weight'] <=94):

        return '74 - 94'

    elif (train['Weight'] >94):

        return 'Over 94'

train['Weight_New'] = train.apply(lambda train:Weight_New(train), axis=1)
def Weight_New(test):

    if test['Weight'] <=58 :

        return '0 - 58'

    elif (test['Weight'] <=73):

        return '59 - 73'

    elif (test['Weight'] <=94):

        return '74 - 94'

    elif (test['Weight'] >94):

        return 'Over 94'

test['Weight_New'] = test.apply(lambda test:Weight_New(test), axis=1) 
train.head()
waleplot('Body mass index')
def BodymassNew(train):

    if train['Body mass index'] <=21:

        return '0 - 21'

    elif (train['Body mass index'] <=31):

        return '22 - 31'

    elif (train['Body mass index'] >31):

        return 'Over 31'

train['BodymassNew'] = train.apply(lambda train:BodymassNew(train), axis=1) 
def BodymassNew(test):

    if test['Body mass index'] <=21:

        return '0 - 21'

    elif (test['Body mass index'] <=31):

        return '22 - 31'

    elif (test['Body mass index'] >31):

        return 'Over 31'

test['BodymassNew'] = test.apply(lambda test:BodymassNew(test), axis=1) 
train.head()
waleplot('Distance from Residence to Work')
def Distance_to_Work(train):

    if train['Distance from Residence to Work'] <=25:

        return '0 - 25'

    elif (train['Distance from Residence to Work'] >25):

        return 'Over 25'

train['Distance_to_Work'] = train.apply(lambda train:Distance_to_Work(train), axis=1) 
def Distance_to_Work(test):

    if test['Distance from Residence to Work'] <=25:

        return '0 - 25'

    elif (test['Distance from Residence to Work'] >25):

        return 'Over 25'

test['Distance_to_Work'] = test.apply(lambda test:Distance_to_Work(test), axis=1) 
train.head()
waleplot('Hit target')
def Hit_TargetNew(train):

    if train['Hit target'] <=95:

        return '0 - 95'

    elif (train['Hit target'] >95):

        return 'Over 25'

train['Hit_TargetNew'] = train.apply(lambda train:Hit_TargetNew(train), axis=1) 
def Hit_TargetNew(test):

    if test['Hit target'] <=95:

        return '0 - 95'

    elif (test['Hit target'] >95):

        return 'Over 95'

test['Hit_TargetNew'] = test.apply(lambda test:Hit_TargetNew(test), axis=1) 
train.head()
train.corr()
drop = ['ID', 'Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Hit target', 'Weight', 'Height', 'Body mass index']

train = train.drop(drop, axis=1)

test = test.drop(drop, axis=1)
train.info()
train.nunique()
binary_grp = ['Disciplinary failure', 'Social drinker', 'Social smoker', 'WorkLoad', 'Distance_to_Work', 'Hit_TargetNew']

multi_group = ['Reason for absence', 'Month of absence', 'Day of the week', 'Seasons', 'Education', 'Son', 'Pet', 'AgeGroup', 'TransportCharge', 'Service_Time', 'Height_New', 'Weight_New', 'BodymassNew']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in binary_grp:

    train[i] = le.fit_transform(train[i])

train.dtypes
train.head()
train = pd.get_dummies(data = train, columns = multi_group)
for i in binary_grp: test[i] = le.fit_transform(test[i])
test.head()
test = pd.get_dummies(data = test, columns = multi_group)
train.info()
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
submi = pd.DataFrame({

    'ID': test_copy['ID'],

    'Absent': prediction

})
submi.to_csv('submi.csv', index=False)
submi = pd.DataFrame({

    'ID': test_copy['ID'],

    'Absent': prediction

})