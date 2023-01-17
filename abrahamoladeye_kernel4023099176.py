import numpy as np

import pandas as pd

import matplotlib.pyplot as Plt

%matplotlib inline

import seaborn as sns

sns.set()
train_data = pd.read_csv('Downloads/train_data.csv')

test_data = pd.read_csv('Downloads/test_data.csv')
train_dataC = train_data.copy()

test_dataC = test_data.copy()
train_data.head()
test_data.head()
print('Train Data Shape', train_data.shape)

print('Test Data Shape', test_data.shape)
train_data.describe().transpose()
train_data.columns.tolist()
train_data.dtypes
train_data.isnull().sum()
train_data.describe().transpose()
train_test_data = train_data + test_data

train_test_data.dtypes
train_data.nunique()
train_data.columns.tolist()
def tammyplot(feature):

    facet = sns.FacetGrid(train_data, hue = 'Absent', aspect =4)

    facet.map(sns.kdeplot, feature, shade = True)

    facet.set(xlim=(0, train_data[feature].max()))

    facet.add_legend()
tammyplot('Weight')
def WeightG(train_data):

    if train_data.Weight <=58:

        return "0 - 58"

    elif (train_data.Weight <=73):

        return "59 - 73"

    elif (train_data.Weight <=82):

        return "74 - 82"

    elif (train_data.Weight <=94):

        return "83 - 94"

    else:

        return "Over 94"

train_data["WeightGroup"] = train_data.apply(lambda train_data:WeightG(train_data), axis=1) 
def WeightG(test_data):

    if test_data.Weight <=58:

        return "0 - 58"

    elif (test_data.Weight <=73):

        return "59 - 73"

    elif (test_data.Weight <=82):

        return "74 - 82"

    elif (test_data.Weight <=94):

        return "83 - 94"

    else:

        return "Over 94"

test_data["WeightGroup"] = test_data.apply(lambda test_data:WeightG(test_data), axis=1) 
tammyplot('Age')
def AgeG(train_data):

    if train_data.Age <=26:

        return "0 - 26"

    elif (train_data.Age <=30):

        return "27 - 30"

    elif (train_data.Age <=32):

        return "31 - 32"

    elif (train_data.Age <=42):

        return "33 - 42"

    else:

        return "Over 42"

train_data["AgeGroup"] = train_data.apply(lambda train_data:AgeG(train_data), axis=1) 
def AgeG(test_data):

    if test_data.Age <=26:

        return "0 - 26"

    elif (test_data.Age <=30):

        return "27 - 30"

    elif (test_data.Age <=32):

        return "31 - 32"

    elif (test_data.Age <=42):

        return "33 - 42"

    else:

        return "Over 42"

test_data["AgeGroup"] = test_data.apply(lambda test_data:AgeG(test_data), axis=1) 
tammyplot('Body mass index')
def BMIG(train_data):

    if train_data['Body mass index'] <=21:

        return "0 - 21"

    elif (train_data['Body mass index'] <=27):

        return "22 - 27"

    elif (train_data['Body mass index'] <=30):

        return "28 - 30"

    else:

        return "Over 30"

train_data["BMIGroup"] = train_data.apply(lambda train_data:BMIG(train_data), axis=1) 
def BMIG(test_data):

    if test_data['Body mass index'] <=21:

        return "0 - 21"

    elif (test_data['Body mass index'] <=27):

        return "22 - 27"

    elif (test_data['Body mass index'] <=30):

        return "28 - 30"

    else:

        return "Over 30"

test_data["BMIGroup"] = test_data.apply(lambda test_data:BMIG(test_data), axis=1)
tammyplot('Transportation expense')
def TEG(train_data):

    if train_data['Transportation expense'] <=160:

        return "0 - 160"

    elif (train_data['Transportation expense'] <=190):

        return "161 - 190"

    elif (train_data['Transportation expense'] <=226):

        return "191 - 226"

    elif (train_data['Transportation expense'] <=315):

        return "227 - 315"

    else:

        return "Over 315"

train_data["Transport_Expense_Group"] = train_data.apply(lambda train_data:TEG(train_data), axis=1) 
def TEG(test_data):

    if test_data['Transportation expense'] <=160:

        return "0 - 160"

    elif (test_data['Transportation expense'] <=190):

        return "161 - 190"

    elif (test_data['Transportation expense'] <=226):

        return "191 - 226"

    elif (test_data['Transportation expense'] <=315):

        return "227 - 315"

    else:

        return "Over 315"

test_data["Transport_Expense_Group"] = test_data.apply(lambda test_data:TEG(test_data), axis=1) 
tammyplot('Distance from Residence to Work')
def DFRTWG(train_data):

    if train_data['Distance from Residence to Work'] <=26:

        return "0 - 26"

    else:

        return "Over 26"

train_data["DFRTWGroup"] = train_data.apply(lambda train_data:DFRTWG(train_data), axis=1)
def DFRTWG(test_data):

    if test_data['Distance from Residence to Work'] <=26:

        return "0 - 26"

    else:

        return "Over 26"

test_data["DFRTWGroup"] = test_data.apply(lambda test_data:DFRTWG(test_data), axis=1)
tammyplot('Service time')
def STG(train_data):

    if train_data['Service time'] <=4:

        return "0 - 4"

    elif (train_data['Service time'] <=14):

        return "5 - 14"

    else:

        return "Over 14"

train_data["Service_TimeGroup"] = train_data.apply(lambda train_data:STG(train_data), axis=1) 
def STG(test_data):

    if test_data['Service time'] <=4:

        return "0 - 4"

    elif (test_data['Service time'] <=14):

        return "5 - 14"

    else:

        return "Over 14"

test_data["Service_TimeGroup"] = test_data.apply(lambda test_data:STG(test_data), axis=1) 
tammyplot('Height')
def HeightG(train_data):

    if train_data.Height <=167:

        return "0 - 167"

    elif (train_data.Height <=171):

        return "168 - 171"

    else:

        return "Over 171"

train_data["Height_Group"] = train_data.apply(lambda train_data:HeightG(train_data), axis=1) 
def HeightG(test_data):

    if test_data.Height <=167:

        return "0 - 167"

    elif (test_data.Height <=171):

        return "168 - 171"

    else:

        return "Over 171"

test_data["Height_Group"] = test_data.apply(lambda test_data:HeightG(test_data), axis=1) 
tammyplot('Work load Average/day ')
def WLAG(train_data):

    if train_data['Work load Average/day '] <=220000:

        return "0 - 224000"

    elif (train_data['Work load Average/day '] <=270000):

        return "220001 - 270000"

    else:

        return "Over 270000"

train_data["Work_LoadGroup"] = train_data.apply(lambda train_data:HeightG(train_data), axis=1) 
def WLAG(test_data):

    if test_data['Work load Average/day '] <=220000:

        return "0 - 224000"

    elif (test_data['Work load Average/day '] <=270000):

        return "220001 - 270000"

    else:

        return "Over 270000"

test_data["Work_LoadGroup"] = test_data.apply(lambda test_data:HeightG(test_data), axis=1) 
train_data.head()
tammyplot('Hit target')
def HTG(train_data):

    if train_data['Hit target'] <=92:

        return "0 - 92"

    else:

        return "Over 92"

train_data["Hit_TargetGroup"] = train_data.apply(lambda train_data:HTG(train_data), axis=1) 
def HTG(test_data):

    if test_data['Hit target'] <=92:

        return "0 - 92"

    else:

        return "Over 92"

test_data["Hit_TargetGroup"] = test_data.apply(lambda test_data:HTG(test_data), axis=1) 
train_data.head()
train_data.Height_Group.value_counts()
drop = ['ID', 'Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Hit target', 'Weight', 'Height', 'Body mass index']

train_data = train_data.drop(drop, axis=1)

test_data = test_data.drop(drop, axis=1)
test_data.nunique()
binary_grp = ['Disciplinary failure', 'Height_Group','Social drinker','Month of absence', 'Social smoker', 'DFRTWGroup', 'Hit_TargetGroup', 'Son', 'Pet', 'WeightGroup','AgeGroup','BMIGroup','Transport_Expense_Group', 'Service_TimeGroup', 'Work_LoadGroup','Education', 'Son', 'Pet']

multiple_grp = ['Reason for absence','Day of the week', 'Seasons']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in binary_grp:

    train_data[i] = le.fit_transform(train_data[i])

train_data.dtypes
train_data = pd.get_dummies(data = train_data, columns = multiple_grp)
train_data.head()
for i in binary_grp:

    test_data[i] = le.fit_transform(test_data[i])
test_data = pd.get_dummies(data = test_data, columns = multiple_grp)
test_data.head()
train_data.head()
test_data.columns.tolist()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=12)

from sklearn.ensemble import RandomForestClassifier
features = train_data.drop('Absent', axis=1)
target = train_data['Absent']
from sklearn.model_selection import train_test_split

X_train_data, X_test_data, y_train_data,y_test_data = train_test_split(features, target, test_size=0.30, random_state=30)
X_train_data.shape, y_train_data.shape, X_test_data.shape
clf = RandomForestClassifier(n_estimators=50)

scoring = 'accuracy'

score = cross_val_score(clf, features, target, cv=k_fold, n_jobs=1)

print("Model Accuracy is: ", score)
round(np.mean(score)*100,2)
clf = RandomForestClassifier(n_estimators=50)

clf.fit(features,target)

prediction = clf.predict(test_data)
submite= pd.DataFrame({

    "ID": test_dataC['ID'],

    "Absent": prediction

})
sub.to_csv('submite.csv', index=False)
test_dataC.dtypes