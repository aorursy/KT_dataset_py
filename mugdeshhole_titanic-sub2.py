# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test_1 = pd.read_csv("/kaggle/input/titanic/test.csv")

train
                      
                      
train.head()
train["Survived"].value_counts()
train[train["Survived"]==1]["Pclass"].value_counts().plot(kind="bar")
train["Sex"].value_counts()
train[train["Survived"]==1]["Sex"].value_counts().plot(kind="bar")
female = train[train.Sex=="female"][train.Survived==1]
male   = train[train.Sex=="male"][train.Survived==1]
rate_female = len(female)/len(train[train.Sex=="female"])
rate_male = len(male)/len(train[train.Sex=="male"])

print(rate_female)
print(rate_male)
train[train["Survived"]==1]["Embarked"].value_counts().plot(kind="bar")
fig,ax = plt.subplots()
plt.subplot(2,2,1)

print(sns.distplot(train["Age"]))
plt.subplot(2,2,2)
print(sns.distplot(train["Fare"]))
fig.set_size_inches(11.7, 8.27)
      


Title = set()
for name in train["Name"]:
    Title.add(name.split(',')[1].split('.')[0].strip())
print(Title)  

train["Title"] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
train.head()


    
Title = set()
for name in test["Name"]:
    Title.add(name.split(',')[1].split('.')[0].strip())
print(Title)  

test["Title"] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
test.head()
train_Age = train.groupby(["Sex","Title","Pclass"]).median()
grouped_median_train = train_Age.reset_index()[['Sex', 'Pclass', 'Title', 'Age','Fare']]

test.info()
test_Age = test.groupby(["Sex","Title","Pclass"]).median()
grouped_median_test = test_Age.reset_index()[['Sex', 'Pclass', 'Title', 'Age','Fare']]

train["Cabin"]
def age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) & 
        (grouped_median_train['Title'] == row['Title']) & 
        (grouped_median_train['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_train[condition]['Age'].values[0]


def process_age():
    global train
    train['Age'] = train.apply(lambda row: age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    return train

train = process_age()

train.Age.isnull().sum()
test.info()
def age(row):
    condition = (
        (grouped_median_test['Sex'] == row['Sex']) & 
        (grouped_median_test['Title'] == row['Title']) & 
        (grouped_median_test['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_test[condition]['Age'].values[0]


def process_age():
    global test
    test['Age'] = test.apply(lambda row: age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    
    return test

test = process_age()

test.info()
test
def fare(row):
    condition = (
        (grouped_median_test['Sex'] == row['Sex']) & 
        (grouped_median_test['Title'] == row['Title']) & 
        (grouped_median_test['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_test[condition]['Fare'].values[0]


def process_fare():
    global test
    test['Fare'] = test.apply(lambda row: fare(row) if np.isnan(row['Fare']) else row['Fare'], axis=1)
    
    return test

test = process_fare()

test.info()
Cabin=set()
train["Cabin"].replace(np.nan,"T",inplace=True)
for label in train:
    Cabin.add(label[0])
print(Cabin) 

train["Cabin"]=train['Cabin'].map(lambda label:label[0].strip())
train.info()


# train_cabin[train["Cabin"]=="T"].value_counts()
# train_cabin.isnull().sum()
Cabin=set()
test["Cabin"].replace(np.nan,"T",inplace=True)
for label in test:
    Cabin.add(label[0])
print(Cabin) 

test["Cabin"]=test['Cabin'].map(lambda label:label[0].strip())
test.info()
train.drop(["PassengerId","Name"],axis=1,inplace=True)
test.drop(["PassengerId","Name"],axis=1,inplace=True)
from sklearn.feature_selection import f_regression
x = train[["Age","Fare","Pclass"]]
y = train["Survived"]
f_regression(x,y)
train["Embarked"].replace(np.nan,"S",inplace=True)
test["Embarked"].replace(np.nan,"S",inplace=True)
train.isnull().sum()
x = train[["Age","Fare","Pclass","SibSp","Parch"]]
y = train["Survived"]
f_regression(x,y)
train.describe()
q = train["Fare"].quantile(0.98)
data = train[train["Fare"]<q]
data.describe(include="all")

train.drop(["Ticket"],axis=1,inplace=True)
test.drop(["Ticket"],axis=1,inplace=True)
test
data_dummies1 = pd.get_dummies(train["Pclass"],prefix="Pclass")
train.drop(["Pclass"],axis=1,inplace=True)
data_dummies1


data_dummies2 = pd.get_dummies(train["SibSp"],prefix="SibSp")
train.drop(["SibSp"],axis=1,inplace=True)
data_dummies2
data_dummies3 = pd.get_dummies(train["Parch"],prefix="Parch")
train.drop(["Parch"],axis=1,inplace=True)
data_dummies3
data_dummies4 = pd.get_dummies(train["Sex"],prefix="Sex")
train.drop(["Sex"],axis=1,inplace=True)
data_dummies4
data_dummies5 = pd.get_dummies(train["Title"],prefix="Title")
train.drop(["Title"],axis=1,inplace=True)
data_dummies5
data_dummies6 = pd.get_dummies(train["Embarked"],prefix="Embarked")
train.drop(["Embarked"],axis=1,inplace=True)
data_dummies6


data_dummies7 = pd.get_dummies(train["Cabin"],prefix="Cabin")
train.drop(["Cabin"],axis=1,inplace=True)
data_dummies7
data_dummies = pd.concat([data_dummies1,data_dummies2,data_dummies3,data_dummies4,data_dummies5,data_dummies6,data_dummies7],axis=1)
data_dummies
train_with_dummies = pd.concat([train,data_dummies],axis=1)
train_with_dummies
x = train_with_dummies.drop(["Survived"],axis=1)
y = train_with_dummies["Survived"]
f_regression(x,y)
 
 

train_with_dummies.isnull().sum()
x = train_with_dummies.drop(["Survived"],axis=1)
y = train_with_dummies["Survived"]
f_regression(x,y)
from sklearn.model_selection import train_test_split
x = train_with_dummies.drop(["Survived"],axis=1)
y = train_with_dummies["Survived"]
x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=365)
clf = RandomForestClassifier(n_jobs=2,random_state=0)
clf.fit(x,y)
preds = clf.predict(x_test)
preds
clf.score(x,y)
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25, 25))

test
Cabin=set()
test["Cabin"].replace(np.nan,"T",inplace=True)
for label in test:
    Cabin.add(label[0])
print(Cabin) 

test["Cabin"]=test['Cabin'].map(lambda label:label[0].strip())
test.info()
test.isnull().sum()

data_dummie1 = pd.get_dummies(test["Pclass"],prefix="Pclass")
test.drop(["Pclass"],axis=1,inplace=True)
data_dummie1



data_dummie2 = pd.get_dummies(test["SibSp"],prefix="Sibsp")
test.drop(["SibSp"],axis=1,inplace=True)
data_dummie2

data_dummie3 = pd.get_dummies(test["Parch"],prefix="Parch")
test.drop(["Parch"],axis=1,inplace=True)

data_dummie4 = pd.get_dummies(test["Title"],prefix="Title")
test.drop(["Title"],axis=1,inplace=True)



data_dummie5 = pd.get_dummies(test["Sex"],prefix="Sex")
test.drop(["Sex"],axis=1,inplace=True)
data_dummie5



data_dummie6 = pd.get_dummies(test["Embarked"],prefix="Embarked")
test.drop(["Embarked"],axis=1,inplace=True)



data_dummie7 = pd.get_dummies(test["Cabin"],prefix="Cabin")
test.drop(["Cabin"],axis=1,inplace=True)


data_dummies_test = pd.concat([data_dummie1,data_dummie2,data_dummie3,data_dummie4,data_dummie5,data_dummie6,data_dummie7],axis=1)
data_dummies_test
test_dummies = pd.concat([test,data_dummies_test],axis=1)
test_dummies


x_test = test_dummies
x_test.isnull().sum()                          
x_test.shape
predictions = clf.predict(x_test)
predictions
predictions.size
test_dummies["Survived"] = predictions
test
test[test["Survived"]==1]["Sex"].value_counts()
test[test["Survived"]==1]["Pclass"].value_counts()

output = pd.DataFrame({'PassengerId': test_1.PassengerId,
                       'Survived': predictions})
output.to_csv(r'C:\Users\ACER\Desktop\titan\submission.csv',index=False)
print(output)
df = pd.read_csv(r'C:\Users\ACER\Desktop\titan\submission.csv')
df
gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
gender
df
df["Survived"].sum()
gender["Survived"].sum()
test[test["Survived"] == 1]["Sex"].value_counts()
