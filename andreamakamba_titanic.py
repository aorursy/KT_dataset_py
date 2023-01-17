import pandas as pd 
import numpy as np 

from matplotlib import pyplot as pit 
import seaborn as sns 
%matplotlib inline 

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.model_selection import cross_val_score, GridSearchCV 
#Loading datasets 
import os 
os.path.realpath('.')

os.chdir(os.getcwd())

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()

test.head()
pd.plot.hist(train)
train[['Survived', 'Pclass']]
train['Survived'].count()
pd.unique(train['Survived'])
train.describe()
train.dtypes
pd.isna(train).sum()
train['Age']
train.dropna()
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train['Sex'] = lb.fit_transform(train['Sex']) 
targets = train.Survived
train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Survived'], axis = 1, inplace=True)
 

from sklearn.preprocessing import Imputer
im = Imputer()
predictors = im.fit_transform(train)

classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)
print(train.info())
train.drop(['Embarked'], axis = 1)
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)
train.dropna()
classifier=DecisionTreeClassifier()
classifier=classifier.fit(predictors,targets)
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)
train.head()
train.drop("Embarked", axis = 1, inplace = True)
train.head()
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)
train.dropna()
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)
train.fillna(train.mean())
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)
np.where(train.values >= np.finfo(np.float32).max)
train.isna().sum()
train = train.dropna()
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)
print(train)

train.reset_index(drop = True)
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)
train.head()
print(train)
train = train.reset_index(drop = True)
train.shape
targets.shape
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)
print(train)
targets = targets.dropna()
targets.shape
print(targets)
train = pd.read_csv("train.csv")
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train['Sex'] = lb.fit_transform(train['Sex'])
train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Survived', 'Embarked'], axis = 1, inplace=True)
count_age_embarked = len(train['Age'][ train.Age.isnull() ])
value_to_fill_age = train['Age'].dropna().mode().values
train['Age'][ train['Age'].isnull() ] = value_to_fill_age
lb2 = LabelEncoder()
train['Age'] = lb2.fit_transform(train['Age']) 
classifier=DecisionTreeClassifier()
classifier=classifier.fit(train,targets)

im = Imputer()
predictors = im.fit_transform(train)

classifier=DecisionTreeClassifier()
classifier=classifier.fit(predictors,targets)

test['Age'].fillna((test['Age'].mean()), inplace=True)
#Cleaning Test data 
lb3 = LabelEncoder()
test['Sex'] = lb3.fit_transform(test['Sex']) #male:1, female:0

count_null = len(test.Age[test.Age.isnull()])
value_to_fill = test.Age.dropna().mode().values
test['Age'][ test.Age.isnull() ] = value_to_fill
lb4 = LabelEncoder()
test['Age'] = lb4.fit_transform(test['Age']) 

test = test.drop(['Name', 'Embarked', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
im2 = Imputer()
test_predictors = im2.fit_transform(test)

test['Age']
age = test['Age']
print(age)
age.isnull().sum()
test = test.drop(['Name', 'Embarked', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
im2 = Imputer()
test_predictors = im2.fit_transform(test)

predictions=classifier.predict(test_predictors)
test_data = pd.read_csv("test.csv").values
result = np.c_[test_data[:,0].astype(int), predictions.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result.to_csv('res1.csv', index=False)
