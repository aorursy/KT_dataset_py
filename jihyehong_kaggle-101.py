# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#import libraries

import pandas as pd

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

#from scipy.stats import norm

#from sklearn.preprocessing import StandardScaler

#from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(10)
train.columns
test.columns
# Drop unnecessary columns

train = train.drop(['PassengerId','Name','Ticket'],axis=1)

test = test.drop(['PassengerId','Name','Ticket'],axis=1)
train.info()

print('---')

test.info()
# Draw a bar chart for the label 'survived'

f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y="Survived", data=train, color="c");
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y="Sex", data=train, color="c");
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y="Embarked", data=train, color="c");
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y="Survived", data=train, color="c");
train.shape, test.shape
train.isnull().values.any(), test.isnull().values.any()
train[pd.isnull(train).any(axis=1)]
train.Embarked.unique()
train.Age.unique()
# Fill Null values of 'Age' as Median

train['Age'] = train['Age'].fillna(train['Age'].median())
train['Cabin'] = train['Cabin'].fillna('0')
train.Cabin.unique()
train['Cabin'][train.Cabin!='0']='1'
# Define 'Cabin' feature as binary class, cabin yes = 1, no = 0

train['Cabin'].unique()
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y="Cabin", data=train, color="c");
train['Pclass'] = train['Pclass'].astype(object)
train.Pclass.unique()
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y="Pclass", data=train, color="c");
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
train['famSize'] = train.SibSp + train.Parch
# drop 'SibSp' and 'Parch' columns

train.drop(['SibSp','Parch'], axis=1)
# Add a new feature 'alone' if the passanger had family aboard on Titanic

train['family'] = np.where(train['famSize']==0, 'N', 'Y')
f, ax = plt.subplots(figsize=(7, 6))

sns.stripplot(x="Survived", y="famSize", data=train);
temp = train.loc[train.Sex=='female']

temp = temp.loc[temp.Pclass==2]

temp.head()
temp = train[['Age','Fare','Survived']]
temp[temp.Survived==0].Age.head()
# Add a new feature 'alone' if the passanger had family aboard on Titanic

train['AgeGroup'] = '0'

train['AgeGroup'] = np.where(train['Age']<=16, 'Child', 'Adult')
f, ax = plt.subplots(figsize=(7, 6))

sns.countplot(x="family", hue="Survived", data=train, palette="Greens_d");
f, ax = plt.subplots(figsize=(7, 6))

sns.countplot(x="famSize", hue="Survived", data=train, palette="Greens_d");
f, ax = plt.subplots(figsize=(7, 4))

sns.countplot(x="Pclass", hue="Survived", data=train, palette="Greens_d");
f, ax = plt.subplots(figsize=(7, 5))

sns.countplot(x="Sex", hue="Survived", data=train, palette="Greens_d");
f, ax = plt.subplots(figsize=(7, 5))

sns.countplot(x="Cabin", hue="Survived", data=train, palette="Greens_d");
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y="Embarked", data=train, color="c");
f, ax = plt.subplots(figsize=(7, 5))

sns.countplot(x="Embarked", hue="Survived", data=train, palette="Greens_d");
f, ax = plt.subplots(figsize=(7, 5))

sns.countplot(x="AgeGroup", hue="Survived", data=train, palette="Greens_d");
from sklearn.utils import class_weight

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, LabelEncoder

 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
print(__doc__)



import itertools

from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
train.columns
X = train.drop(['Survived','SibSp','Parch','Embarked','Age','Fare','famSize'],axis=1) #Age, Fare

Y = train.Survived
X.columns
X.Pclass.unique(), X.Sex.unique(), X.Cabin.unique(), X.family.unique(), X.AgeGroup.unique()
X['Sex'] = np.where(train['Sex']=='female', 0, 1)

X['Cabin'] = np.where(train['Cabin']=='0', 0, 1)

X['family'] = np.where(train['family']=='Y', 1, 0)

X['AgeGroup'] = np.where(train['AgeGroup']=='Adult', 0, 1)
test.columns
X_train = X

Y_train = Y

X_test = test.drop
X.shape, Y.shape
# Validation Set approach

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)



accuracyList = {}



clf = tree.DecisionTreeClassifier(class_weight="balanced")

gaussian = GaussianNB()

logreg = LogisticRegression(class_weight="balanced")

boost = GradientBoostingClassifier()

knn = KNeighborsClassifier(n_neighbors = 3)

forest = RandomForestClassifier(n_estimators = 20)



models = [clf,gaussian,logreg,boost,knn,forest]



for model in models:

    accuracyList[model] = 0

    

for model in models:

    modelV = model.fit(X_train, Y_train)

    Y_pred = modelV.predict(X_test)

    accuracyList[model] = round(modelV.score(X_test,Y_test)*100,3)



#models = [clf,gaussian,logreg,boost,knn,forest]

accResult = pd.DataFrame({

    'Model': ['Random Forest', 'KNN', 'Naive Bayes', 

              'Logistic Regression', 'Decision Tree', 'Gradient Boosting'],

    'Score': [accuracyList[forest], accuracyList[knn], accuracyList[gaussian], 

              accuracyList[logreg], accuracyList[clf], accuracyList[boost]]})

accResult.sort_values(by='Score', ascending=False)
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
trainOrg = df_train.copy()

testOrg = df_test.copy()



trainOrg = trainOrg.drop(['PassengerId','Name','Ticket'], axis=1)

testOrg = testOrg.drop(['PassengerId','Name','Ticket'], axis=1)

trainOrg.head(5)
# Fill Null values of train set

trainOrg['Age'] = trainOrg['Age'].fillna(trainOrg['Age'].median())

trainOrg['Cabin'] = trainOrg['Cabin'].fillna(0)

trainOrg['Cabin'][trainOrg.Cabin!='0']=1

trainOrg['Embarked'] = trainOrg['Embarked'].fillna('S')



# Make Categorical values into numeric

trainOrg['famSize'] = trainOrg.SibSp + trainOrg.Parch

trainOrg['family'] = np.where(trainOrg['famSize']==0, 'N', 'Y')

trainOrg['Embarked'] = np.where(trainOrg['Embarked']=='C', 1, 0)

trainOrg['AgeGroup'] = '0'

trainOrg['AgeGroup'] = np.where(trainOrg['Age']<=16, 'Child', 'Adult')



trainOrg['Sex'] = np.where(trainOrg['Sex']=='female', 0, 1)

trainOrg['Cabin'] = np.where(trainOrg['Cabin']=='0', 0, 1)

trainOrg['family'] = np.where(trainOrg['family']=='Y', 1, 0)

trainOrg['AgeGroup'] = np.where(trainOrg['AgeGroup']=='Adult', 0, 1)
trainOrg.head()
#'SibSp','Parch','Embarked','Survived','Age','Fare' best so far

X = trainOrg.drop(['Survived','SibSp','Parch','Age','Fare','family','Embarked','Pclass'],axis=1)

Y = trainOrg.Survived
X.head()
# Validation Set approach

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=48)

accuracyList = {}



clf = tree.DecisionTreeClassifier(class_weight="balanced")

gaussian = GaussianNB()

logreg = LogisticRegression(class_weight="balanced")

boost = GradientBoostingClassifier()

knn = KNeighborsClassifier(n_neighbors = 3)

forest = RandomForestClassifier(n_estimators = 18)



models = [clf,gaussian,logreg,boost,knn,forest]



for model in models:

    accuracyList[model] = 0

    

for model in models:

    modelV = model.fit(X_train, Y_train)

    Y_pred = modelV.predict(X_test)

    accuracyList[model] = round(modelV.score(X_test,Y_test)*100,3)



#models = [clf,gaussian,logreg,boost,knn,forest]

accResult = pd.DataFrame({

    'Model': ['Random Forest', 'KNN', 'Naive Bayes', 

              'Logistic Regression', 'Decision Tree', 'Gradient Boosting'],

    'Score': [accuracyList[forest], accuracyList[knn], accuracyList[gaussian], 

              accuracyList[logreg], accuracyList[clf], accuracyList[boost]]})

accResult.sort_values(by='Score', ascending=False)
gboost = GradientBoostingClassifier(n_estimators = 100)

model = gboost.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

round(model.score(X_test,Y_test)*100,3)
forest = RandomForestClassifier(n_estimators = 100)

model = forest.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

round(model.score(X_test,Y_test)*100,3)
testOrg.head()
# Fill Null values of train set

testOrg['Age'] = testOrg['Age'].fillna(testOrg['Age'].median())

testOrg['Cabin'] = testOrg['Cabin'].fillna(0)

testOrg['Cabin'][testOrg.Cabin!=0]=1

testOrg['Embarked'] = testOrg['Embarked'].fillna('S')

testOrg['Fare'] = testOrg['Fare'].fillna(testOrg['Fare'].median())



# Make Categorical values into numeric

testOrg['famSize'] = testOrg.SibSp + testOrg.Parch

testOrg['family'] = np.where(testOrg['famSize']==0, 'N', 'Y')

testOrg['AgeGroup'] = '0'

testOrg['AgeGroup'] = np.where(testOrg['Age']<=16, 'Child', 'Adult')



testOrg['Sex'] = np.where(testOrg['Sex']=='female', 0, 1)

testOrg['family'] = np.where(testOrg['family']=='Y', 1, 0)

testOrg['AgeGroup'] = np.where(testOrg['AgeGroup']=='Adult', 0, 1)
#X_test = testOrg.drop(['SibSp','Parch','Embarked'],axis=1)

#'Survived','SibSp','Parch','Age','Fare','family','Embarked'

X_test = testOrg.drop(['SibSp','Parch','Embarked','Age','Fare','family','Pclass'],axis=1)
X_test[pd.isnull(X_test).any(axis=1)]
Y_pred = model.predict(X_test)
Y_pred.shape
df1 = pd.DataFrame(df_test.PassengerId, columns=['PassengerId'])

df2 = pd.DataFrame(Y_pred.astype(object), columns=['Survived'])

result = pd.concat([df1, df2], join='outer', axis=1)
result.head(20)
result.to_csv('result2.csv', index=False)