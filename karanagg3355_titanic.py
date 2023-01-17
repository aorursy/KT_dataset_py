# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
traindf = pd.read_csv("/kaggle/input/titanic/train.csv")

testdf = pd.read_csv("/kaggle/input/titanic/test.csv")

traindf.head()
print(" Table rows:{} \n Table columns:{}".format(traindf.shape[0],traindf.shape[1]))

traindf.count()
missing_data = traindf.isna().sum()

missing_per = (traindf.isna().sum()/traindf.count())*100

missing_total = pd.concat([missing_data,missing_per],axis=1,keys=['Total','Percent'])

missing_total
traindf.describe()
traindf.sample(5)
PassengerId = traindf['PassengerId'].copy()

traindf = traindf.drop('PassengerId',1)

traindf = traindf.drop('Name',1)

traindf.head(10)

#traindf.tail()   to check last rows of table.
traindf.Sex[traindf.Sex=='male']=0

traindf.Sex[traindf.Sex=='female']=1

traindf.tail()
traindf['Sex'].unique()
#As interpreted cabin has a less amount of data. So we need to interepret if cabin name is actually important to us or not

s = traindf[pd.notnull(traindf['Cabin'])]

s[['Cabin','Survived']]

s['Cabin'].nunique()
fig = plt.figure(figsize=(4,6))

traindf.Survived.value_counts().plot(kind = "bar", alpha = 0.5)

#fig = plt.figure(figsize=(4,6))

#traindf.Survived.value_counts(normalize=True).plot(kind = "bar", alpha = 0.5)

#Normalize is used to check in percentage


fig = plt.figure(figsize=(4,6))

traindf.Sex.value_counts().plot(kind = "bar", alpha = 0.5)

print("Total Male:",traindf.Sex.value_counts()[0])

print("Total Female:",traindf.Sex.value_counts()[1])
plt.scatter(traindf.Survived, traindf.Age, alpha=0.1)

sns.heatmap(traindf.isnull(),yticklabels=False, cmap = 'winter')
sns.countplot(x='Survived',hue='Sex',data=traindf,palette='RdBu_r')
sns.countplot(x='Survived', hue='Pclass',data= traindf, palette='rainbow')
sns.distplot(traindf['Age'].dropna(),kde=False,bins=20,color='darkred')

traindf['Age'].hist(bins=30,color='darkred',alpha=0.7)
#Number of Siblings

sns.countplot(x='SibSp',data=traindf)
sns.countplot(x='Survived',hue='SibSp',data=traindf)
fig = plt.figure(figsize=(15,3))

sns.countplot(x='Fare',hue='Survived',data=traindf)
sns.boxplot(x='Pclass', y='Age',data=traindf,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age

traindf['Age'] = traindf[['Age','Pclass']].apply(impute_age,axis=1)

traindf.isna().sum()
traindf = traindf.drop('Cabin',1)

#remove 2 null values of Embarked

traindf.dropna(inplace=True)

traindf.isna().sum()
traindf.head()
sns.countplot(x='Embarked',data=traindf)
traindf.Embarked[traindf.Embarked=='S']=0

traindf.Embarked[traindf.Embarked=='C']=1

traindf.Embarked[traindf.Embarked=='Q']=2

traindf.tail()
traindf = traindf.drop('Ticket',1)

traindf.head()
#Modifying the test data.

passenger = testdf['PassengerId']

testdf = testdf.drop('PassengerId',1)

testdf = testdf.drop('Name',1)

testdf = testdf.drop('Ticket',1)

testdf = testdf.drop('Cabin',1)

testdf.Embarked[testdf.Embarked=='S']=0

testdf.Embarked[testdf.Embarked=='C']=1

testdf.Embarked[testdf.Embarked=='Q']=2

testdf.Sex[testdf.Sex=='male']=0

testdf.Sex[testdf.Sex=='female']=1

testdf.tail()
#Separating X values and Y values.

train_y = traindf["Survived"]

traindf = traindf.drop("Survived",axis=1)

traindf.head()
testdf['Age'] = testdf[['Age','Pclass']].apply(impute_age,axis=1)
impute = testdf[(testdf['Age']>=60) & (testdf['Age']<=61)].Fare.sum()/testdf[(testdf['Age']>=60) & (testdf['Age']<=61)].Fare.count()

testdf.Fare[testdf['Fare'].isna()] = impute

testdf.isna().sum()

testdf
#Now we will apply different types of algorithms for our analysis.

# 1) Logistic Regression

model = LogisticRegression()

model.fit(traindf, train_y)

predictionlr = model.predict(testdf)

model.score(traindf, train_y)

# 2) Support Vector machine 

model = SVC()

model.fit(traindf,train_y)

predictionsvc = model.predict(testdf)

model.score(traindf,train_y)

# 3) Random Forest

model = RandomForestClassifier(n_estimators = 150)

model.fit(traindf,train_y)

predictionrf = model.predict(testdf)

model.score(traindf,train_y)
# 4) KNN

model = KNeighborsClassifier(n_neighbors = 3)

model.fit(traindf,train_y)

predictionsknn = model.predict(testdf)

model.score(traindf,train_y)
# 5) Gaussian

model = GaussianNB()

model.fit(traindf,train_y)

predictiongb = model.predict(testdf)

model.score(traindf, train_y)
#As could be compared that random forest works best in such type of datasets.

#Note: All the ML algorithms are compared for understanding purpose.

submission = pd.DataFrame({

        "PassengerId": passenger,

        "Survived": predictionrf

    })

submission.to_csv('titanic.csv', index=False)