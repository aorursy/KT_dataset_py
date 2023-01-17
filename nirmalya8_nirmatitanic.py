# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.shape
train.head()
train.info()
train.describe()
train.isnull().sum()
train["Embarked"].value_counts()
train['Embarked'].fillna('S',inplace = True)

train['Embarked'].value_counts()
train['Age'].fillna(train['Age'].median(),inplace = True)
train.isnull().sum()
train['Cabin'].fillna("U",inplace = True)
import matplotlib.pyplot as plt

import seaborn as sns
train['Age'].hist(bins=15)
plt.boxplot(train['Age'])
train['Fare'].hist()
plt.boxplot(train['Fare'])
print(train['Pclass'].value_counts())

train['Pclass'].hist()
train['SibSp'].value_counts()
train['SibSp'].hist()
plt.boxplot(train['SibSp'])
train['Parch'].value_counts()
train['Parch'].hist()
plt.boxplot(train['Parch'])
train['Age']
train['NewAge'] = np.sin(train['Age'])

plt.boxplot(train['NewAge'])
train['NewAge'].describe()
train['NewFare'] = np.cos(train['Fare'])

plt.boxplot(train['NewFare'])
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train['Cab'] = train['Cabin'].astype(str).str[0]

train.head()
train['Cab'].value_counts()
train.head()
train['fam'] = train['SibSp']+train['Parch']

train.head()
train[['fam', 'Survived']].groupby(['fam'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train['fam'].value_counts()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
import re

def findTitle(name):

    match = re.search("(Dr|Mrs?|Ms|Miss|Master|Rev|Capt|Mlle|Col|Major|Sir|Jonkheer|Lady|the Countess|Mme|Don)\\.",name)

    if match:

        title = match.group(0)

        if (title == 'Don.' or title == 'Major.' or title == 'Capt.'):

            title = 'Sir.'

        if (title == 'Mlle.' or title == 'Mme.'):

            title = 'Miss.'

        return title

    else:

        return "Other"

train["Title"] = train["Name"].apply(findTitle)



train.head()
train.info()
pd.crosstab(train['Title'], train['Sex'])
train["T"] = train["Ticket"].apply(lambda x: str(x)[0])

train.head()
train['T'].value_counts()
train['T'].replace({'S':10,'P':11,'C':12,'A':13,'W':14,'F':15,'L':16},inplace = True)

train['T'].value_counts()
train.head()
train['Cab'].value_counts()
enc_cab = {'U':0,'C':1,'B':2,'D':3,'E':4,'A':5,'F':6,'G':7,'T':8}

train['Cab'] = train['Cab'].replace(enc_cab)

train['Cab'].value_counts()
Uncommon = {'Lady.':0, 'the Countess.':0,'Capt.':0, 'Col.':0,'Don.':0, 'Dr.':0, 'Major.':0, 'Rev.':0, 'Sir.':0, 'Jonkheer.':0, 'Dona.':0}



train['Title'] = train['Title'].replace(Uncommon)

train['Title'].value_counts()
t = {5 : 'u'}

train['Title'] = train['Title'].replace(t)

train['Title'].value_counts()
title_encode = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4,'u':0,'Ms.':0}

train['Title'] = train['Title'].replace(title_encode)

train['Title'].value_counts()
train = train.drop(['Name', 'PassengerId'], axis=1)

train.head()
sex_encode = {"male":1,"female":2}

train['Sex'] = train['Sex'].replace(sex_encode)

train.head()
train['Embarked'].value_counts()

embarked_encode = {'S':1,'C':2,'Q':3}

train['Embarked'] = train['Embarked'].replace(embarked_encode)

train.head()
grid = sns.FacetGrid(train, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
plt.figure(figsize= (16,10))

sns.heatmap(train.corr(),cmap = 'Dark2',annot = True,linewidths=1.0,linecolor='black')
plt.figure(figsize= (16,10))

sns.heatmap(np.abs(train.corr()),cmap = 'Dark2',annot = True,linewidths=1.0,linecolor='black')
test.shape
test.head()
test.info()
test.isnull().sum()
test['Age'].fillna(test['Age'].median(),inplace = True)
test.isnull().sum()
test['Fare'].fillna(test['Fare'].median(),inplace = True)
test.isnull().sum()
test['Cabin'].fillna('U',inplace = True)

test.head()
test['NewAge'] = np.sin(test['Age'])

plt.boxplot(train['NewAge'])
test['NewFare'] = np.cos(test['Fare'])

plt.boxplot(test['NewFare'])
test['Cab'] = test['Cabin'].astype(str).str[0]

test.head()
test['Cab'].value_counts()
test['fam'] = test['SibSp']+test['Parch']

test.head()
test["T"] = test["Ticket"].apply(lambda x: str(x)[0])
print(test['T'].value_counts())

test.head()
test['T'].replace({'S':10,'P':11,'C':12,'A':13,'W':14,'F':15,'L':16},inplace = True)

test['T'].value_counts()
test["Title"] = test["Name"].apply(findTitle)



test.head()
enc_cab = {'U':0,'C':1,'B':2,'D':3,'E':4,'A':5,'F':6,'G':7,'T':8}

test['Cab'] = test['Cab'].replace(enc_cab)

test['Cab'].value_counts()
Uncommon = {'Lady.':'u', 'the Countess.':'u','Capt.':'u', 'Col.':'u','Don.':'u', 'Dr.':'u', 'Major.':'u', 'Rev.':'u', 'Sir.':'u', 'Jonkheer.':'u', 'Dona.':'u', 'Other':'u','Ms.':'u'}



test['Title'] = test['Title'].replace(Uncommon)

test['Title'].value_counts()

title_encode = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4,'u':0,'Ms.':0}

test['Title'] = test['Title'].replace(title_encode)

test['Title'].value_counts()

sex_encode = {"male":1,"female":2}

test['Sex'] = test['Sex'].replace(sex_encode)

test.head()
embarked_encode = {'S':1,'C':2,'Q':3}

test['Embarked'] = test['Embarked'].replace(embarked_encode)

test.head()
train.head()
from sklearn.linear_model import LogisticRegression   

from sklearn.model_selection import KFold 

from sklearn.ensemble import RandomForestClassifier 

from sklearn import metrics
def classification_model(model, data, predictors, outcome):  

    #Fit the model:  

    model.fit(data[predictors],data[outcome])    

    #Make predictions on training set:  

    predictions = model.predict(data[predictors])    

    #Print accuracy  

    accuracy = metrics.accuracy_score(predictions,data[outcome])  

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    #Perform k-fold cross-validation with 5 folds  

    kf = KFold(5,shuffle=True)  

    error = []  

    for train, test in kf.split(data):

        # Filter training data    

        train_predictors = (data[predictors].iloc[train,:])        

        # The target we're using to train the algorithm.    

        train_target = data[outcome].iloc[train]        

        # Training the algorithm using the predictors and target.    

        model.fit(train_predictors, train_target)

        #Record error from each cross-validation run    

        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

     

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))) 

    # %s is placeholder for data from format, next % is used to conert it into percentage

    #.3% is no. of decimals

    return model
output = 'Survived'

model = RandomForestClassifier()

predict = ['Sex','Title','Pclass','T']

classification_model(model,train,predict,output)

m = classification_model(model,train,predict,output)

a = m.predict(test[predict])

a
output = 'Survived'

model = RandomForestClassifier()

predict = ['Sex','Title','Pclass','Cab','Embarked']

mod = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

classification_model(mod,train,predict,output)

mod.best_params_
output = 'Survived'

model = RandomForestClassifier(n_estimators = 800, min_samples_split= 2, min_samples_leaf= 1, max_features= 'auto', max_depth= 100, bootstrap= True)

predict = ['Sex','Title','Pclass','Cab','Age']

classification_model(model,train,predict,output)

m = classification_model(model,train,predict,output)

a = m.predict(test[predict])

a
my_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': a})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
output = 'Survived'

model = RandomForestClassifier()

predict = ['Sex','Title','Pclass','Cab','Embarked','T']

classification_model(model,train,predict,output)

m = classification_model(model,train,predict,output)

a = m.predict(test[predict])

a
from sklearn import svm
output = 'Survived'

model = svm.SVC()

predict = ['Sex','Title','Pclass','Cab','Embarked']

classification_model(model,train,predict,output)

m = classification_model(model,train,predict,output)

a = m.predict(test[predict])

a
from sklearn.neighbors import KNeighborsClassifier



output = 'Survived'

model = KNeighborsClassifier(n_neighbors = 3)

predict = ['Sex','Title','Pclass']

classification_model(model,train,predict,output)

m = classification_model(model,train,predict,output)

a = m.predict(test[predict])

a