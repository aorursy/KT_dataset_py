### Titantic Introduction 



## Competition Notes

# 1. 1502/2224 did not survive

# 2. Women, children and upper class more likely to survive



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display

import seaborn as sns

import matplotlib

%matplotlib inline

import matplotlib.pylab as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load training data into dataframe and preview 

trainDF = pd.read_csv('../input/train.csv')

print(trainDF.columns.values)

trainDF.head(5)
# Load test data into pandas and print first few rows and what the headers are

testDF = pd.read_csv('../input/test.csv')

print(testDF.columns.values)

testDF.head(5)
# Note there are alot of NaNs for Cabin, separate into numerical and categorical features

# - Categorical: Sex, Survived, Embarked, Pclass (ordinal)

# - Numerical: Age, Fare, Parch, SibSp

# - Mixed: Cabin, Ticket

# Name contains titles which could be helpful



# Show distributions of numerical and categorical objects in dataframe

display(trainDF.describe()); display( trainDF.describe(include=['O']) )
# From the above see that Age and Cabin contain missing/empty values

# The next step is to look at statistics/correlations that we are given in problem 

# In the above we see that 38% survived 

#

# First examine Sex and Survival

trainDF[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False)
# Can see 74% women survived 

# Next look at Pclass where 1 is elite class

trainDF[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
# Next let's look at distributions of Age for those that Survived and those that did not

g=sns.FacetGrid(trainDF,col='Survived')

g.map(plt.hist,'Age',bins=20)
#Large # who died were in early 20s but spike of kids less than 5 survived

# Now let's look at age distribution and survival among Pclass

grid=sns.FacetGrid(trainDF,col='Survived',row='Pclass')

grid.map(plt.hist,'Age',bins=20)
# Look at correlation with categorical features like Embarked

grid=sns.FacetGrid(trainDF,row='Embarked')

grid.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep');grid.add_legend()



trainDF[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# Those who embarked at C have higher survival, and men survived more than women

grid=sns.FacetGrid(trainDF,row='Embarked',col='Survived',size=2.2,aspect=1.5)

grid.map(sns.barplot,'Sex','Fare',alpha=.5, ci=None);grid.add_legend()

#consistent above for S and C is that higher fare had higher survival rate



# drop ticket and cabin from data frames

trainDF = trainDF.drop(['Ticket','Cabin'],axis=1)

testDF  = testDF.drop(['Ticket','Cabin'],axis=1)

combined=[trainDF,testDF]

trainDF.head(5)
# Extract title from Names and make it a new feature

for df in combined:

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(trainDF['Title'],trainDF['Sex'])
# Convert Sex to Categorical

sex_dict={'male':0,'female':1}

for df in combined:

    df['Sex'] = df['Sex'].map(sex_dict).astype(int)



testDF.head(5)
#Want to reduce the above to Mr, Mrs,Miss,Master and Rare

for df in combined:

    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dona','Dr','Major','Rev','Sir','Jonkheer'],'Rare')

    df['Title'] = df['Title'].replace('Mme','Mrs')

    df['Title'] = df['Title'].replace('Ms','Miss')

    df['Title'] = df['Title'].replace('Mlle','Miss')

display(trainDF.describe(include=['O']) )

trainDF[['Title','Survived']].groupby(['Title'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# Rare and Master had slightly better odds than Mr.

# Now conver title into integer categorical

title_dict={'Mr':1,'Miss':2, "Mrs": 3, "Master": 4, "Rare": 5}

for df in combined:

    df['Title'] = df['Title'].map(title_dict).astype(int)

    df['Title'] = df['Title'].fillna(0)

trainDF.head(10)
# Get rid of name and PassengerID from dataframe training but keep for test since we need it

trainDF = trainDF.drop(['Name','PassengerId'],axis=1)

testDF  = testDF.drop(['Name'],axis=1)

combined=[trainDF,testDF]

trainDF.head(5)
## Complete umerical values that are missing or NaN - Age

# use correlations among age, gender and Pclass along with mean and standard deviation of Age

grid=sns.FacetGrid(trainDF,row='Pclass',col='Sex',size=2.2,aspect=1.6)

grid.map(plt.hist,'Age',alpha=0.5,bins=20);grid.add_legend

# Array for guessing ages

guess_ages=np.zeros((2,3)) # sex and pclass

for df in combined:

    for i in range(0,2):

        for j in range(0,3):

            guess_df= df[(df['Sex']==i)&(df['Pclass']==j+1)]['Age'].dropna()

            

            age_guess= guess_df.median()

            guess_ages[i,j] = int(age_guess/0.5 +0.5)*0.5

     

            df.loc[(df.Age.isnull() ) & (df.Sex==i) & (df.Pclass==j+1),'Age' ]=guess_ages[i,j]

            

    df['Age'] = df['Age'].astype(int)



trainDF.head(5)
# Create agebands for age try 5 bins

trainDF['AgeBand'],agebins=pd.cut(trainDF['Age'],10,retbins=True)

trainDF[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending=True)
#Use agebands as categorical for Age, using numpy digitize

for df in combined:

    df['Age'] = np.digitize(df['Age'].values,agebins) -1

trainDF.head(5)
#Remove AgeBand

trainDF = trainDF.drop(['AgeBand'],axis=1)

trainDF.head(5)

combined=[trainDF,testDF]
# Create new feature on family size

for df in combined:

    df['FamilySize'] = df['SibSp'] + df['Parch']



trainDF[['FamilySize','Survived']].groupby(['FamilySize']).mean().sort_values(by='Survived',ascending=False)
# Create new feature if alone 

for df in combined:

    df['IsAlone'] = 0

    df.loc[(df.FamilySize==0),'IsAlone'] = 1



trainDF[['IsAlone','Survived']].groupby(['IsAlone']).mean()

# Higher chance of survival if not alone

# drop FamilySize,Parch, and SibSp

testDF = testDF.drop(['FamilySize','Parch','SibSp'],axis=1)

trainDF = trainDF.drop(['FamilySize','Parch','SibSp'],axis=1)

combined=[trainDF,testDF]

trainDF.head(5)
#Combine Age and Pclass into own feature

for df in combined:

    df['AgeClass'] = df.Age*df.Pclass

trainDF.loc[:, ['AgeClass', 'Age', 'Pclass']].head(10)
# Recall from above that Embarked is missing two values, fill it in with the most common occuring

port_mode= trainDF['Embarked'].dropna().mode()[0]

print(port_mode)
# Fill in in above

for df in combined:

    df.loc[ df.Embarked.isnull(),'Embarked'] = port_mode

    #df.Embarked=df.Embarked.fillna(port_mode)
## Change Embarked to Ordinal Categorical

embark_dict={'S':0,'C':1,'Q':2}

for df in combined:

    df.Embarked = df.Embarked.map(embark_dict)

trainDF.head(5)
# Fill in Fare with most common 

fare_med= trainDF.Fare.dropna().median()

trainDF.loc[trainDF.Fare.isnull(),'Fare'] = fare_med
# Instead of cutting by bins over range like for age, cut by quantiles for fares

trainDF['FareQ'],farebins = pd.qcut(trainDF.Fare,10,retbins=True)

print(farebins)
trainDF[['FareQ','Survived']].groupby(['FareQ'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# Make fare into categorical feature

for df in combined:

    df.Fare = np.digitize(df.Fare.values,farebins)-1

    df.Fare = df.Fare.astype(int)
# Drop FareQ from training

trainDF = trainDF.drop(['FareQ'],axis=1)

combined = [trainDF,testDF]
# Look at train and test, make sure setup right

display( trainDF.head(5))

testDF.head(5)
## Model Predicting 

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



Xtrain = trainDF.drop(['Survived'],axis=1)

Ytrain = trainDF.Survived

Xtest  = testDF.drop(['PassengerId'],axis=1).copy()

Xtrain.shape,Ytrain.shape,Xtest.shape
## Logistic regression 

logireg = LogisticRegression()

logireg.fit(Xtrain,Ytrain)

Ypred_lr = logireg.predict(Xtest)

acc_lr   = round(logireg.score(Xtrain,Ytrain)*100,2)
## Support Vector Machines

svc= SVC()

svc.fit(Xtrain,Ytrain)

Ypred_svc = svc.predict(Xtest)

acc_svc=round(svc.score(Xtrain,Ytrain)*100,2)
## K-Nearest Neighbor Classification

knn= KNeighborsClassifier(n_neighbors=3)

knn.fit(Xtrain,Ytrain)

Ypred_knn=knn.predict(Xtest)

acc_knn=round(knn.score(Xtrain,Ytrain)*100,2)
## Gaussian Naive Bayes

gauss = GaussianNB()

gauss.fit(Xtrain,Ytrain)

Ypred_gauss=gauss.predict(Xtest)

acc_gauss=round(gauss.score(Xtrain,Ytrain)*100,2)
## Perceptron

percept = Perceptron()

percept.fit(Xtrain,Ytrain)

Ypred_percept = percept.predict(Xtest)

acc_percept = round(percept.score(Xtrain,Ytrain)*100,2)
## Decision Tree

dtree = DecisionTreeClassifier()

dtree.fit(Xtrain,Ytrain)

Ypred_dtree=dtree.predict(Xtest)

acc_dtree = round(dtree.score(Xtrain,Ytrain)*100,2)
## Random Forest 

rforest= RandomForestClassifier(n_estimators=100)

rforest.fit(Xtrain,Ytrain)

Ypred_rforest=rforest.predict(Xtest)

acc_rforest=round(rforest.score(Xtrain,Ytrain)*100,2)
## Rank Models

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron',  

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_lr, 

              acc_rforest, acc_gauss, acc_percept, 

             acc_dtree],

    'Ypred': [Ypred_svc, Ypred_knn, Ypred_lr,

              Ypred_rforest,Ypred_gauss,Ypred_percept,

               Ypred_dtree]

        })

models[['Model','Score']].sort_values(by='Score',ascending=False)



Ybest = models.loc[models.Score.idxmax(),'Ypred']
## Submission 

submission = pd.DataFrame({

                'PassengerId':testDF['PassengerId'],

                'Survived':Ybest

                })
submission.to_csv('submission.csv',index=False)