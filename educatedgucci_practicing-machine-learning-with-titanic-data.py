# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra|

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import statistics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Let's fill out missing data first



df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

df.isnull().sum()
test.isnull().sum()
#fill missing vlaue on test[Cabin] columns 



test[test['Fare'].isnull()]
# His Pclass is 3, so let's put average fare of 3rd class.

savg = test['Fare'][test['Pclass']==3].mean()

test['Fare'].fillna(savg, inplace =True)
#just filled Embarked  'S' since most Embarked is 'S'



df['Embarked'].value_counts(sort=True, ascending=False)

df['Embarked'].fillna('S',inplace =True)
df[df["Age"].isna()]



# At least I don't see any characteristic to distinguish age...

# Then let's just fill average age
age = df['Age']

avgage = age.sum(axis = 0, skipna = True) /len(age)

df['Age'].fillna(avgage, inplace = True)



age1 = test['Age']

testavgage = age1.sum(axis = 0, skipna = True) /len(age1)

test['Age'].fillna(testavgage, inplace = True)



# It seems Cabin doesn't affect the result of any analysis since it's just seat number.

# so this time i'll just remove Cabin column

df['Cabin']
df = df.drop('Cabin', axis = 1)

test = test.drop('Cabin', axis=1)
# Finally filled all missing data

df.isnull().sum()
test.isnull().sum()
df.sample(20)
def bar_chart(feature):

    survived = df[df['Survived']==1][feature].value_counts()

    dead = df[df['Survived']==0][feature].value_counts()

    df1 = pd.DataFrame([survived,dead])

    df1.index= ['Survived','Dead']

    df1.plot(kind='bar',stacked =True, figsize=(10,5))

# female Survived more than male



bar_chart('Sex')
# First class guests Survived more than 2nd or 3rd.



bar_chart('Pclass')
# Data Preprocessing

combined = [df,test]



for i in combined:

    i['Title'] = i['Name'].str.extract(' ([A-Za-z]+)\.', expand =True)



df['Title'].value_counts()

test['Title'].value_counts()
# Change each title to number.



title_map = {'Mr' : 0, 'Mrs':1, 'Miss':2, 'Master':3, 'Don':3, 'Rev':3, 'Dr':3, 'Mme':3, 'Ms':3,

       'Major':3, 'Lady':3, 'Sir':3, 'Mlle':3, 'Col':3, 'Capt':3, 'Countess':3,

       'Jonkheer':3,'Dona' :3}

for i in combined:

    i['Title'] = i['Title'].map(title_map)
df.head()
# Now we don't need whole name anymore. Remove name column



df.drop('Name',axis = 1, inplace =True)

test.drop('Name',axis = 1, inplace =True)

df.head()
# Change sex columns. Male equals 1, female equals 2 

sex_map = {'male' : 1, 'female' :2}

for i in combined:

    i['Sex'] = i['Sex'].map(sex_map)



df.head()
# Change age range. Let's count 0~13 as child, 14 ~ 20 as young, 

# 21 ~ 30 as adult, 31 ~ 55 as middle-age, 51~ as senior



for i in combined:

    i.loc[i["Age"] <= 13, 'Age'] = 0,

    i.loc[(i["Age"] > 13) & (i["Age"]  <=20),'Age'] =1,

    i.loc[(i["Age"] > 20) & (i["Age"]  <=30),'Age'] =2,

    i.loc[(i["Age"] > 30) & (i["Age"]  <=55),'Age'] =3,

    i.loc[i["Age"] >  55, 'Age'] = 4

    

            
bar_chart('Age')
# Embarked -> number

embarked_map = {'S':0, 'C': 1, 'Q' :2}

for i in combined:

    i['Embarked'] = i['Embarked'].map(embarked_map)
# And we might be able to combine parch and sibsp columns, which makes you see data eaiser.

df['Familysize'] = df['SibSp'] + df['Parch']

test['Familysize'] = test['SibSp'] + test['Parch']
df.head()
# And we don't need Sibsp, Parch columns anymore, so just remove them.

# Plus Ticket, Passengerid columns too...

for i in [df,test]:

    del i['Ticket'];  del i['Parch']; del i['SibSp']

train =df.drop(['PassengerId'],axis=1)

testset= test.drop(['PassengerId'],axis=1)
# modeling

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier



x = train.drop(['Survived'], axis=1)

y = train['Survived']

xtrain,xtest, ytrain,ytest = train_test_split(x,

    y,test_size = 0.25, random_state = 123)



xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

# 1. K-fold Classifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=123)



clf = KNeighborsClassifier(n_neighbors = 10) 

scoring = 'accuracy'

score = cross_val_score(clf, xtrain, ytrain, cv=k_fold, n_jobs=1, scoring=scoring)

print('K-fold score : ' ,score)



print('Average Accuracy  :' , round(np.mean(score)*100, 2))
# 2. Decision Tree 

clf = DecisionTreeClassifier()

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

            max_features=None, max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, presort=False, random_state=None,

            splitter='best')

score = cross_val_score(clf, xtrain, ytrain, cv=10, n_jobs=1, scoring=scoring)

print(score)



print('Average Accuracy :' , round(np.mean(score)*100, 2))
# 3. Random Forest

# I used 20 decision tree

clf = RandomForestClassifier(n_estimators=20, random_state = 101) 

score = cross_val_score(clf, xtrain, ytrain, cv=10, n_jobs=1, scoring=scoring)

print(score)

print('Average Accuracy :' , round(np.mean(score)*100, 2))
# 4. Naive Bayes



clf = GaussianNB()

score = cross_val_score(clf, xtrain, ytrain, cv=10, n_jobs=1, scoring=scoring)

print(score)

print('Average Accuracy :' , round(np.mean(score)*100, 2))

# 5. SVM

clf = SVC()

score = cross_val_score(clf, xtrain, ytrain, cv=10, n_jobs=1, scoring=scoring)

print(score)

print('Average Accuracy :' , round(np.mean(score)*100, 2))
# 6. Logistic Regression



clf = LogisticRegression()

score = cross_val_score(clf, xtrain, ytrain, cv=10, n_jobs=1, scoring=scoring)

print(score)

print('Average Accuracy :' , round(np.mean(score)*100, 2))
# 7. XGBOOST

#from sklearn.model_selection import KFold,GridSearchCV

#from xgboost import XGBClassifier

#clf = XGBClassifier()

#param_grid={     'silent':[True],

                 #'max_depth':[5,6,8],

                 #'min_child_weight':[1,3,5],

                 #'gamma':[0,1,2,3],

                 #'nthread':[4],

                 #'colsample_bytree':[0.5,0.8],

                 #'colsample_bylevel':[0.9],

                 #'n_estimators':[50],

                 #'random_state':[2]}



#gcv=GridSearchCV(clf, param_grid=param_grid, cv=6, scoring='f1', n_jobs=4)



#gcv.fit(xtrain,ytrain)

#print('final params', gcv.best_params_) 

#print('best score', gcv.best_score_)  



# The best parameters are (clf = XGBClassifier(colsample_bylevel= 0.9,

                    #colsample_bytree = 0.8, 

                    #gamma=0,

                    #max_depth= 5,

                    #min_child_weight= 1,

                    #n_estimators= 50,

                    #nthread= 4,

                    #random_state= 2,

                    #silent= True))

                    

# grid_search on Kaggle is so slow that I didn't save this code.



# XGBoost shows the highest accuracy, so I'll use XGBoost model for test.

import sklearn.metrics as metrics

from xgboost import XGBClassifier

clf = XGBClassifier()

clf = XGBClassifier(colsample_bylevel= 0.9,

                    colsample_bytree = 0.8, 

                    gamma=0.2,

                    max_depth= 5,

                    min_child_weight= 1,

                    n_estimators= 3000,

                    nthread= 4,

                    random_state= 2,

                    silent= True)

clf.fit(xtrain,ytrain)

test_prediction = clf.predict(xtest)  # Prediction with xtest

print('Test Accuracy : ', metrics.accuracy_score(test_prediction,ytest))
# Train all dataset in XGBoost and make a real prediciton on test

clf = XGBClassifier(colsample_bylevel= 0.9,

                    colsample_bytree = 0.8, 

                    gamma=0,

                    max_depth= 5,

                    min_child_weight= 1,

                    n_estimators= 50,

                    nthread= 4,

                    random_state= 2,

                    silent= True)

clf.fit(x,y)

score = cross_val_score(clf, x,y, cv=10, n_jobs=1, scoring=scoring)

print(score)

print('Average Accuracy :' , round(np.mean(score)*100, 2))
# Prediciton by XGBoost



result = clf.predict(testset)



result[0:10,]
import collections, numpy



collections.Counter(result)



submission= pd.DataFrame({

    "PassengerId" : test["PassengerId"],

    'Survived' : result

})



submission.to_csv('submission.csv',index=False)
