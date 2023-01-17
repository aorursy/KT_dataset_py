# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score
train = pd.read_csv('../input/train.csv')



test = pd.read_csv('../input/test.csv')
train.info()
train.head(10)
train.head()
#because our target variable "survived" consists of 1's and 0's we can see a few things fairly quickly!

print("Total Survivors: ",train['Survived'].sum())



percentSurvived = round(train['Survived'].mean(),2)

print("Percent Survive: ", percentSurvived)
#Did women have a better chance of survival?

cols = ['Sex','Survived']

genderSurvival = train[cols].groupby('Sex').mean().round(2)

genderSurvival
#we can show this same thing with a bar plot

ax = sns.barplot(x="Sex", y="Survived", data=train, ci=None)
train.tail(10)
#first, let's get an idea of the age breakdown of the passengers

ax = sns.kdeplot(train['Age'])
#What about children? They were supposed to do better, too, right?



#make a new column to classify a child from adult

train['ageClass'] = np.where(train['Age'] < 18, 'child','adult')



#lets make sure this worked. We can view our data using the "head" method

train.head(10)
#Okay, so how do kids do?

cols = ['ageClass','Survived']

ageSurvival = train[cols].groupby(['ageClass']).mean().round(2)

ageSurvival
#we can show this same thing with a bar plot

ax = sns.barplot(x="ageClass", y="Survived", data=train, ci=None)
train.head(10)
#update our ageClass column to include "young adults"

YA = (train['Age'] > 13) & (train['Age'] < 18)

train['ageClass'] = np.where(YA, 'young adult',train['ageClass'])



train.head(10)
#With young adults added, we can run the exact same code!

cols = ['ageClass','Survived']

ageSurvival = train[cols].groupby(['ageClass']).mean().round(2)



ax = sns.barplot(x="ageClass", y="Survived", data=train, ci=None)

ageSurvival
ax = sns.lineplot(x="Age", y="Survived", data=train)
#We can layer multiple fields into our table

cols = ['Sex','ageClass','Survived']

ageSexSurvival = train[cols].groupby(['Sex','ageClass']).mean().round(2)



ax = sns.barplot(x="Sex", y="Survived", hue='ageClass', data=train, ci=None)

ageSexSurvival
train['ageClass'] = np.where(train['Age'].isna(), 'missing',train['ageClass'])



#With young adults added, we can run the exact same code!

cols = ['ageClass','Survived']

ageSurvival = train[cols].groupby(['ageClass']).mean().round(2)



ax = sns.barplot(x="ageClass", y="Survived", data=train, ci=None)

ageSurvival
filt = train['ageClass'] == 'missing'

train[filt].head(10)
#Well, there are some features we could use to help deduce their age

#We could just take the overall average

avgAge = round(train['Age'].mean(),0)

print('Average Age: ', round(avgAge,0))
#Master is a title given to young boys back then, think "Master Wayne"

condition = (train['Name'].str.contains(pat = "Master")) & (train['Age'].isna())

train['ageClass'] = np.where(condition, 'child',train['ageClass'])



#Miss implies an unmarried girl. Probably a young adult or child, we'll just put YA for now

condition = (train['Name'].str.contains(pat = "Miss")) & (train['Age'].isna())

train['ageClass'] = np.where(condition, 'young adult',train['ageClass'])



#If they have more than one Sibling/Spouse, they are definitely travelling with a sibling. We are going to guess that those with siblings are travelling as a family and are kids

condition = (train['SibSp'] > 1) & (train['Age'].isna())

train['ageClass'] = np.where(condition, 'child',train['ageClass'])



#Assigning everyone else the average age of 30 making them adults

condition = (train['ageClass'] == 'missing')

train['ageClass'] = np.where(condition, 'adult',train['ageClass'])
#We can layer multiple fields into our table

cols = ['Sex','ageClass','Survived']

ageSexSurvival = train[cols].groupby(['Sex','ageClass']).agg({'Survived':'mean','ageClass':'count'}).round(2)



ax = sns.barplot(x="Sex", y="Survived", hue='ageClass', data=train, ci=None)

ageSexSurvival
#Looks like the more prestigious tickets did better than than the others

cols = ['Pclass','Survived']

classSurvival = train[cols].groupby(['Pclass']).agg({'Survived':'mean'}).round(2)



ax = sns.barplot(x="Pclass", y="Survived", data=train, ci=None)

classSurvival
#And first class women did particularly well

cols = ['Pclass','Sex','Survived']

classSurvival = train[cols].groupby(['Sex','Pclass']).agg({'Survived':'mean'}).round(2)



ax = sns.barplot(x="Pclass", y="Survived", hue='Sex', data=train, ci=None)

classSurvival
#And first class women did particularly well

cols = ['Pclass','Sex','ageClass','Survived']

classSurvival = train[cols].groupby(['Sex','ageClass','Pclass']).agg({'Survived':'mean','ageClass':'count'}).round(2)



ax = sns.catplot(x="ageClass", y="Survived", hue='Sex', col='Pclass', data=train, kind='bar', ci=None)

classSurvival
train.head(10)
#we could just assign binary or integer values to our fields.

train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train.head()
#We could also do something called "One-hot encoding"

cols = ['PassengerId','ageClass']

train2 = train[cols]

train2 = pd.get_dummies(train2)

train2.head()
#we have 2 missing values in Embarked, I'm going to use the Fare to impute them

emCost = train.groupby(['Embarked']).agg({'Fare':'mean'})

emCost
#Hmm, I wonder where these passengers embarked from?

train[train['Embarked'].isna()]
train['Embarked'] = np.where(train['Embarked'].isna(),'C',train['Embarked'])



#I'm going to roughly order these in terms of average fare cost

train['Embarked'] = train['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)

train.head()
#I'm going to roughly order age classification as well

train['ageClass'] = train['ageClass'].map( {'child': 0, 'young adult': 1, 'adult': 2} ).astype(int)

train.head()
#Here are the columns I want to keep

cols = ['Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked','ageClass']

train = train[cols]

train.head()
#I'm also going to 'feature engineer' one more field that identifies people who are alone and family size

train['companions'] = train.loc[:,'SibSp'] + train.loc[:,'Parch']

train['loner'] = np.where(train['companions'] < 1, 1,0)

train.head()
train.info()
#I'm replicating all the transformations we did to our training dataset to our test dataset

test['ageClass'] = np.where(test['Age'] < 18, 'child','adult')

YA = (test['Age'] > 13) & (test['Age'] < 18)

test['ageClass'] = np.where(YA, 'young adult',test['ageClass'])

test['ageClass'] = np.where(test['Age'].isna(), 'missing',test['ageClass'])



condition = (test['Name'].str.contains(pat = "Master")) & (test['Age'].isna())

test['ageClass'] = np.where(condition, 'child',test['ageClass'])



condition = (test['Name'].str.contains(pat = "Miss")) & (test['Age'].isna())

test['ageClass'] = np.where(condition, 'young adult',test['ageClass'])



condition = (test['SibSp'] > 1) & (test['Age'].isna())

test['ageClass'] = np.where(condition, 'child',test['ageClass'])



condition = (test['ageClass'] == 'missing')

test['ageClass'] = np.where(condition, 'adult',test['ageClass'])



test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



test['Embarked'] = np.where(test['Embarked'].isna(),'C',test['Embarked'])

test['Embarked'] = test['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)



test['ageClass'] = test['ageClass'].map( {'child': 0, 'young adult': 1, 'adult': 2} ).astype(int)



pID = test["PassengerId"].copy(deep=True)



cols = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','ageClass']

test = test[cols]



test['companions'] = test.loc[:,'SibSp'] + test.loc[:,'Parch']

test['loner'] = np.where(test['companions'] < 1, 1,0)



test.head()
test.info()
#we have one null value in our fare, lets impute that with the mean

meanFare = test['Fare'].mean()

test['Fare'] = np.where(test['Fare'].isna(),meanFare,test['Fare'])

test.info()
#everything besides "Survived" is an independent variable, so we are going to set them asside and denote them with an X

X_train = train.drop("Survived", axis=1)



#Survived is our dependent or target variable, we will pull that aside as well and denot it with a Y

Y_train = train["Survived"]



#We already don't have a "Survived" column in our test data set, so we are just going to grab a copy

X_test  = test.copy(deep=True)



#The two training sets hsould retain the same length

#and the two X sets should retain the same width

columns = X_train.columns

X_train.shape, Y_train.shape, X_test.shape
from sklearn.metrics import accuracy_score



naive = X_train['Sex'].copy(deep=True)

acc_naive = round(accuracy_score(Y_train, naive)*100,2)

acc_naive
# Logistic Regression



#define the model

logreg = LogisticRegression()



#fit he model to training data

logreg.fit(X_train, Y_train)



#predict test data

logPred = logreg.predict(X_test)





acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
cvAcc_log = round(cross_val_score(logreg, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

print('Train Acc: ',acc_log)

print('CV Acc:    ',cvAcc_log)
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

svmPred = svc.predict(X_test)





acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc



cvAcc_svc = round(cross_val_score(svc, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

print('Train Acc: ',acc_svc)

print('CV Acc:    ',cvAcc_svc)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)





knnPred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn





cvAcc_knn = round(cross_val_score(knn, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

print('Train Acc: ',acc_knn)

print('CV Acc:    ',cvAcc_knn)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)





gauPred = gaussian.predict(X_test)



acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian



cvAcc_gaussian = round(cross_val_score(gaussian, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

print('Train Acc: ',acc_gaussian)

print('CV Acc:    ',cvAcc_gaussian)
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)



perPred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron



cvAcc_perceptron = round(cross_val_score(perceptron, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

print('Train Acc: ',acc_perceptron)

print('CV Acc:    ',cvAcc_perceptron)
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



svcPred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc



cvAcc_linear_svc = round(cross_val_score(linear_svc, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

print('Train Acc: ',acc_linear_svc)

print('CV Acc:    ',cvAcc_linear_svc)
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)



sgdPred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd



cvAcc_sgd = round(cross_val_score(sgd, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

print('Train Acc: ',acc_sgd)

print('CV Acc:    ',cvAcc_sgd)
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)



trePred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree



cvAcc_decision_tree = round(cross_val_score(decision_tree, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

print('Train Acc: ',acc_decision_tree)

print('CV Acc:    ',cvAcc_decision_tree)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100, max_depth=4)

random_forest.fit(X_train, Y_train)



ranPred = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)



cvAcc_random_forest = round(cross_val_score(random_forest, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)

print('Train Acc: ',acc_random_forest)

print('CV Acc:    ',cvAcc_random_forest)
#we can get feature importance out of random forests too! What are some thoughts we can gain from this?

importances = list(random_forest.feature_importances_)

feature_importances = pd.DataFrame([(feature, round(importance, 2)) for feature, importance in zip(columns, importances)])

feature_importances.columns = ['feature','importance']

feature_importances.sort_values('importance',ascending=False)
#scikit-learn has a voting object that acts just like any of our other models

voter = VotingClassifier(estimators=[('random_forest', random_forest), ('sgd', sgd), ('linear_svc', linear_svc),

                                    ('logreg',logreg),('svc',svc),('knn',knn),('gaussian',gaussian)], voting='hard')



scores = cross_val_score(voter, X_train, Y_train, cv=5, scoring='accuracy')



cvAcc_ensemble = round(cross_val_score(voter, X_train, Y_train, cv=5, scoring='accuracy').mean()*100,2)



voter.fit(X_train, Y_train)

ensPred = voter.predict(X_test)

acc_ensemble = round(voter.score(X_train, Y_train) * 100, 2)

print('Train Accuracy: ', acc_ensemble)

print("CV Accuracy:    %0.2f (+/- %0.2f)" % (round(scores.mean()*100,2), scores.std()))
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree','Ensemble','Naive'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree,acc_ensemble,acc_naive],

    'ScoreCV': [cvAcc_svc, cvAcc_knn, cvAcc_log, 

                  cvAcc_random_forest, cvAcc_gaussian, cvAcc_perceptron, 

                  cvAcc_sgd, cvAcc_linear_svc, cvAcc_decision_tree,cvAcc_ensemble,acc_naive]})

models.sort_values(by='ScoreCV', ascending=False)
submission = pd.DataFrame({

        "PassengerId": pID,

        "Survived": ranPred

    })

#submission.to_csv('../output/submission.csv', index=False)