# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

import warnings

%matplotlib inline



warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read csv file



dataset = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
# Checking the Null Values in Dataset and count them



dataset.isnull().sum()
# In order the learn and predict the survived passengers some features plays more important role then others. These features

# are Pclass(Class of the Passenger), Age(age of passenger), Sex(Gender of passenger), Embarked. So we have to preprocess these 

# features in order the make them suitable of calculations but before that we need to visualise them that whether these features

# are important or not



# Feature Visualisation 



# Pclass vs Survived



survived = dataset[dataset['Survived']==1]['Pclass'].value_counts()

died = dataset[dataset['Survived']==0]['Pclass'].value_counts()



print('Survived', survived, 'Died', died, sep='\n')
df_class = pd.DataFrame([survived, died])

df_class.index = ['Survived', 'Died']

df_class.plot(kind='bar', figsize=(5, 3), stacked=True, title='Survived/Died By Class')



class1 = df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100

class2 = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100

class3 = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100



print("Percentage of Class 1 survived:", round(class1), "%")

print("Percentage of Class 2 survived:", round(class2), "%")

print("Percentage of Class 3 survived:", round(class3), "%")



display(df_class)
# Sex vs Survived



survived_sex = dataset[dataset['Survived']==1]['Sex'].value_counts()

died_sex = dataset[dataset['Survived']==0]['Sex'].value_counts()



df_sex = pd.DataFrame([survived_sex, died_sex])

df_sex.index = ['Survived', 'Died']

df_sex.plot(kind='bar', stacked=True, figsize=(5,3), title='Survived/Died by Sex')



display(df_sex)



female_survived = df_sex.iloc[0,0]/df_sex.iloc[:,0].sum()*100

male_survived = df_sex.iloc[0,1]/df_sex.iloc[:,1].sum()*100



print("Percentage of Female Survived:", round(female_survived), "%")

print("Percentage of Male Survived:", round(male_survived), "%")
# Embarked vs Survived



survived_embark = dataset[dataset['Survived']==1]['Embarked'].value_counts()

Died_embark = dataset[dataset['Survived']==0]['Embarked'].value_counts()



df_embark = pd.DataFrame([survived_embark, Died_embark])

df_embark.index = ['Survived', "Died"]

df_embark.plot(kind='bar', stacked=True, figsize=(5,3), title='Survived/Died by Embarked')

display(df_embark)



S_survived = df_embark.iloc[0,0]/df_embark.iloc[:,0].sum()*100

C_survived = df_embark.iloc[0,1]/df_embark.iloc[:,1].sum()*100

Q_survived = df_embark.iloc[0,2]/df_embark.iloc[:,2].sum()*100



print("Percentage of S Embark survived:", round(S_survived), "%")

print("Percentage of C Embark survived:", round(C_survived), "%")

print("Percentage of Q Embark survived:", round(Q_survived), "%")
# Data Cleaning and Feature selection



X = dataset.drop(['PassengerId', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1)

X_test = test_data.drop(['PassengerId', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1)

y = dataset['Survived']



X = X.drop(['Survived'], axis=1)
# Feature Engineering



#SEX

label_encode = LabelEncoder()

X.Sex = label_encode.fit_transform(X.Sex)

X_test.Sex = label_encode.fit_transform(X_test.Sex)
#EMBARKED



#First fill the null values in Embarked with the most common type i.e S



row_index = X.Embarked.isnull()

X.loc[row_index, 'Embarked'] = 'S'



Embarked = pd.get_dummies(X.Embarked, prefix='Embarked')

X = X.drop(['Embarked'], axis=1)

X = pd.concat([X, Embarked], axis=1)



X = X.drop(['Embarked_S'], axis=1)
X.head(30)
Embarked1 = pd.get_dummies(X_test.Embarked, prefix='Embarked')



X_test = X_test.drop(['Embarked'], axis=1)

X_test = pd.concat([X_test, Embarked1], axis=1)

X_test = X_test.drop(['Embarked_S'], axis=1)

X_test
# Now we try to fill the null Age fields with the help Name titles



got = dataset.Name.str.split(',').str[1]

X.iloc[:, 1] = pd.DataFrame(got).Name.str.split('\s+').str[1]

X.head()
ax = plt.subplot()

ax.set_ylabel('Average Age')

X.groupby('Name').mean()['Age'].plot(kind='bar', figsize=(13, 8), ax=ax)
got1 = test_data.Name.str.split(',').str[1]

X_test.Name = pd.DataFrame(got1).Name.str.split('\s+').str[1]

X_test.head()
title_mean_age = []

title_mean_age.append(list(set(X.Name)))

title_mean_age.append(X.groupby('Name').Age.mean())

title_mean_age
# Replace the null age fields with its title mean age

training = dataset.shape[0]

training1 = test_data.shape[0]

titles = len(title_mean_age[1])



for i in range(0, training):

    if np.isnan(X.Age[i]):

        for j in range(0, titles):

            if X.Name[i] == title_mean_age[0][j]:

                X.Age[i] = title_mean_age[1][j]



for i in range(0, training1):

    if np.isnan(X_test.Age[i]):

        for j in range(0, titles):

            if X_test.Name[i] == title_mean_age[0][j]:

                X_test.Age[i] = title_mean_age[1][j]

# Drop the name field



X = X.drop(['Name'], axis=1)

X_test = X_test.drop(['Name'], axis=1)
# we can further alter the age feature such that those whose age is above 18 will be considered as adults otherwise minors



for i in range(0, training):

    if X.Age[i] > 18:

        X.Age[i] = 0

    else:

        X.Age[i] = 1



for i in range(0, training1):

    if X_test.Age[i] > 18:

        X_test.Age[i] = 0

    else:

        X_test.Age[i] = 1
X_test.head(30)
# Now our data is preprocessed, we can now pass it to different Classifiers and see which performs better in creating a model

# of classification for this data.



# Logistic Regression



lc = LogisticRegression(penalty='l2', random_state=0)

acc1 = cross_val_score(estimator=lc, X=X, y=y, cv=10)

print("Logistic Regression:\n Accuracy:", acc1.mean(), '+/-', acc1.std(), '\n')
lc.fit(X, y)
sol1 = pd.DataFrame(lc.predict(X_test)) # predict the result of test data using Logistic Regression Classifier

sol1
# Support Vector Classifier (SVM)



clf = SVC()

acc2 = cross_val_score(estimator=clf, X=X, y=y, cv=10)

clf.fit(X, y)

print("SVC:\n Accuracy:", acc2.mean(), '+/-', acc2.std(), '\n')
sol2 = pd.DataFrame(clf.predict(X_test)) #Predict X_test using SVM Classifier

sol2
# Random Forest Classifier



rfc = RandomForestClassifier(n_estimators=200)

acc3 = cross_val_score(estimator=rfc, X=X, y=y, cv=10)

rfc.fit(X,y)

print("Random Forest:\n Accuracy:", acc3.mean(), '+/-', acc3.std(), '\n')
sol3 = pd.DataFrame(rfc.predict(X_test))

sol3