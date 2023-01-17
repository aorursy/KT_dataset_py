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
# importing necessary libraries, required for analysis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import math
# Importing data file - CSV



gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

train_df = pd.read_csv("../input/titanic/train.csv")



#combine = [train_df, test_df]
gender_submission.head()
test_df.head()
train_df.head()
#Since we have 2 different dataset Train and Test, hence both set needs to be checked seperately 

#Check info to know the missing values and data type in both dataset



train_df.info() ,test_df.info()



#Age & cabin in both dataset, has some missing values

#Embarked in train dataset also some missing values

#Fare in test dataset has some missing values
#checking shape - 'In train dataset 'Survived' is an additional'

train_df.shape, test_df.shape
train_df.groupby('Survived').count()#['Cabin']#.corr()
# Since the number of people surviving were less than otherwise. While survival rate of female was higher than male. Dataset is imbalanced.

plt.figure(figsize=(12,4))

sns.countplot(x='Survived',  data = train_df, hue='Sex')
#pre-analysis checking  correlation of Survived features with other features

train_df.corr()['Survived']
print("Total Null values in Cabin: ", train_df['Cabin'].isnull().sum(), ". Length of datframe: ", len(train_df), ". Non - Null Values : ", len(train_df) -  train_df['Cabin'].isnull().sum(), 

      ". Null %: ", train_df['Cabin'].isnull().sum()/len(train_df))
#Verifying Pclass and Cabin details

train_df.groupby('Pclass').count()['Cabin']
# Since the number of people surviving were less than otherwise. While survival rate of female was higher than male. Dataset is imbalanced.

plt.figure(figsize=(12,4))

sns.countplot(x='Survived',  data = train_df, hue='Pclass')
#pre-analysis checking  correlation of Pclass features with other features

train_df.corr()['Pclass']
#Understanding Cabin columns

set(train_df['Cabin'])



# Many Uniquie values and Cabin is more of a bnary (Yes / No) than the cabin number 
def check(val):

    if(pd.isnull(val)):

        return 0

    else:

        return 1
# Change Cabin to binary (0 or 1)

train_df['Cabin'] = train_df['Cabin'].apply(check)

test_df['Cabin'] = test_df['Cabin'].apply(check)
#Post EDA, Checking - Survived columns correlation with other columns 

train_df.corr()['Survived']



# WIth following details - EDA has resulted in identifying some positive correlated columns 
train_df.groupby('Survived').count()#['Cabin']#.corr()
# Passenger with cabin, had a higher chance of survival

plt.figure(figsize=(12,4))

sns.countplot(x='Survived',  data = train_df, hue='Cabin')
# Understanding survival based on price and Class 

train_df.groupby(['Pclass', 'Survived']).mean()['Fare']



plt.figure(figsize=(12,4))

sns.boxenplot(x='Pclass', y='Fare', data=train_df, hue='Survived')
# Extract Title from name 

combine = [train_df, test_df] # combining train and test dataset



for dataset in combine:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# more than 15 title  and majority of the title are in single digit

train_df['Title'].value_counts()
# Understand % split

train_df['Title'].value_counts() / len(train_df)
# Convert the titile to category columns - 0 for Mr, 1 for Miss, 2 for Mrs and 3 for rest all

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, 

                 "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3, "Ms": 3, 

                 "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }



for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)
# Recheck how, EDA is contributing to overall cleaning process

train_df.corr()['Pclass']#,'Cabin']
train_df['Title'].value_counts()
#Check the dataframe

train_df.head()
# Passenger ID and Name is not contibuting whereas all passenger will have a ticket. passenger Class and cabin details is already present, hence this columns will be dropped 

train_df = train_df.drop(['Name', 'PassengerId', 'Ticket'], axis=1)

test_df = test_df.drop(['Name', 'Ticket'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
#Convert 'Sex' to categorical column 1 for Female and 0 for Male



for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
train_df['Age'].isnull().sum()
# Age of few person is missing in dataframe, we can either delete the rows or imput via median / mean

train_df.groupby(['Survived','Pclass']).mean()['Age'], train_df.groupby(['Survived','Pclass']).median()['Age']
train_df.groupby(['Survived','Pclass','Sex']).mean()['Age'], train_df.groupby(['Survived','Pclass','Sex']).median()['Age']
# fill missing age with Mean passenger age based on Sex, PClass and survived

str_Survived = train_df['Survived'].unique()

str_Pclass = train_df['Pclass'].unique()

str_Sex = train_df['Sex'].unique()



def fillage(val):

    if(pd.isnull(val)):

        for val_Survived in str_Survived:

            for val_Pclass in str_Pclass:

                for val_Sex in str_Sex:

                    avg_Age = np.round(train_df[(train_df['Survived']==val_Survived) & (train_df['Pclass']==val_Pclass) & (train_df['Sex']==val_Sex)]['Age'].mean())

    else:

        avg_Age = val

    return avg_Age;
train_df['Age'] = train_df['Age'].apply(fillage)

test_df['Age'] = test_df['Age'].apply(fillage)



combine = [train_df, test_df]
train_df.head()
# Check Null Values

train_df.isnull().sum(), test_df.isnull().sum()
# Embarked has some missing values and is still a non - numeric columnn. Missing values needs to be either filled with value or dropped

train_df.groupby("Embarked").count()
# know values of other column for knowing the Embarked column based on features of other columns

train_df[train_df["Embarked"].isnull()]
# Unique columns looks to be Survived=1, Pclass=1, Sex=1 and Cabin=1

train_df[(train_df['Survived']==1) & (train_df['Pclass']==1) & (train_df['Sex']==1) & 

         (train_df['Cabin']==1)].groupby(['Embarked']).mean()['Fare']
train_df[(train_df['Survived']==1) & (train_df['Pclass']==1) & 

         (train_df['Cabin']==1)].groupby('Embarked')['Fare'].describe().T 
def fixEmbarked(val):

    if(pd.isnull(val)):

        return 'S'

    else:

        return val
# Missing value of Embarked to be filled with 'S'

train_df["Embarked"] = train_df["Embarked"].apply(fixEmbarked)

test_df["Embarked"] = test_df["Embarked"].apply(fixEmbarked)



# Family Size column is included

train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1

test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1



# SibSp, Parch columns is dropped

train_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

test_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Converting Embarked column to Categorical - numberic

Embarked_mapping = {"S": 0, "C": 1, "Q": 2}

    

train_df['Embarked'] = train_df['Embarked'].map(Embarked_mapping)

test_df['Embarked'] = test_df['Embarked'].map(Embarked_mapping)
train_df.head()
# For understanding, how it is grouped

min_age_limit = np.int(np.floor(np.min(train_df['Age'])/16)*16)

max_age_limit = np.int(np.ceil(np.max(train_df['Age'])/16)*16)



pd.cut(train_df['Age'],

               (range(min_age_limit,max_age_limit + 16, 16)),

               right=True).value_counts()/len(train_df)

min_age_limit = np.int(np.floor(np.min(train_df['Age'])/16)*16)

max_age_limit = np.int(np.ceil(np.max(train_df['Age'])/16)*16)



train_df['agerange'] = pd.cut(train_df['Age'],

               (range(min_age_limit,max_age_limit + 16, 16)),

               right=True)





train_df['agerange'] = train_df['agerange'].astype('str')



agerange_df = pd.get_dummies(train_df['agerange'])

train_df = pd.concat([train_df, agerange_df],axis=1)





train_df.drop(['Age','agerange'], axis=1, inplace=True)

train_df.head()

#agerange_df
test_df['agerange'] = pd.cut(test_df['Age'],

               (range(min_age_limit,max_age_limit + 16, 16)),

               right=True)



test_df['agerange'] = test_df['agerange'].astype('str')



agerange_df = pd.get_dummies(test_df['agerange'])



test_df = pd.concat([test_df, agerange_df],axis=1)



test_df.drop(['Age','agerange'], axis=1, inplace=True)

test_df.head()
train_df.head()
test_df.head()
# One row in Test Dataset still has missing values, this needs to be either imputed or deleted before modeling

test_df[(test_df['Fare'].isnull()==True)]#.apply()
test_df[(test_df['Pclass'] == 3) & (test_df['Cabin'] == 0) & 

        (test_df['Embarked'] == 0) & (test_df['FamilySize'] == 1)]['Fare'].mean()
# Update the missing value fare with mean of other columns feature

def fixFare(val):

    if(pd.isnull(val)):

        return test_df[(test_df['Pclass'] == 3) & (test_df['Cabin'] == 0) & 

                    (test_df['Embarked'] == 0) & (test_df['FamilySize'] == 1)]['Fare'].mean()

    else:

        return val
test_df['Fare'] = test_df['Fare'].apply(fixFare)
train_df.isnull().sum(), '-------', test_df.isnull().sum()
train_df.corr()['Survived']
sns.pairplot(train_df)
plt.figure(figsize=(24,8))

sns.heatmap(train_df.corr(), annot = True)
# Importing Classifier Modules

import xgboost as xgb

from sklearn import preprocessing

#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.linear_model import LogisticRegression



from imblearn.over_sampling import SMOTE # Oversampling



from sklearn.metrics import accuracy_score, f1_score, fbeta_score, make_scorer



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
'''X = train_df.drop(['Survived'], axis=1)

y = train_df['Survived']'''
train_df.columns, test_df.columns
# Survived is predicting variable and PassengerID column needs to be dropped



X_train = train_df.drop('Survived', axis=1)

y_train = train_df['Survived']



X_test = test_df.drop(['PassengerId'], axis=1)

#y_test = test_df['Survived']



random_state= 101



X_train.shape, y_train.shape, X_test.shape

#---------------------------------------------------

X_train.columns
# Since dataset is inbalanced, we need to apply SMOTE - it will include new dummy rows for analysis. This needs to be done only on train dataset 

random_state = random_state

features = X_train.columns

sm = SMOTE(random_state=random_state)#, ratio=1.0)

X_train, y_train = sm.fit_resample(X_train, y_train)

X_train = pd.DataFrame(X_train, columns=features)
# Appling MinMaxScaler (fit and transform) on train data and only transform on test data to avoid data leakages 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)



X_test = scaler.transform(X_test)
# baselining  

naive_predictor_accuracy = accuracy_score(y_train,np.ones(len(y_train)))

naive_predictor_f1score = f1_score(y_train, np.ones(len(y_train)))



print("Naive predictor accuracy: %.3f" % (naive_predictor_accuracy))

print("Naive predictor f1-score: %.3f" % (naive_predictor_f1score))
k_fold = KFold(n_splits=10, shuffle=True, random_state=random_state)
# KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# kNN Score

round(np.mean(score)*100, 2)
# DecisionTreeClassifier

clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# decision tree Score

round(np.mean(score)*100, 2)
# RandomForestClassifier

clf = RandomForestClassifier(n_estimators=13)

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Random Forest Score

round(np.mean(score)*100, 2)
# GaussianNB

clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Naive Bayes Score

round(np.mean(score)*100, 2)
# SVC

clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
# Working model



gbm = xgb.XGBClassifier(max_depth=3, n_estimators=400, learning_rate=0.05).fit(X_train, y_train)

predictions = gbm.predict(X_test)



# Kaggle needs the submission to have a certain format;

# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv

# for an example of what it's supposed to look like.

submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submission.csv", index=False)



submission = pd.read_csv('submission.csv')

submission['Survived'].value_counts()