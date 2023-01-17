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
#import dependencies 

%matplotlib inline



#python imports

import math

import time

import random

import datetime





#data manipilation

import numpy as np

import pandas as pd



#Visualisation

import matplotlib.pyplot as plt

import missingno

import seaborn as sns



plt.style.use('seaborn-whitegrid')



#import train,test and gender submission data in notebook

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
#view traing data

train.head()
#view test data

test.head()
#view gender submission

gender_submission.head()
#plot graphic of missing values

missingno.matrix(train,figsize =(30,10))
#calulualte no of missing values 

def find_missing_values(df,columns):

    missing_vals= {}

    print("no of missing and Nan values ")

    df_length=len(df)

    for column in columns:

        total_column_values=df[column].value_counts().sum()

        missing_vals[column]=df_length - total_column_values

    return  missing_vals



missing_values=find_missing_values(train, columns=train.columns)

missing_values
df_bin = pd.DataFrame() #for discrete continuous variables

df_con = pd.DataFrame() #for continuous variables
#how many people survived?

fig= plt.figure(figsize=(20,2))

sns.countplot(y='Survived',data=train)

print(train.Survived.value_counts())
df_bin['Survived']=train['Survived']

df_con['Survived']=train['Survived']
df_bin.head()


figp=plt.figure(figsize=(20,2))

sns.countplot(y='Pclass',data=train)

print(train.Pclass.value_counts())
sns.distplot(train.Pclass)
df_bin['Pclass']=train['Pclass']

df_con['Pclass']=train['Pclass']
df_con.head()
#Check how many dupplicate names are there in dataset

train.Name.value_counts()
#sex of the passenger male or female

plt.figure(figsize=(20,2))

sns.countplot(y='Sex',data=train)  
#check missing values for sex 

missing_values['Sex']
#add sex to subset dataframe

df_bin['Sex']=train['Sex']



df_bin['Sex']=np.where(df_bin['Sex']== 'female', 1, 0)



df_con['Sex']=train['Sex']





#how Sex variable look compared to surviaval 

fig=plt.figure(figsize=(10,10))

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label':'Survived'},kde=0)

sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label':'Did not Survived'},kde=0)



#how many missing values of Age have?

missing_values['Age']
missing_values['SibSp']
(train['SibSp']).value_counts()
df_bin['SibSp']=train['SibSp']

df_con['SibSp']=train['SibSp']
#function to plot distribution and count plot of labeled variables and target variable 

def plot_dist_or_count(data,bin_df,label_column,target_column,figsize=(20,5),use_bin_df=False):

    if use_bin_df:

        fig=plt.figure(figsize=figsize)

        plt.subplot(1,2,1)

        sns.countplot(y=target_column,data=bin_df)

        plt.subplot(1,2,2)

        sns.distplot(data.loc[data[label_column] ==1 ][target_column],kde=0,kde_kws={'label':'Survived'})

        sns.distplot(data.loc[data[label_column] == 0][target_column],kde=0,kde_kws={'label':'Did not Survived'})

    else:

        fig=plt.figure(figsize=figsize)

        plt.subplot(1,2,1)

        sns.countplot(y=target_column,data=data)

        plt.subplot(1,2,2)

        sns.distplot(data.loc[data[label_column] ==1 ][target_column],kde=0,kde_kws={'label':'Survived'})

        sns.distplot(data.loc[data[label_column] == 0][target_column],kde=0,kde_kws={'label':'Did not Survived'})

        
#lets visualize the count of siblings and spouse with survived category

plot_dist_or_count(train,bin_df=df_bin,label_column='Survived',target_column='SibSp',figsize=(20,10))
#lets calculate missing values in Parch

missing_values['Parch']
#lets pass them to our testing data frames

df_bin['Parch']=train['Parch']

df_con['Parch']=train['Parch']
#lets visualize the count of parents and childern with survived category

plot_dist_or_count(train,bin_df=df_bin,label_column='Survived',target_column='Parch',figsize=(20,10))
#how many missing values ticket column have

missing_values['Ticket']
train.Ticket.value_counts()[:20]
#lets plot a count plot of ticket 

sns.countplot(y='Ticket',data=train)
#how much ticket cost its a fare 

#lets calculate missing values in fare

missing_values['Fare']
sns.countplot(y='Fare',data=train)
#cost fare categories with count 

train.Fare.value_counts()
#lets find unique no of fares

print("there are {} unique fare values ".format(len(train.Fare.unique())))
#lets do grouping for fare in our test data frames

df_con['Fare']=train['Fare']

df_bin['Fare']=pd.cut(train['Fare'],bins=5) # for discrete distribution we divide it in 5 classes

df_bin.head()
#lets visualize fare distribution with survived ones.

plot_dist_or_count(train,bin_df=df_bin,label_column='Survived',target_column='Fare',figsize=(20,10),use_bin_df=True)
#lets see missing values from embarked

missing_values['Embarked']
train.Embarked.value_counts()
sns.countplot(y='Embarked',data=train)
#lets add to our testing data set

df_bin['Embarked']=train['Embarked']

df_con['Embarked']=train['Embarked']
#drop  missing values 

df_bin=df_bin.dropna(subset=['Embarked'])

df_con=df_con.dropna(subset=['Embarked'])

print(len(df_con))

df_bin.head()
df_con.head()
train.head()
test.head()
#les compute features for prediction and classificaton of dataset

features = ['Pclass','Sex','SibSp','Parch','Embarked']



y=train['Survived']



X=pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])

from sklearn.model_selection import train_test_split



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
from sklearn.ensemble import RandomForestClassifier





# Define the models

model_1 = RandomForestClassifier(n_estimators=50, random_state=0)

model_2 = RandomForestClassifier(n_estimators=100, random_state=0)

model_3 = RandomForestClassifier(n_estimators=100,criterion='gini', random_state=0)

model_4 = RandomForestClassifier(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)



models = [model_1,model_2,model_3,model_4,model_5]





from sklearn.metrics import mean_absolute_error



# Function for comparing different models

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1,mae))
my_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0) 

my_model.fit(X, y)



# Generate test predictions

preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring

output = pd.DataFrame({'PassengerId': test.PassengerId,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)
