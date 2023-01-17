# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Here I am performing an EDA  of the titanic dataset.Mainly trying to figure out the factors that influenced 
#the survival rate by drawing different plots and queries.The main details given in the description are:
#1.On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 
#1502 out of 2224 passengers and crew. Translated 32% survival rate.
#2.Although there was some element of luck involved in surviving the sinking, some groups of people were 
#more likely to survive than others, such as women, children, and the upper-class.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
titanic=pd.concat([train_df,test_df],axis=0,sort=False)
#we will combine both train and test data test to do eda.
titanic.head()
titanic.info()
#It is observed from the count that there are null values in Survived,Age,Cabin,Fare and embarked.
titanic.describe(include='all')
#There are 4 categorical columns:embarked,survived,sex,Pclass.Ticket is a mix of numeric and 
#alphanumeric data types. Cabin is alphanumeric.Numerical:AGE,FARE(continuous),SibSp,Parch(discrete.)
titanic['Family']=titanic['SibSp']+titanic['Parch']
titanic[['Family','SibSp','Parch']].head()
pd.isnull(titanic).sum()
#There are lot of null values in Survived,Age,cabin.Let us impute the values for the age column.
titanic[['Age','Pclass']].groupby('Pclass').mean()
#From the figure it is observed that the mean age of class 1 is 39,class 2 is 29 and class 3 is 25 approximately.
def calc_age(col):
    Age = col[0]
    Pclass = col[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29 
        else:
            return 25
    else:
        return Age
titanic['Age']=titanic[['Age','Pclass']].apply(calc_age,axis=1)
titanic['Survived'].value_counts()
#OUt of 891 people in the sample,we know that 342 has survived.That is a survival rate of 38%.
labels={1:'First class',
       2:'Second class',
       3:'Third class'}
titanic['Pclass']=titanic['Pclass'].replace(labels)
titanic['Pclass'].value_counts()
pc=(titanic['Pclass'].value_counts()/titanic.shape[0])*100
pc.plot.bar(color='steelblue',figsize=(12,5))
#Most people were travelling in third class.
titanic['Sex'].value_counts()
sc=(titanic['Sex'].value_counts()/titanic.shape[0])
sc.plot.bar(color='red')
#Nearly 65% of the passengers were male.
titanic['Embarked'].value_counts()
ec=(titanic['Embarked'].value_counts()/titanic.shape[0])
ec.plot.bar(color='green')
#Nearly 70% of the passengers embarked from 'S'
age_bins = [18,30, 60, 90]
labels = {0: 'kids',
          1: 'youth',
          2: 'elders',
          3: 'senior citizen'}
titanic['Age_bin'] = titanic['Age'].apply(lambda v: np.digitize(v, bins=age_bins))
titanic['Age_bin'] = titanic['Age_bin'].replace(labels)
titanic['Age_bin'].value_counts()
#Age vs sex
titanic.boxplot('Age',by='Sex',figsize=(10,5),rot=10)
titanic[['Age','Sex']].groupby('Sex').mean()
#We can see that the average age of both male and female passengers lie between 28-30.There are a number 
#of ouliers representing elder citizens.
#Age vs class
titanic.boxplot('Age',by='Pclass',figsize=(10,5))
titanic[['Age','Pclass']].groupby('Pclass').mean()
#Fare vs class
titanic.boxplot('Fare',by='Pclass',figsize=(10,5))
titanic[['Fare','Pclass']].groupby('Pclass').mean()
#First class tickets are very costly compared to third class as expected.
#Fare vs Agebin
titanic.boxplot('Fare',by='Age_bin',figsize=(10,5),rot=10)
titanic[['Fare','Age_bin']].groupby('Age_bin').mean()
#Fare is mainly dependent on the class than the age.
titanic[['Age','Fare']].corr()
sns.set(rc={'figure.figsize':(10,5)})
p = sns.heatmap(titanic[['Age','Fare']].corr(), cmap='Blues')
#Correlation does not exist between age and fare.
#Survived vs Sex
obs = titanic.groupby(['Survived', 'Sex']).size()
obs.name = 'Freq'
obs = obs.reset_index()
obs = obs.pivot_table(index='Survived', columns='Sex',
                values='Freq')
sns.heatmap(obs, cmap='CMRmap_r')
titanic[['Survived','Sex']].groupby('Sex').mean()
#The survival rate is high in females and very low in males.
#Survived vs Age
obs = titanic.groupby(['Survived', 'Age_bin']).size()
obs.name = 'Freq'
obs = obs.reset_index()
obs = obs.pivot_table(index='Survived', columns='Age_bin',
                values='Freq')
sns.heatmap(obs, cmap='CMRmap_r')
titanic[['Survived','Age_bin']].groupby('Age_bin').mean().reset_index().sort_values(by='Survived',ascending=False)
#Survival rate is highest in kids and least in senior citizens.
##Survived vs Class
obs = titanic.groupby(['Survived', 'Pclass']).size()
obs.name = 'Freq'
obs = obs.reset_index()
obs = obs.pivot_table(index='Survived', columns='Pclass',
                values='Freq')
sns.heatmap(obs, cmap='CMRmap_r')
titanic[['Survived','Pclass']].groupby('Pclass').mean().reset_index().sort_values(by='Survived',ascending=False)
#First class people has the highest survival rate.