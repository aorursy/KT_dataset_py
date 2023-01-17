# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.columns
#to get a perspective on the missing data

sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')
train.head()
#countplot animal type

plt.figure(figsize=(14,6))

sns.countplot(train['AnimalType'])
#countplothue sex upon outcome

plt.figure(figsize=(14,6))

sns.countplot(train['SexuponOutcome'])
#to sort the breeds into mix and pure



#first create a list to store the data

new = []

#for each row in the column

for item in train['Breed']:

    if 'Mix' in item.split(): #is the word 'Mix' is in the row

        new.append('mixed') #append list to say mixed

    else:

        new.append('pure') # if 'Mix is not in the row, then append to say pure

        

#add the new list to our column in the dataframe

train['nBreed'] = new
#countplot breed type

plt.figure(figsize=(14,6))

sns.countplot(train['nBreed'])
#countplot Outcome type

plt.figure(figsize=(14,6))

sns.countplot(x='OutcomeType',data=train)
# compare outcome type by animal type

plt.figure(figsize=(14,6))

sns.countplot(x='AnimalType',data=train, hue='OutcomeType')
#countplot Outcome type (hue sex upon outcome)

plt.figure(figsize=(14,6))

sns.countplot(x='SexuponOutcome',data=train,hue='OutcomeType')
#countplot breed type (hue OutcomeType)

sns.countplot(x='nBreed', data=train, hue='OutcomeType')
train.head()
#CATEGORISE OUTCOME INTO HAPPY OR SAD for ML

train['OutcomeType'].unique()
train.head()
#first drop all the irrelevant columns

train.drop(['AnimalID','Name','DateTime','OutcomeSubtype','AgeuponOutcome','Breed','Color'],axis=1,inplace=True)
train.head()
#create dummy values

outcome = pd.get_dummies(train['OutcomeType'])

animal = pd.get_dummies(train['AnimalType'],drop_first=True)

sex = pd.get_dummies(train['SexuponOutcome'],drop_first=True)

breed = pd.get_dummies(train['nBreed'],drop_first=True)
#time to join all the columns together

train = pd.concat([train,outcome,animal,sex,breed],axis=1)
train.head()
#drop all unnecessary data

train.drop(['OutcomeType','AnimalType','SexuponOutcome','nBreed','Transfer','Euthanasia','Return_to_owner','Died'],axis=1,inplace=True)
#Data is ready for ML algorithm

train.head()
sns.countplot(train['Adoption'])
#divide data into selected input (x), and expected outputs (y)

x = train[['Dog','Intact Male','Neutered Male','Spayed Female','Unknown','pure']]

y = train['Adoption']
from sklearn.cross_validation import train_test_split
#split into train and test data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
#Import logistic regression module

from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(x_train,y_train)
#test the model

prediction = lm.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, prediction))