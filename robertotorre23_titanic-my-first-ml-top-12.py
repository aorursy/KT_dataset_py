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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

pd.set_option('display.max_rows',None)
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
test.info()
print('Total by Sex')

print(train.Sex.value_counts())

print('\nTotal Survived by Sex')

print(train.loc[train.Survived==1].Sex.value_counts())
#let's divided the age and search for a pattern

train_with_age=train.query('Age!="NaN"')

print('Total Survived by Age under 10')

print(train_with_age.loc[train_with_age.Age <= 10].Survived.value_counts())

print('\nTotal Survived by Age between 10 and 20')

print(train_with_age.loc[(train_with_age.Age > 10)&(train_with_age.Age <= 20)].Survived.value_counts())

print('\nTotal Survived by Age between 20 and 30')

print(train_with_age.loc[(train_with_age.Age > 20)&(train_with_age.Age <= 30)].Survived.value_counts())

print('\nTotal Survived by Age between 30 and 45')

print(train_with_age.loc[(train_with_age.Age > 30)&(train_with_age.Age <= 45)].Survived.value_counts())

print('\nTotal Survived by Age above 45')

print(train_with_age.loc[(train_with_age.Age > 45)].Survived.value_counts())
print('Total by Pclass')

print(train.Pclass.value_counts())

print('\nTotal Survived by Pclass')

print(train.loc[train.Survived==1].Pclass.value_counts())
print('Not survived Fare mean')

print(train.loc[train.Survived==0].Fare.mean())

print('\nSurvived Fare mean')

print(train.loc[train.Survived==1].Fare.mean())
complete=pd.concat([train,test],ignore_index=True)
complete.head(10)
complete.describe()
#cabin and ticket columns seem to not have importance, then I drop'em

complete.drop(['Cabin','Ticket'],axis=1,inplace=True)
#we can extract important information in the name column, between the comma and dot

#this code I get from https://www.kaggle.com/ashish2070/titanic-survival-predictions-beginner

complete['Title']=complete.Name.str.extract(' ([A-Za-z]+)\.')
#here I want to see if there is a correlation between Age and other columns, with the purpose to fill the missing ages

with_age=complete.query('Age!="NaN"')

sns.regplot(x='SibSp',y='Age',data=with_age)
#checking Age and Title correlation

plt.figure(figsize=(8,8))

sns.barplot(y='Title', x='Age', data=with_age)
sns.regplot(x='Fare',y='Age',data=with_age)
#that are to many Fare outliers, lets remove them

q1_fare=complete.Fare.quantile(0.25)

q3_fare=complete.Fare.quantile(0.75)

IQR=q3_fare-q1_fare

min_val=q1_fare-(IQR*1.5)

max_val=q3_fare+(IQR*1.5)

print('Minimum: {}'.format(min_val))

print('Maximum: {}'.format(max_val))
sns.regplot(x='Fare',y='Age',data=with_age.loc[with_age.Fare<=max_val])
#filling missing ages

for row in range(len(complete)):

    if np.isnan(complete.loc[row,'Age'])==True:

        complete.loc[row,'Age']=complete.loc[(complete.Title==(complete.loc[row,'Title'])) & (complete.SibSp==(complete.loc[row,'SibSp']))].Age.mean()
#filling missing fares

complete.Fare.fillna(complete.Fare.mean(),inplace=True)
complete.info()
#checking why Age still missing data

complete.loc[complete.Age.isna()]
#I choose to fill them with their Title mean age

for row in range(len(complete)):

    if np.isnan(complete.loc[row,'Age'])==True:

        complete.loc[row,'Age']=complete.loc[complete.Title==(complete.loc[row,'Title'])].Age.mean()
#checking again

complete.info()
#dropping the Name and Embarked columns

complete.drop(['Name','Embarked'],axis=1,inplace=True)
#transforming 'male' and 'female' into numbers

def sex(x):

    if x == 'male':

        return 0

    else:

        return 1



complete['Sex']=complete.Sex.apply(sex)
def age(x):

    if x <= 10:

        return 0

    elif x <= 20:

        return 1

    elif x <= 30:

        return 2

    elif x <= 45:

        return 3

    else:

        return 4

    

complete['Age']=complete.Age.apply(age)
#splitting the data into the original form

train=complete.loc[0:890]

test=complete.loc[891:]
#checking if the split was right

train.tail()
test.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
#I used this function to test the different parameters and ran it with the loop in the bottom

def tuning_random_forest(MaxLeafNodes,MaxDepth,NEstimators):

    model=RandomForestClassifier(random_state=1,max_leaf_nodes=MaxLeafNodes,max_depth=MaxDepth,n_estimators=NEstimators)

    X=train[['Age','Fare','Pclass','Sex']]

    y=train.Survived

    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1)

    model.fit(X_train,y_train)

    predicted=model.predict(X_test)

    accuracy=accuracy_score(y_test,predicted)

    if accuracy > 0.8:

        print('Max Leaf Nodes:{}   Max Depth:{}   N Estimators:{}'.format(MaxLeafNodes,MaxDepth,NEstimators))

        print('Accuracy: {}'.format(accuracy_score(y_test,predicted)))

        print('\n')

        

        

#for i in range(20,60,5):

#    for j in range(10,90,10):

#        for k in range(50,1000,50):

#            tuning_random_forest(i,j,k)
model=RandomForestClassifier(random_state=1,max_leaf_nodes=30,max_depth=10,n_estimators=200)

X=train[['Age','Fare','Pclass','Sex']]

y=train.Survived

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1)

model.fit(X_train,y_train)

predicted=model.predict(X_test)

accuracy_score(y_test,predicted)
#trying to fit better, I will use all train data

test2=test.loc[:,['Age','Fare','Pclass','Sex']]

X=train[['Age','Fare','Pclass','Sex']]

y=train.Survived

model.fit(X,y)

predicted=model.predict(test2)
test2=test.loc[:,['PassengerId']]

submission=pd.DataFrame({'PassengerId':test2.PassengerId,'Survived':predicted})

submission=submission.astype('int32')

submission.to_csv('submission.csv',index=False)