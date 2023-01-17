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
#This would import the necessary visualization labraries

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
#Now lets load up our data set

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
#Lets check out the head of our training data set

train.head()
#Lets check out the head of our testing data set

test.head()
#Lets check out some more information of our data set



train.info()
train.describe()
test.info()
#Lets check out the missing data in our data set

train.isnull()
'''What we get a bunch of booleans values, that doesn't give us much sense,

   hence we are going to plot these boolean values into a plot through seaborn'''



sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')
#Lets check out about the survived column using the countplot

sns.countplot(x='Survived',data=train,palette='ocean')
#Lets check the same countplot this time using sex as hue

sns.countplot(x='Survived',data=train,palette='ocean',hue='Sex')
#Lets check out the survived trend in terms of Pclass

sns.countplot(x='Pclass',data=train,hue='Survived',palette='plasma')
#Lets us now plot the passenger class 

sns.countplot(x='Pclass',data=train)
#Now we want visualize the Age column of the data set using histogram

#It seems that it is bimodial distribution

train['Age'].plot.hist(bins=30)
#Now lets check out the spread of the Fare 

plt.figure(figsize=(8,4))

train['Fare'].plot.hist(bins=50)

plt.xlabel('Fare')
#Dealing with the missing values

#Lets check our heatmap once again



sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#In this scinario we could take the mean values of the age and then replace those nan values with

#this mean values, or we could do relate age column with Pclass and lets see what we get



#lets plot a boxplot pclass vs age



plt.figure(figsize=(10,6))

sns.boxplot(x='Pclass',y='Age',data=train,palette='rainbow')
'''From the above boxplot we can observe that the age column is somehow related to the pclass

column, it seems that the first class passengers are more older than the second class, again the 

second class passengers are more older than the third class. This makes sense because wealth comes

after a certain level of age

'''



#Considering the above boxplot now lets write a function to deal with missing values in age column



def imput_age(cls):

    Age = cls[0]

    Pclass = cls[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 38

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age



#Now use this imput functions into our Age column



train['Age'] = train[['Age', 'Pclass']].apply(imput_age,axis=1)
#now lets check our function worked correctly or not

#it seems it did a pretty amazing job



sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#now lets drop the cabin column



train.drop('Cabin', axis=1,inplace=True)
#lets check one more time for missing values

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
Sex = pd.get_dummies(train['Sex'], drop_first=True)

Embarked = pd.get_dummies(train['Embarked'], drop_first=True)

train.drop(['Sex', 'Embarked','Name','Ticket'],axis=1, inplace=True)

train = pd.concat([train, Sex, Embarked], axis=1)

Sex = pd.get_dummies(test['Sex'], drop_first=True)

Embarked = pd.get_dummies(test['Embarked'], drop_first=True)

test.drop(['Sex', 'Embarked','Name','Ticket'],axis=1, inplace=True)

test = pd.concat([test, Sex, Embarked], axis=1)

test['Age'] = test[['Age', 'Pclass']].apply(imput_age, axis=1)
test['Fare']= test['Fare'].fillna(test['Fare'].mean())

test.drop('Cabin',axis=1,inplace=True)
test.isnull().sum()
train.isnull().sum()
X_train = train.drop('Survived', axis=1)

X_test = test.copy()

y_train = train['Survived']
# from sklearn.linear_model import LogisticRegression

# logmodel = LogisticRegression()

# logmodel.fit(X_train,y_train)

# prediction = logmodel.predict(X_test)

# logmodel.score(X_train,y_train)
# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=5)

# knn.fit(X_train,y_train)

# prediction = knn.predict(X_test)

# # acc_knn = round(knn.score(X_train,y_train)*100, 2)

# # acc_knn

# knn.score(X_train,y_train)
# from sklearn.tree import DecisionTreeClassifier

# dtree = DecisionTreeClassifier()

# dtree.fit(X_train,y_train)

# prediction = dtree.predict(X_test)

# dtree.score(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier(n_estimators=100)

rforest.fit(X_train,y_train)

prediction = rforest.predict(X_test)

rforest.score(X_train,y_train)
submission = pd.DataFrame({ 'PassengerId' : test['PassengerId'],

                           'Survived': prediction})



submission.to_csv('titanic.csv', index=False)