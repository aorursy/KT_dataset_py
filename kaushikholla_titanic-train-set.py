# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/titanic/train.csv")
#Checking number of NAN from Info

train.info()

train.head()
# Exploratory Data analysis



#Checking number of Nan values from heatmap

sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
#Checking number of survived

#Blue represents not survived and 1 represents survived



sns.countplot(x = 'Survived', hue='Sex', data=train)



#We can see that men are the most among people who didnt survive and Female are the most among people who survived
#To check the relation between people who survived and there class



sns.countplot(x='Survived', hue='Pclass',data=train)



#We can see that the people who didnt survive most belong to 3rd Class. 

#3rd class had the cheapest ticket price
#To check for distribution of Age.

#PLotting distplot



#for getting the background grid lines 

sns.set_style('whitegrid')



#Plotting the graph

#kde draws the curved line around the plots 

sns.distplot(train['Age'].dropna(),kde=False,bins=30)



#Many passengers around 20-30 years
#Analysing the next column to see if i can find any info.

#SibSp

#To check how many siblings people onboard had



sns.countplot(x='SibSp',data=train)



#From the plot we can see that most people had no siblings.
#To check if people who had siblings and Spouse had any effect on survival



sns.countplot(x='Survived',hue='SibSp',data=train)

#Exploring the Fare column



#Plotting the graph



sns.distplot(train['Fare'],kde=False,bins=30)



#As we alreaddy know most people were in 3rd class hence the price paid is also less for them.
#Filling and dropping missing data



#Fill the age by the mean of the class they belong to.



#Checking for avg value of each type of class



sns.boxplot(x='Pclass',y='Age',data=train)

#Replacing the missing values in each class by avg value of the class



#Defining function to add avg age in place of missing value



#Sending Age. If age is null return the avg. If age is not null then return avg age.



def adding_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return 37

        

        if Pclass == 2:

            return 29

        

        if Pclass == 3:

            return 24

    else:

        return Age

    

train['Age'] = train[['Age','Pclass']].apply(adding_age,axis=1)
#Plotting heatmap to see how it looks after filling in the values for age

sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
#Dropping Cabin as there are too many missing values



train.drop('Cabin',axis=1,inplace=True)

train.head()
#Dropping the missing values in embark

train.dropna(inplace = True)
train.info()

#There were two Nan values in Embark so 2 rows were dropped. Now data from 991 has come down to 889
#Creating dummies for sex column. Dropping First row to avoid Multicollinearity

sex = pd.get_dummies(train['Sex'],drop_first = True)
#Creating dummies for embarked column. Dropping First row to avoid Multicollinearity

embarked = pd.get_dummies(train['Embarked'], drop_first = True)
#Dropping the categorical columns

train.drop(['Sex','Embarked','Name','Ticket'], axis = 1, inplace = True)
train = pd.concat([train,sex,embarked],axis = 1)
train.head()
#Fitting the logistic regression model



X = train.drop('Survived',axis=1)

y = train['Survived']
from sklearn.model_selection import train_test_split

X_train, Xtest, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
prediction = logmodel.predict(Xtest)
#Moving to evaluation as model has been created and predicted



from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,prediction)
from sklearn.metrics import classification_report

print(classification_report(y_test,prediction))
test = pd.read_csv("../input/titanic/test.csv")
test.info()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test.drop("Cabin",axis=1,inplace=True)
test.info()
sns.boxplot(x='Pclass',y='Age',data=test)
def fill_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 42

        if Pclass == 2:

            return 25

        if Pclass == 3:

            return 23

    else:

        return Age

    

test['Age'] = test[['Age','Pclass']].apply(fill_age,axis=1)
test.info()
test.dropna(inplace=True)
test.info()
sex = pd.get_dummies(test['Sex'],drop_first = True)

embarked = pd.get_dummies(test['Embarked'],drop_first = True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test.info()
test = pd.concat([test,sex,embarked],axis = 1)
test.head()
test_predict = logmodel.predict(test)
test_predict
np.savetxt("Survival.csv", test_predict, delimiter = '\n')
data={'PassengerId':test['PassengerId'],'Survived':test_predict}
df = pd.DataFrame(data)
np.savetxt("Survival1.csv", df)
df