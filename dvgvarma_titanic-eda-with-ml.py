# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

dfTrain= pd.read_csv('../input/train.csv')
dfTest= pd.read_csv('../input/test.csv')

# Explore Traainig Data
dfTrain.head()

#Explore Test Data
dfTest.head()
dfTrain.info()
print('_'*40)
dfTest.info()
#Check Any missing values
dfTrain.isnull().sum()

# Dropping columns with large missing values & Unusefull information (PassengerId,Ticket & Cabin)
dfTrain.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace=True)
dfTrain.head()

# Dropping columns with large missing values & Unusefull information (PassengerId,Ticket & Cabin)
dfTest.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace=True)
dfTest.head()
# Lets Fill Missing Values using mean
# Lets Calculate Mean value for Age for Category Male & Female
avgfemaleTrain=dfTrain[dfTrain.Sex=='female']['Age'].mean()
avgmaleTrain=dfTrain[dfTrain.Sex=='male']['Age'].mean()

avgfemaleTest=dfTest[dfTest.Sex=='female']['Age'].mean()
avgmaleTest=dfTest[dfTest.Sex=='male']['Age'].mean()

dfTrain.loc[dfTrain.Sex=='female','Age']=dfTrain.loc[dfTrain.Sex=='female','Age'].fillna(avgfemaleTrain)
dfTrain.loc[dfTrain.Sex=='male','Age']=dfTrain.loc[dfTrain.Sex=='male','Age'].fillna(avgmaleTrain)

dfTest.loc[dfTest.Sex=='female','Age']=dfTest.loc[dfTest.Sex=='female','Age'].fillna(avgfemaleTest)
dfTest.loc[dfTest.Sex=='male','Age']=dfTest.loc[dfTest.Sex=='male','Age'].fillna(avgmaleTest)

# Fill missing Fair using median
dfTrain["Fare"].fillna(dfTrain["Fare"].median(), inplace=True)
dfTest["Fare"].fillna(dfTrain["Fare"].median(), inplace=True)
#Lets check any missing values still appears or not
dfTrain.isnull().sum()
# Lets check which category of people survived
dfTrain['Title']=dfTrain.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dfTrain.groupby('Title')['Survived'].sum()
# Lets do the same on Test Set
dfTest['Title']=dfTest.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dfTest.drop(['Name'],axis=1,inplace=True)
dfTest.head()
# Lets Drop Column Name as we extracted Title from the Name which is more meaningfull
dfTrain.drop(['Name'],axis=1,inplace=True)
dfTrain.head()
# Give proper names to Tiles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
dfTrain['Title']=dfTrain.Title.map(Title_Dictionary)
dfTest['Title']=dfTest.Title.map(Title_Dictionary)
dfTrain.groupby('Title')['Survived'].sum()

# Lets See Which is the most frequent Embarked from
dfTrain.Embarked.value_counts()
# Lets Fill with missing Embarked with 'S'
dfTrain.Embarked.fillna('S',inplace=True)

#Check Missing Values if Any
dfTrain.isnull().sum()
# Lets Do some Visual Data Exploration
# Setting SNS Grid Color
sns.set(style="darkgrid")
sns.countplot(x='Survived',data=dfTrain,palette='Set1')
plt.title('Non-Survived vs Survived')
plt.show()


# Lets see people who survived based on Age at an Embarked Location
sns.FacetGrid(dfTrain,col='Embarked',row='Survived',size=4).map(plt.hist,'Age').add_legend()
plt.show()
# Lets see one more time without Age

g=sns.FacetGrid(dfTrain,col='Embarked',hue='Survived',size=5)
g.map(plt.hist,'Survived').set(xticks=[0,1]).add_legend()
plt.show()
# Lets see which class people survived most by Age
plt.figure(figsize=(20,8))
ax=sns.stripplot(x='Pclass',y='Age', data=dfTrain,hue='Survived',jitter=True,dodge=True)
plt.show()
# Lets go little Deeper and See which Category of Sex Had Survived in Which Class
sns.factorplot(x='Sex',y='Age',data=dfTrain,col='Pclass',hue='Survived',kind='strip',jitter=True,size=6)
plt.show()
# Lets see Statistics of Survived in each class
pd.crosstab(dfTrain['Pclass'],dfTrain['Survived'])
Titles=pd.get_dummies( dfTrain.Title )
dfTrain=dfTrain.join(Titles)
dfTrain.drop(['Title'],axis=1,inplace=True)

#Lets Make Categorical Variable 'Sex' to Numeric '0' or '1'
dfTrain['Sex']=dfTrain['Sex'].map({"male":0,"female":1})
dfTrain.head()

dfTrain=pd.get_dummies(dfTrain)
dfTrain.head()
Titles=pd.get_dummies( dfTest.Title )
dfTest=dfTest.join(Titles)
dfTest.drop(['Title'],axis=1,inplace=True)

dfTest['Sex']=dfTest['Sex'].map({"male":0,"female":1})
dfTest.head()

dfTest=pd.get_dummies(dfTest)
dfTest.head()
#Lets Build a model

logreg=LogisticRegression()

X_train=dfTrain.drop('Survived',axis=1)
y_train=dfTrain['Survived']

logreg.fit(X_train,y_train)
y_pred=logreg.predict(dfTest)

logreg.score(X_train,y_train)
