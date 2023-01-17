# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
train.head()
# Exploratory Data Analysis:

sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)

#Thus more female survived as compared to male.
sns.countplot(x='Survived',hue='Pclass',data=train)

#Thus majority people survived were from class 1
sns.distplot(train['Fare'],kde=False,bins=25)

#Thus the majority of the people belong to the class 3 as the Fare price is Low.  
sns.countplot(x='SibSp',data=train)

# Majority of the people were without any spouse and without siblings.

#Means they were young.
sns.distplot(train['Age'],kde=False,bins=30)

# Majority of the people were of the Age between 20-30 and as seen above were without siblings and spouse.
# Now lets have a look at the missing data.

sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')

# As you can see the Age and the Cabin column have the missing values.
# As we cannot predict the Cabin let's drop it but we can predict the Age.

train.drop('Cabin',axis=1,inplace=True)
#Now for the Age lets take the mean of the age of the people in each Pclass.

plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass',y='Age',data=train)

def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 25

    else:

        return Age

        
train['Age']= train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')

# No null value remains.
train.head()
# Converting Categorical Values.

Sex=pd.get_dummies(train['Sex'],drop_first=True)
Embark = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,Sex,Embark],axis=1)
train.head()

# As we can see there is no need of the Embarked and age so lets drop Emabarked & Sex
train.drop(['Sex','Embarked'],axis=1,inplace=True)
train.head()

#Now as you can see the data only contains numerical data, so we can apply machine learning algorithm 
from sklearn.model_selection import train_test_split
X = train.drop('Survived',axis=1)

y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
pred=logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))

print('/n')

print(classification_report(y_test,pred))