# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")



train.head()
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train)
sns.countplot(x = 'Survived', hue = 'Sex', data = train)
sns.countplot(x='Survived',hue='Pclass',data=train)
sns.distplot(train['Age'].dropna(),kde=False,bins=40)
sns.countplot(x='SibSp',data=train)
sns.distplot(train['Fare'],kde=False,bins=40, color = 'darkgreen')
plt.figure(figsize = (10,6))



sns.boxplot(x = 'Pclass', y = 'Age', data = train)
# define average age according to Pclass



def defage(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(defage,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
train.head()
train.drop('Cabin',axis=1,inplace=True)
# now get_dummies used to remove categorical variables 



sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))