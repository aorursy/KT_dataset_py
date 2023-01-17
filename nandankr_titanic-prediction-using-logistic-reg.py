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
import seaborn as sns

import matplotlib as plt

%matplotlib inline

import math



titanic_data=pd.read_csv('/kaggle/input/titanicdataset-traincsv/train.csv')



titanic_data.head(10)
print("totsl no. of passenger in data set :"+str(len(titanic_data)))

print(titanic_data.shape)


sns.countplot(x='Survived' , data=titanic_data)
sns.countplot(x='Survived' , hue='Sex' , data=titanic_data)

sns.countplot(x='Survived' , hue='Pclass' , data=titanic_data)

titanic_data['Age'].plot.hist()

titanic_data['Fare'].plot.hist(bins=20,figsize=(15,5))

titanic_data.info()

sns.countplot(x='SibSp' , data=titanic_data)

titanic_data.isnull()

titanic_data.isnull().sum()

sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap="viridis")

sns.boxenplot(x="Pclass" , y="Age" , data=titanic_data)

titanic_data.head()

titanic_data.drop("Cabin" , axis=1,inplace=True)

titanic_data.head()

titanic_data.dropna(inplace=True)

sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap="viridis")

titanic_data.isnull().sum()

titanic_data.head()

sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)

sex.head()
embarked=pd.get_dummies(titanic_data['Embarked'],drop_first=True)

embarked.head()


pcal=pd.get_dummies(titanic_data['Pclass'],drop_first=True)

pcal.head()
titanic_data=pd.concat([titanic_data,sex,embarked,pcal],axis=1)

titanic_data.head()
titanic_data.drop(['PassengerId' ,'Sex' ,'Name' , 'Ticket' , 'Embarked','Pclass'],axis=1,inplace=True)



titanic_data.head()
##Train and test the Data
# independent variable

X=titanic_data.drop("Survived" ,axis=1)

# dependent variable

y=titanic_data["Survived"]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)

from sklearn.metrics import classification_report

classification_report(y_test,predictions)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,predictions)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)*100



from sklearn.metrics import precision_score

precision_score(y_test,predictions)
from sklearn.metrics import f1_score

f1_score(y_test,predictions)
from sklearn.metrics import recall_score

recall_score(y_test,predictions)
# pls do upvote as well you you think this is beneficial for you 