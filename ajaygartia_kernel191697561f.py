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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

data=pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
##Lets observe our dataset and find the correlations with each other

data.describe()

correlations=data.corr()

sns.heatmap(data=correlations,annot=True)

##Lets check the Null Values

data.isnull().sum()

##1.Data Preprocessing for Null Values



x=data.iloc[:,5:6]

from sklearn.preprocessing import Imputer

imp=Imputer()

imp.fit(x)

x=imp.transform(x)

data['Age']=x

##adding most frequest data for Embarked column

data['Embarked']=data['Embarked'].fillna(data['Embarked'].mode().iloc[0])
data.isnull().sum()
##Transform Categorical data to Int

data['Sex']=pd.get_dummies(data['Sex'],drop_first=True)

data['Embarked']=pd.get_dummies(data['Embarked'],drop_first=True)
data.head()
##Delete the column which are not required

list(data.columns.values)

data.drop(['PassengerId','Cabin','Fare','Ticket','Name'],axis=1,inplace=True)
X=data.loc[:,['Pclass', 'Sex', 'Age', 'Embarked']].values

Y=data.loc[:,['Survived']].values
##Splitting into train and test data



from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.30,random_state=10)
#ModelSelection



from sklearn.ensemble import RandomForestClassifier

classfier=RandomForestClassifier(n_estimators=100,criterion='entropy')

classfier.fit(Xtrain,Ytrain)



##Predicting the result

Ypred=classfier.predict(Xtest)

#Validation

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

CR=classification_report(Ytest,Ypred)

cm=confusion_matrix(Ytest,Ypred)

Accuracy=accuracy_score(Ytest,Ypred)









print(CR)

print(cm)

print(Accuracy)
