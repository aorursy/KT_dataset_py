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
train = pd.read_csv("/kaggle/input/titanic/train.csv")



test=pd.read_csv("/kaggle/input/titanic/test.csv")
import math   #to calculate basic mathematical function

import matplotlib.pyplot as plt

import seaborn as sns    #for statistical plotting 
sns.countplot(x = "Survived", data = train) #to display who survived or who not survived

sns.countplot(x = "Survived", hue = "Sex", data = train) #to display how many m/f survived

sns.countplot(x = "Survived", hue = "Pclass", data = train) #to display the record of how many passenger travelling in which class

train["Age"].plot.hist() #to display the ages of the pessengr working on the titanic

train["Fare"].plot.hist(bins = 20, figsize = (10,5)) #to display the fare of the passenger

sns.countplot(x = "SibSp", data = train)  #to display the record of the sibbling of the passenger

train.isnull()   #to display the null values (true = null, false = not null)



train.isnull().sum()   #to display the aggregate clomns which are having the null values

train.drop("Cabin", axis = 1, inplace = True) #to drop the "cabin" colomns because no use of it

train.dropna(inplace = True)   #to drop the na values

sex = pd.get_dummies(train["Sex"], drop_first=True)  #to convert the string variable(sex) to the categorical variable and remove the female because 1 = male, 0 = female

embark = pd.get_dummies(train["Embarked"],drop_first = True) #same which done with the sex

pcl = pd.get_dummies(train["Pclass"],drop_first = True) #same which done with the sex

    
train = pd.concat([train,sex,embark,pcl],axis = 1)  #to concat all the altered variable to the origina;l

train.drop(["Sex","Embarked","PassengerId","Name","Ticket", "Pclass"], axis = 1, inplace = True)  #dropping the useless colomns

x = train.drop("Survived", axis = 1)

y = train["Survived"]

from sklearn.model_selection import train_test_split

    
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)

    
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)

prediction = logmodel.predict(x_test)

from sklearn.metrics import classification_report

classification_report(y_test,prediction) #to calculate the accuracy and the peformance

from sklearn.metrics import accuracy_score

accuracy_score(y_test,prediction) #to display the accuracy in percentage
