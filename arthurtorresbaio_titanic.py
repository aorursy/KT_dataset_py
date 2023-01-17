# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
 ## Importing helpfull Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
##Loading the data
####data.groupby('Initial')['Age'].mean() #lets check the average age by Initials

""" Some code that will Help me to undestand the data from : https://www.kaggle.com/ash316/eda-to-prediction-dietanic
 data.groupby(['Sex','Survived'])['Survived'].count()
 data['Initial']=data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
 pd.crosstab(data.Parch,data.Pclass).style.background_gradient(cmap='summer_r')
 data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
"""

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

datasets = [train, test]

#Check if there is relation between sex and survive

train.groupby(['Sex'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')

print(train.info())
print( '-'* 40)
print(test.info())
sns.distplot(train.Age)
print("The minium age is : ",train.Age.min())
print("The max age is : ", train.Age.max())
print("The Average age is : ", train.Age.mean())
print("The Standard Deviation age is : ",train.Age.std())
for dataset in datasets : # To fill the train and test data 
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
    
print(train.Title.value_counts())  ##See the count of the Titles that we gathered.


for dataset in datasets :

    dataset['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col'
                       ,'Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs',
                                                   'Other','Other','Other','Mr','Mr','Mr'],inplace=True)
train.groupby('Title')['Age'].mean()["Mrs"] #lets check the average age by Ttile and change the Age


for dataset in datasets :    ##If the Age is null look for their Title is change de Age by Title
    dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Mr'),'Age']=train.groupby('Title')['Age'].mean()["Mr"]
    dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Mrs'),'Age']=train.groupby('Title')['Age'].mean()['Mrs']
    dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Master'),'Age']=train.groupby('Title')['Age'].mean()["Master"]
    dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Miss'),'Age']=train.groupby('Title')['Age'].mean()["Miss"]
    dataset.loc[(dataset.Age.isnull())&(dataset.Title=='Other'),'Age']=train.groupby('Title')['Age'].mean()["Other"]

# Check if there is a null value in Age :
train.Age.isna().sum()
    
# Put the categories to use in the Logistic Regression model

X = np.asarray(train[['PassengerId','Pclass', 'Sex','Age', 'SibSp', 'Parch','Embarked']])
y = np.asarray(train['Survived'])



from sklearn.model_selection import train_test_split

# divide my data in Train data and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 2)

from sklearn.linear_model import LogisticRegression

 # creating the model and training it!!!!
logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, y_train)
predictions = logisticRegression.predict(X_test)
 # predict the results of my test sample.

print(confusion_matrix(y_test, predictions))  ## Finding Confusion Matrix
accuracy = (86+51)/(86+14+28+61)#n Matrix to check the accuracy of my model
print("Accuracy is : ", accuracy)  ## Showing Accuracy.
test["Survived"]= logisticRegression.predict(test)

Result=test[['PassengerId', 'Survived']]
Result.to_csv('Result_Titanic.csv', index = False)