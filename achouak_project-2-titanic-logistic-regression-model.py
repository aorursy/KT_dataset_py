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
#import libraries for data visualisation 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#read train cvs file with pandas 

file_path ='../input/titanic-machine-learning-from-disaster/train.csv'

data =pd.read_csv( file_path)

#show the first 5 rows of data dataframe 

data.head()
# identify missing data 

missing_data=data.isnull()

missing_data
# visualize missing data with heatmap of boolean values 

sns.heatmap(missing_data, cbar=False)
sns.countplot(data["Survived"], hue="Sex", data=data, palette='RdBu_r')
sns.countplot(data["Survived"], hue="Pclass", data=data, palette='RdBu_r')
sns.countplot(data["Pclass"], data=data, palette='RdBu_r')
data.groupby("Pclass")["Age"].mean()
#define a function that return missing value per class 

def fill_missing(cols) : 

    Age = cols[0] #Age column

    Pclass = cols[1] #Pclass column

    if pd.isnull(Age): #if the age value is missing 

        if Pclass==1: 

            return 38

        elif Pclass==2: 

            return 29

        else :

            return 25

    else : 

        return Age 

    

data["Age"]= data[["Age","Pclass"]].apply(fill_missing, axis=1)
#Checking visually that our function filled the missing value 

sns.heatmap(data.isnull(), cbar=False)
data.drop("Cabin", axis=1, inplace=True) 

#if you don't use inplace =True the cabin column will still exist on your data 
sns.heatmap(data.isnull(), cbar=False)
pd.get_dummies(data["Sex"])
#create dummiee variables for sex and Embarked

sex=pd.get_dummies(data["Sex"],drop_first=True)

embarked= pd.get_dummies(data["Embarked"],drop_first=True)

#Add this two variable to our data : 

data=pd.concat([data,sex,embarked], axis=1)

#Check the first row of our data

data.head(1)
data.drop(["Sex","Embarked","Name","Ticket"], axis=1, inplace=True)
#drop the passanger ID

data.drop("PassengerId", axis=1, inplace=True)
#check data

data.head()
#select the output y and the features X

y=data["Survived"]

features = ["Pclass","Age", "SibSp", "Parch","Fare","male","Q","S"]

X=data[features]
#Import scikit learn libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



#split our data into a train and test data

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.30, random_state=0)



#Create an instance of a logistic model 

lgmodel= LogisticRegression()



#train the lgmodel 

lgmodel.fit(X_train,y_train)
predictions = lgmodel.predict(X_test)
from sklearn.metrics import confusion_matrix

#show the confusing matrix 

print(confusion_matrix(y_test,predictions))
from sklearn.metrics import classification_report 

#show a full classification report

print(classification_report(y_test,predictions))