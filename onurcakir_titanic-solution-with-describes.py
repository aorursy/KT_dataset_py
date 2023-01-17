# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualization 

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#READ THE DATA

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

clear_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_sub_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
#Convert lowercase to easy coding

train_data.columns = train_data.columns.str.lower()

test_data.columns = test_data.columns.str.lower()

gender_sub_data.columns = gender_sub_data.columns.str.lower()
#Function to convert embarked values

def convert_embarked(x):

    if   x == "S":

        return 0

    elif x == "C":

        return 1

    elif x == "Q":

        return 2
def clean_data(data):

    data.dropna(subset=['embarked','age'],inplace=True)

    data.drop(['cabin','name','passengerid','ticket'],axis=1,inplace=True)

    data.embarked = [convert_embarked(each) for each in data.embarked]

    data.sex = [1 if each == "male" else 0 for each in data.sex]

    return data

    
#DATA CLEANING and PREPEARING

train_data = clean_data(train_data)

y = train_data.iloc[:,0]

x_data = train_data.drop(['survived'],axis=1)
#Normalization

x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))
#test,train split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
def clean_test_data(data):

    data.drop(['cabin','name','ticket'],axis=1,inplace=True)

    data.embarked = [convert_embarked(each) for each in data.embarked]

    data.sex = [1 if each == "male" else 0 for each in data.sex]

    return data
test_data.fillna(test_data.mean(),inplace=True)
#normalization for test data

clean_test_data(test_data)



test_data = (test_data - np.min(test_data)) /(np.max(test_data)-np.min(test_data))

#Choose best algorithm for data

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 15)

knn.fit(x_train,y_train)

test_data['predicted_survived'] = knn.predict(test_data.drop('passengerid',axis=1))

print(" knn score: {} ".format(knn.score(x_test,y_test)))





submission = pd.DataFrame({

        "PassengerId": clear_test_data["PassengerId"],

        "Survived": test_data['predicted_survived']

    })

submission.to_csv('submission.csv',index=False)