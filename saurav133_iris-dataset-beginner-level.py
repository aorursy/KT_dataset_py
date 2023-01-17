# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# using pandas to read data

data = pd.read_csv('../input/Iris.csv')
# .head() method by defalut return first 5 records

# one can pass any number in the method to see more/less records

# example: data.head(x) where x can be any number

data.head()
# .info method returns the info related to kind of data in the dataframe

data.info()
# axis=1 to tell pandas to take the column out (axis=1 signifies rows) 

# inplace=True to save some memory of using another variable to store new modified dataframe

data.drop('Id', axis=1, inplace=True)
# .describe() is used to get some statistical insight from the data

# one thing to note here is that this method basically deals with numeric data

data.describe()
import matplotlib.pyplot as plt

import seaborn as sns



# to get the plots inline with the code

%matplotlib inline  
sns.pairplot(data)
# .corr is used to find the correlation of features

# this method basically deals with numeric data only

data.corr()
sns.heatmap(data.corr(), annot=True)
print(data['Species'].unique())
# assigning dummy values 0,1,2 to Iris-setosa, Iris-versicolor, Iris-virginica respectively.



def getDummies(species):

    if species == 'Iris-setosa':

        return 0

    elif species == 'Iris-versicolor':

        return 1

    else:

        return 2
data['Species'] = data['Species'].apply(getDummies)
print(data['Species'].unique())
#for performance metrics

from sklearn.metrics import accuracy_score



#for splitting data

from sklearn.model_selection import train_test_split
X = data.drop('Species', axis=1)

y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#importing LinearSVC

from sklearn.svm import LinearSVC
LSVCModel = LinearSVC()  #creating LinearSVC model

LSVCModel.fit(X_train, y_train)  #fitting training data to the model

predictions = LSVCModel.predict(X_test) #predicting values from model

print("Linear SVC Accuracy:", accuracy_score(y_test, predictions)) #accuracy score
#importing KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
KNCModel = KNeighborsClassifier()  #creating KNeighborsClassifier model

KNCModel.fit(X_train, y_train)  #fitting training data to the model

predictions = KNCModel.predict(X_test) #predicting values from model

print("Linear SVC Accuracy:", accuracy_score(y_test, predictions)) #accuracy score
#importing SVC

from sklearn.svm import SVC
SVCModel = SVC()  #creating SVC model

SVCModel.fit(X_train, y_train)  #fitting training data to the model

predictions = SVCModel.predict(X_test) #predicting values from model

print("Linear SVC Accuracy:", accuracy_score(y_test, predictions)) #accuracy score
#importing RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
RFCModel = RandomForestClassifier(n_estimators=100)  #creating RandomForestClassifier model

RFCModel.fit(X_train, y_train)  #fitting training data to the model

predictions = RFCModel.predict(X_test) #predicting values from model

print("Random Forest Classifier Accuracy:", accuracy_score(y_test, predictions)) #accuracy score