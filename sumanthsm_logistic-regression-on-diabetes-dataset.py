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
#First we have to import all required Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#reading the csv file using pandas 



Data=pd.read_csv('/kaggle/input/diabetes-dataset/diabetes2.csv')

Data.head() #top 5 rows of data

Data.info()
#finding out null values

#Data.isnull().sum().sum()

Data.isnull() # so there is no null values in our dataset
#finding the null values using heatmap 

sns.heatmap(Data.isnull(),cmap='viridis') #so there is null values or there is no shades
#graph using seaborn library 

sns.countplot('Outcome',data=Data) #here In this graph you see the count of Diabetes members 
#now we are creating X(independent features) & y(dependent values)



X=Data.iloc[:,[0,1,2,3,4,5,6,7]].values # from column 0 to 7 

y=Data.iloc[:,-1].values #last column
#splitting dataset into training and test set

#here we are importing traintestsplit function from library sklearn



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=33) #iam giving a testsize value is 30% & random state is 30 value
X_train
#lets create a model using logistic regression algorithm



from sklearn.linear_model import LogisticRegression



clf=LogisticRegression(random_state=0) #creating a object called clf

clf.fit(X_train,y_train) # fit the train datasets
#now predict the X_test

y_pred=clf.predict(X_test)
#finding accuracy of model

from sklearn.metrics import confusion_matrix,classification_report



print(confusion_matrix(y_pred,y_test))
#macro average (averaging the unweighted mean per label)

#weighted average (averaging the support-weighted mean per label)(i.e. the number of correctly predicted instances in that class, divided by the total number of instances in that class)

print(classification_report(y_pred,y_test))