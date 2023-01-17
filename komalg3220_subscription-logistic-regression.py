

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn import linear_model

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing  # to normalisation

from sklearn.model_selection import train_test_split as dsplit



print(os.listdir("../input"))



df = pd.read_csv('../input/bank-full.csv' , sep = ";")  # only in case to split the data.



df.drop(['poutcome','month'],axis=1,inplace=True) #to drop column





df.head(6)
#null value

df.isnull().any()
#To check the data type of coloumn



df.dtypes
#To check wheater column is contnious or categorical



for column in df.columns:

    print(column,len(df[column].unique()))
#To define x and y

x = df.loc[:, df.columns != 'y'] #to  select multiple column except one data point may be that we want to predict

y = df["y"]





#convert categorical values (either text or integer) 

x=pd.get_dummies(x,columns=['job','marital','education','default','housing','loan','contact','campaign','pdays','previous'])



#To Normalise the equation

x=preprocessing.normalize(x)



print(x)

print(y)

       
from sklearn.metrics import confusion_matrix 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 
#train and test dataset creation

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
#logestic regression

x_train, x_test, y_train, y_test = dsplit(x, y, random_state = 1)

reg = LogisticRegression()

reg.fit(x_train, y_train)

predicted = reg.predict(x_test)

from sklearn.metrics import accuracy_score

print(reg.score(x_train, y_train))

print(reg.score(x_test, y_test))