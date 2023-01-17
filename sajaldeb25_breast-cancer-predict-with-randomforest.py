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
# here we will import the libraries used for machine learning

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. I like it most for plot

%matplotlib inline

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.model_selection import train_test_split # to split the data into two parts

#from sklearn.cross_validation import KFold # use for cross validation

from sklearn.model_selection import GridSearchCV# for tuning parameter

from sklearn.ensemble import RandomForestClassifier # for random forest classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm # for Support Vector Machine

from sklearn import metrics # for the check the error and accuracy of the model

# Any results you write to the current directory are saved as output.

# dont worry about the error if its not working then insteda of model_selection we can use cross_validation
data = pd.read_csv("../input/data.csv",header=0)
data.head()
data.info()
# now i am going to drop the column Unnamed: 32

data.drop("Unnamed: 32",axis=1,inplace=True) # in this process this will change in our data itself 

# if you want to save your old data then you can use below code

# data1=data.drop("Unnamed:32",axis=1)  --> here axis 1 means we are droping the column
data.head()
data.columns
data.drop("id",axis=1,inplace=True)
# The data can be divided into three parts.lets divied the features according to their category

features_mean= list(data.columns[1:11])

features_se= list(data.columns[11:20])

features_worst=list(data.columns[21:31])

print(features_mean)

print("-----------------------------------")

print(features_se)

print("------------------------------------")

print(features_worst)
# lets now start with features_mean 

# now as ou know our diagnosis column is a object type so we can map it to integer value

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data.head()
# lets get the frequency of cancer stages

sns.countplot(data['diagnosis'],label="Count")
# now lets draw a correlation graph so that we can remove multi colinearity it means the columns are

# dependenig on each other so we should avoid it because what is the use of using same column twice

# lets check the correlation between features

# now we will do this analysis only for features_mean then we will do for others and will see who is doing best

corr = data[features_mean].corr() # .corr is used for find corelation

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15}, xticklabels= features_mean, yticklabels= features_mean,cmap= 'coolwarm')

# for more on heatmap you can visit Link(http://seaborn.pydata.org/generated/seaborn.heatmap.html)
prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

# now these are the variables or collumns which will use for prediction
#now split our data into train and test

train, test = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test

# we can check their dimension

print(train.shape)

print(test.shape)
train_X = train[prediction_var]# taking the training data input 

train_y=train.diagnosis# This is output of our training data

# same we have to do for test

test_X= test[prediction_var] # taking test data inputs

test_y =test.diagnosis   #output value of test dat
model=RandomForestClassifier(n_estimators=100)# a simple random forest model
CmR = model.fit(train_X,train_y)# now fit our model for traiing data
prediction=model.predict(test_X)# predict for the test data

# prediction will contain the predicted value by our model predicted values of dignosis column for test inputs
metrics.accuracy_score(prediction,test_y) # to check the accuracy

# here we will use accuracy measurement between our predicted value and our test output values
from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score

ac = accuracy_score(test_y,CmR.predict(test_X))

print('Accuracy is: ',ac)

cm = confusion_matrix(test_y,CmR.predict(test_X))

sns.heatmap(cm,annot=True,fmt="d")
prediction_var = features_mean # taking all features

train_X= train[prediction_var]

train_y= train.diagnosis

test_X = test[prediction_var]

test_y = test.diagnosis







model=RandomForestClassifier(n_estimators=100)



CmR2 = model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
ac = accuracy_score(test_y,CmR2.predict(test_X))

print('Accuracy is: ',ac)

cm = confusion_matrix(test_y,CmR2.predict(test_X))

sns.heatmap(cm,annot=True,fmt="d")




prediction_var = features_worst  # taking #rd part of column



train_X= train[prediction_var]

train_y= train.diagnosis

test_X = test[prediction_var]

test_y = test.diagnosis



model=RandomForestClassifier(n_estimators=100)

CmR3 = model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
ac = accuracy_score(test_y,CmR3.predict(test_X))

print('Accuracy is: ',ac)

cm = confusion_matrix(test_y,CmR3.predict(test_X))

sns.heatmap(cm,annot=True,fmt="d")