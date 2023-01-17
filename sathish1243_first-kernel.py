## I'm beginner and this is my first kernel on Iris dataset.
## Your suggestions means a lot to me and will help to improve this kernel
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read the Iris dataset.
#index_col=0, to use first column in the dataset as index. Id is the first column
irisdataset = pd.read_csv("../input/Iris.csv",index_col=0)
#Find the type
type(irisdataset)
#Fetch First three rows
irisdataset.head(3)
#Fetch last three rows
irisdataset.tail(3)
#Check any data missing
irisdataset.isnull().sum()
#Dataset has 6 columns. ID is just Id which is not useful metric here
#Among 5, Species is dependent variable remaining four are independent variables.
#So the change in independent variables cause change in dependent variable
#Let's check stats here
irisdataset.describe()
#As target or dependent variable having text data, which doesn't understand by the machine. 
#So we will convert that to numeric values
from sklearn.preprocessing import LabelEncoder
LblEncoder = LabelEncoder()
irisdataset["Species"] = LblEncoder.fit_transform(irisdataset['Species'])
irisdataset.head(3)
#Observe Species column
irisdataset.tail(3)
#Split the data into training and test sets
from sklearn.model_selection import train_test_split
#feature selection
features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X_train, X_test, y_train, y_test = train_test_split(irisdataset[features],irisdataset.Species,test_size = 0.25)
#Check shape of X_train
X_train.shape
#Check shape of X_test
X_test.shape
#Implemented KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1) #Started with neighbor = 1 and later we will check what's the best value 
knn.fit(X_train,y_train) # trained the model with train data 
#metric = 'minkowski', which means it uses euclidean distance. When we have two points(x1,y1) and (x2,y2)
#It calculates as Squareroot ((x2-x1)power 2 + (y2-y1)power 2)
#Predicting Species for test data
y_pred = knn.predict(X_test)
#Our model is build. Let's evaluate model using Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
#Check the Number of Neighbors VS error rate
error_rate= []
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    error_rate.append('{'+ str(i) +': '+ str(np.mean(pred != y_test))+ '}')
error_rate
#From the above i think n_neighbors = 5 is the best fit
#Lookforward for your suggestions
my_submission = pd.DataFrame({'Id': X_test.index, 'Species': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
