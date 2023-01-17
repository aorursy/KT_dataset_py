#importing the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#Importing the dataset

data = pd.read_csv("../input/social-network-ads/Social_Network_Ads.csv")

data.head(5)
X=data.iloc[:,2:4].values #Age and EstimatedSalary are our primary requirements (storing them as a NUMPY ARRAY)

X.shape
y=data.iloc[:,-1]  #IF x is the input(Age and EstimatedSalary), then y is the output that is the customer purcases the product or not.

y.shape
#importing the train_test_split module to train a module with 80% data and testing with the rest of the data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
X_train.shape #Training the data
X_test.shape #Testing the data
#The AGE and ESTIMATED SALARY data is far off, therefore we need to scale the data . Hence STANDARD SCALING is used .
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#Fitting the values to scale them down so that the square of the distances between x1,y1 and x2,y2 etc could be reduced

X_train= scaler.fit_transform(X_train)

X_train
X_test=scaler.transform(X_test)

X_test
np.sqrt(X_train.shape[0])

#This is the value of K

k=17 #approx

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=k)
#Training our model

knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

y_pred.shape

#ensure than y_pred.shape is equal to y_test data
#Calculating accuracy score

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

#hence the model has a 92% accuracy
#Lets try an approach to try and increase the accuracy

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
accuracy=[]

for i in range(4,22):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    accuracy.append(accuracy_score(y_test,knn.predict(X_test)))
len(accuracy)
plt.plot(range(1,19),accuracy)
#Function to provide input data to the classifier.

def pred_output():

    age=int(input("Enter your age: "))

    salary=int(input("Enter your salary: "))

    

    X_new= np.array([[age],[salary]]).reshape(1,2)   #input for our knn classifier. (2D required)

    X_new=scaler.transform(X_new)

    

    if knn.predict(X_new)[0]==0:

        return "Will not Purchase"

    else:

        return "Will Purchase"

    
pred_output()

