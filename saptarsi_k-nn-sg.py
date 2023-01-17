import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
print(os.listdir(".."))
mydata = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
print(mydata.shape)
print(mydata.head(2))
# Looking at the class
print(mydata.iloc[:,1].describe())
# Looking at first few variables
print(mydata.iloc[:,2:6].describe())
mydata.boxplot(column=[ 'texture_mean','perimeter_mean'], by='diagnosis', grid=False)
# Dropping Columns
mydata=mydata.drop(['id', 'Unnamed: 32'], axis = 1) 
print(mydata.shape)
y = mydata.iloc[:,0]
x = mydata.iloc[:,1:31]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)
x_scaled
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
#Initalize the classifier
knn = KNeighborsClassifier(n_neighbors=15)
#Fitting the training data
knn.fit(x_train, y_train)
#Predicting on test
y_pred=knn.predict(x_test)
#Accuracy
print('Accuracy = ', knn.score(x_test, y_test))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
print('\nConfusion matrix')
print(confusion_matrix(y_test, y_pred))

#Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report')
print(classification_report(y_test, y_pred))  
nc=np.arange(10,1,-2)
nc
nc=np.arange(1,100,2)
acc=np.empty(50)
i=0
for k in np.nditer(nc):
    knn = KNeighborsClassifier(n_neighbors=int(k))
    knn.fit(x_train, y_train)
    temp= knn.score(x_test, y_test)
    print(temp)
    acc[i]=temp
    i = i + 1
acc
x=pd.Series(acc,index=nc)
x.plot()
# Add title and axis names
plt.title('Neighbor vs Accuracy')
plt.xlabel('Count of Neighbor')
plt.ylabel('Accuracy')
plt.show() 
