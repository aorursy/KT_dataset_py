#---------Mice problem----------

# In this problem we have a total of 7 different categories
# in class,That defines different types of protein in the cortex of 
# Mice and by given attributes we have to conclude which type protein it 
# is.
# We have used SVC that is support vector classifier that is used
# for classification purpose and as this dataset has many attributes
# SVM is a great choice for this types of problem where we have must 
# attributes that help us to classify.

#importing libraries

import pandas as pd
import numpy as np

#importing and cleaning the dataset

dataset = pd.read_csv('../input/Data_Cortex_Nuclear.csv')
dataset = dataset.dropna()
dataset.head()
#total columns in the dataset

dataset.columns
#using labelencoder to encode the strings
# that is the unique strings in the columns 

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
dataset['class'] = enc.fit_transform(dataset['class'])
dataset.head()
X_data = dataset.iloc[:,1:78]
X_data.columns
# Getting X_data and printing instances of X_data

X_data = X_data.values
X_data[0:2]
#Getting Y_data

Y_data = dataset['class'].values
Y_data[0:5]
#Getting the unique classes in Y_data and the no of instances of each 
# class

unique , counts = np.unique(Y_data,return_counts=True)
print(unique,counts)
#scaling the inputs using StandardScaler

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
X_data[0:2]
# splitting data into training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_data,Y_data,test_size=0.3)
#Getting our classifier

from sklearn.svm import SVC
classifier = SVC(gamma='scale')
classifier.fit(x_train,y_train)
#using cross_val_score for checking accuracy on different folds

from sklearn.model_selection import cross_val_score
results = cross_val_score(classifier,X_data,Y_data,cv=5)
print(results)
# in the best case scenario we got a accuracy score of 91 %.
#finally printing the accuracy on training set

classifier.score(x_train,y_train)
classifier.score(x_test,y_test)