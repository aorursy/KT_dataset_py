from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm,metrics
df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv', sep=',')

dataset = df.values

positives = dataset[dataset[:,8]==1,:]

negatives = dataset[dataset[:,8]==0,:]



# Import train_test_split function

y = dataset[:,-1]

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(dataset[:,:],y, test_size=0.3,random_state=42)



# Standardize features

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
for i in range(8):

    n, bins, patches = plt.hist(negatives[:,i], 20, facecolor='red', alpha=0.5)

    n, bins, patches = plt.hist(positives[:,i], 20, facecolor='green', alpha=0.5)

    plt.xlabel(df.columns[i])

    plt.show()
#Create a svm Classifier

clf = svm.SVC(kernel='linear')

#Train the model using the training sets

clf.fit(X_train, y_train)

#Predict the response for test dataset

y_pred = clf.predict(scaler.transform(X_test))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print("F-Score",metrics.f1_score(y_test, y_pred))