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
#Import scikit-learn dataset library

from sklearn import datasets



#Load dataset

cancer = datasets.load_breast_cancer()
# print the names of the 13 features

print("Features: ", cancer.feature_names)



# print the label type of cancer('malignant' 'benign')

print("Labels: ", cancer.target_names)
# print data(feature)shape

cancer.data.shape
# print the cancer data features (top 5 records)

print(cancer.data[0:5])
# print the cancer labels (0:malignant, 1:benign)

print(cancer.target)
# Import train_test_split function

from sklearn.model_selection import train_test_split



# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test
#Import svm model

from sklearn import svm



#Create a svm Classifier

clf = svm.SVC(kernel='linear') # Linear Kernel



#Train the model using the training sets

clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("recall:",metrics.recall_score(y_test, y_pred))

print("precision:",metrics.precision_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix ,accuracy_score ,classification_report

print(classification_report(y_test,y_pred))