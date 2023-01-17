# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import the necesary libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import roc_auc_score
#Loading The data

df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#Getting a glimpse of data

df.head()
#The details of data

df.shape
#Splitting the data

X = df.iloc[:,:-1]

y = df.iloc[:,-1]
#Splitting the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
#Fitting the Logistic Regression Model

model = LogisticRegression(random_state = 0)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test,y_test)

print('The accuracy of the model is:', accuracy)

print('Recall:', recall_score(y_test,y_pred))

print('Precision:', precision_score(y_test,y_pred))

print('F1 score:', f1_score(y_test, y_pred))

print('Confusion matrix: \n', confusion_matrix(y_test,y_pred))
#Code starts here

from imblearn.under_sampling import RandomUnderSampler
# Create random under sampler object

rus = RandomUnderSampler(random_state = 0)



#Undersampling the train data

X_sample2,y_sample2 = rus.fit_sample(X_train,y_train)

sns.countplot(y_sample2)
#Initiating a logistic regression model

model_rus = LogisticRegression(random_state=0)



#Fitting the model with sampled data

model_rus.fit(X_sample2, y_sample2)



#Making prediction of test values

y_pred=model_rus.predict(X_test)



#Finding the accuracy score

accuracy_rus=model_rus.score(X_test,y_test)

print("Accuracy:",accuracy_rus)       



#Finding the recall score

recall_rus=recall_score(y_test, y_pred)

print ("Recall:",recall_rus)



#Finding the precision score

precision_rus=precision_score(y_test, y_pred)

print ("Precision:",precision_rus)



#Finding the f1 score

f1_rus=f1_score(y_test, y_pred)

print ("f1_score:", f1_rus)



#Finding the confusion matrix

confusion_mat_rus=confusion_matrix(y_test, y_pred)

print ("Confusion Matrix:\n",confusion_mat_rus)

from imblearn.under_sampling import ClusterCentroids





# Code starts here

cc = ClusterCentroids(random_state = 0)



X_sample3, y_sample3 = cc.fit_sample(X_train, y_train)

sns.countplot(y_sample3)
#Initiate Logistic Regression Model for Cluster Centroids

model_cc = LogisticRegression(random_state = 0)



#Fitting the model with sampled data

model_cc.fit(X_sample3, y_sample3)



#Making prediction of test values

y_pred = model_cc.predict(X_test)



#Finding the Accuracy score

accuracy_cc = model_cc.score(X_test,y_test)

print("Accuracy:", accuracy_cc)



#Finding the Recall score

recall_cc = recall_score(y_test,y_pred)

print("Recall:", recall_cc )



#Finding the Precision score

precision_cc = precision_score(y_test,y_pred)

print("Precision score:", precision_cc)



#Finding the f1 score

f1_cc = f1_score(y_test,y_pred)

print("F1 score:", f1_cc)



#Finding the confusion matrix

confusion_mat_cc = confusion_matrix(y_test, y_pred)

print('Confusion Matrix:\n',confusion_mat_cc)
from imblearn.under_sampling import TomekLinks



#Code starts here

tl = TomekLinks()



X_sample4,y_sample4 = tl.fit_sample(X_train, y_train)

sns.countplot(y_sample4)
#Initiate Logistic Regression Model for Tomek Links



model_tl = LogisticRegression(random_state = 0)



#Fitting the model with sampled data

model_cc.fit(X_sample4, y_sample4)



#Making prediction of test values

y_pred = model_cc.predict(X_test)



#Finding the Accuracy score

accuracy_cc = model_cc.score(X_test,y_test)

print("Accuracy:", accuracy_cc)



#Finding the Recall score

recall_cc = recall_score(y_test,y_pred)

print("Recall:", recall_cc )



#Finding the Precision score

precision_cc = precision_score(y_test,y_pred)

print("Precision score:", precision_cc)



#Finding the f1 score

f1_cc = f1_score(y_test,y_pred)

print("F1 score:", f1_cc)



#Finding the confusion matrix

confusion_mat_tl = confusion_matrix(y_test, y_pred)

print('Confusion Matrix:\n',confusion_mat_tl)

from imblearn.over_sampling import RandomOverSampler



# Code starts here

ros = RandomOverSampler(random_state = 0)



X_sample5, y_sample5 = ros.fit_sample(X_train, y_train)



#Fitting the model with sampled data

model_ros = LogisticRegression(random_state = 0)

model_ros.fit(X_sample5, y_sample5)



#Making prediction of test values

y_pred = model_ros.predict(X_test)



#Finding the Accuracy score

accuracy_ros = model_ros.score(X_test, y_test)

print("Accuracy score:", accuracy_ros)



#Finding the Recall scor

recall_ros = recall_score(y_test, y_pred)

print("Recall:", recall_ros)



#Finding the Precision score

precision_ros = precision_score(y_test, y_pred)

print("Precision:", precision_ros)



#Finding the f1 score

f1_ros = f1_score(y_test, y_pred)

print("F1 Score", f1_ros)



#Finding the confusion matrix

confusion_mat_ros = confusion_matrix(y_test,y_pred)

print('The Confusion Matrix: \n', confusion_mat_ros)
from imblearn.over_sampling import SMOTE



# Code starts here

smote = SMOTE(random_state = 0)

X_sample6, y_sample6 = smote.fit_sample(X_train, y_train)



model_smote = LogisticRegression(random_state = 0)



model_smote.fit(X_sample6,y_sample6)



y_pred = model_smote.predict(X_test)



accuracy_smote = model_smote.score(X_test, y_test)

print("Accuarcy:", accuracy_smote)



recall_smote = recall_score(y_test, y_pred)

print("Recall:", recall_smote)



precision_smote = precision_score(y_test, y_pred)

print("Precision:", precision_smote)



f1_smote = f1_score(y_test, y_pred)

print("F1 score:", f1_smote)



confusion_mat_smote = confusion_matrix(y_test, y_pred)

print(confusion_mat_smote)



# Code ends here