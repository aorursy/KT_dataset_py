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
#loading the train data
train_data = pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv')
# showing the first 5 rows of the train dataset
train_data.head()
#loading the test data
test_data = pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv')
test_data.head()
#shape of the dataset
train_data.shape, test_data.shape
#summary of the train data
train_data.describe()
#summary of the test data
test_data.describe()
#checking the null values in train data
train_data.isnull().sum()
#checking the null values in the test data
test_data.isnull().sum()
#choosing the independent feature
x = train_data.drop('flag',axis = 1)
x.head()
#choosing the dependent variable
y = train_data['flag']
y.head()
#checking the counting of flag column

train_data['flag'].value_counts()
#splitting the train & test data for model selection
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0) 
X_train.shape , X_test.shape, y_train.shape, y_test.shape
# choosing the decision tree classification model
from sklearn.tree import DecisionTreeClassifier

# initializing the model
dtclf = DecisionTreeClassifier()

#fitting the model
dtclf.fit(X_train,y_train)
#predicting the y pred
y_pred = dtclf.predict(X_test)
y_pred
#computing the accuracy, f1 score & confusion matrix
from sklearn import metrics
print('Confusion matrix is :' , metrics.confusion_matrix(y_test,y_pred))

print('Accuracy of the model is :' , metrics.accuracy_score(y_test,y_pred))

print('F1 score is :' , metrics.f1_score(y_test,y_pred))
#again initializing the model with informaion gain
dtclf1 = DecisionTreeClassifier(criterion = 'entropy')

#fitting the model
dtclf1.fit(X_train,y_train)
#predicting the y pred for entropy
y_pred1 = dtclf1.predict(X_test)
y_pred1
#computing the accuracy, f1 score & confusion matrix
from sklearn import metrics
print('Confusion matrix is :' , metrics.confusion_matrix(y_test,y_pred1))

print('Accuracy of the model is :' , metrics.accuracy_score(y_test,y_pred1))

print('F1 score is :' , metrics.f1_score(y_test,y_pred1))
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rfclf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
rfclf= rfclf.fit(X_train,y_train)

y_pred_rf = clf.predict(X_test)
#computing the accuracy, f1 score & confusion matrix
from sklearn import metrics
print('Confusion matrix is :' , metrics.confusion_matrix(y_test,y_pred_rf))

print('Accuracy of the model is :' , metrics.accuracy_score(y_test,y_pred_rf))

print('F1 score is :' , metrics.f1_score(y_test,y_pred_rf))
#Create a Gaussian Classifier
rfclf1 = RandomForestClassifier(n_estimators=100,criterion = 'entropy')

#Train the model using the training sets y_pred=clf.predict(X_test)
rfclf1 = rfclf1.fit(X_train,y_train)

y_pred_rf1 = clf.predict(X_test)
#computing the accuracy, f1 score & confusion matrix
from sklearn import metrics
print('Confusion matrix is :' , metrics.confusion_matrix(y_test,y_pred_rf1))

print('Accuracy of the model is :' , metrics.accuracy_score(y_test,y_pred_rf1))

print('F1 score is :' , metrics.f1_score(y_test,y_pred_rf1))
#choosing the test data to pred the y pred test data

test_data.head()
# predicting the test data
y_pred_testdata = rfclf.predict(test_data)
y_pred_testdata
# choosing the sample submission csv to update the predicted flag value
sample = pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')
sample.head()
#updating the new predicted flag values
sample['flag'] = y_pred_testdata
sample.head()
#saving the new csv file
sample.to_csv('sample submission random forest.csv',index = False)