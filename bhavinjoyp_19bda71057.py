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
#importing all necessary python libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
#importing the data for model fitting using read_csv

data = pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")

#importing data for prediction

test_data = pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")

data.head()
#  creating Correlation matrix

corr = data.corr()

corr.style.background_gradient(cmap='coolwarm')
# using the correlation matrix we are only selecting the important features which are 'currentBack','currentFront','trackingDeviationBack','positionBack','motorTempFront'.

labels=data['flag'] # preparing target column

# preparing feature columns

features=data.drop('flag',axis=1)



# splitting the data to test and train

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33, random_state = 42)
plot=pd.plotting.scatter_matrix(data,c=labels, figsize=[40,40])
rf = RandomForestClassifier(n_estimators = 70, criterion='entropy', random_state=42)

rf.fit(train_features, train_labels) # fitting a random forest classifier to the train data

y_pred = rf.predict(test_features) # predicting the test target using the random forest model

print("Accuracy=",accuracy_score(test_labels,y_pred))

print("Confusion matrix",confusion_matrix(test_labels,y_pred))

print(classification_report(test_labels,y_pred))
lr = LogisticRegression()

lr.fit(train_features, train_labels) # fitting a logistic regression model to the train data

y_pred_lr = lr.predict(test_features) # predicting the test target using the logistic regression model

print("Accuracy=",accuracy_score(test_labels,y_pred_lr))

print("Confusion matrix",confusion_matrix(test_labels,y_pred_lr))

print(classification_report(test_labels,y_pred_lr))
dt = DecisionTreeClassifier(class_weight='balanced')

dt.fit(train_features, train_labels) # fitting a decision tree model to the train data

y_pred_dt = dt.predict(test_features) # predicting the test target using the decision tree model

print("Accuracy=",accuracy_score(test_labels,y_pred_dt))

print("Confusion matrix",confusion_matrix(test_labels,y_pred_dt))

print(classification_report(test_labels,y_pred_dt))
sub=pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")

sample_features=test_data

sample_pred=rf.predict(sample_features)

sub['flag']=sample_pred

sub.to_csv("rf_6.csv",index=False)