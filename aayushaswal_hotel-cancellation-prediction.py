#importing the basic library



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import pandas_profiling
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# reading the CSV file to python



hotel = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
hotel
# checking the summary statistics of data

hotel.describe()
#check the top 5 row of dataset

hotel.head()
#checking the bottom 5 row of dataset

hotel.tail()
#checking the data information

hotel.info()
#calculating the missing value in each column

hotel.isnull().sum()
#commpany column in dataset has maximum no of null values

# so we remove the column  

hotel= hotel.drop(['company'],axis=1)
#removing all the row having missing value

hotel= hotel.dropna(axis=0)
hotel.info()
# again checking the missing value 

hotel.isnull().sum()
#checking the unique value of hotel

hotel['hotel'].unique()
#checking the data type for all feature in dataset  

hotel.info()
#converting the required object type feature to categorical

categorical_features = ['hotel','is_canceled','arrival_date_week_number','meal','country','market_segment',

                        'distribution_channel','is_repeated_guest','reserved_room_type','assigned_room_type',

                        'deposit_type','agent','customer_type','reservation_status','arrival_date_month']
hotel[categorical_features]=hotel[categorical_features].astype('category')
# checking the converted data type

hotel.info()


hotel['meal'].unique()
# seperating the dataset into features and target variables

y=hotel['is_canceled']
y
X = hotel.drop(['is_canceled','reservation_status_date'],axis=1)
X
#converting the categorical data into dummy variable  

X_dum=pd.get_dummies(X,prefix_sep='-',drop_first=True)
X_dum
#Splitting the data into train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X_dum,y, test_size=.25,random_state=40)
X_train
# preparing a logistic regression model

from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression()
logistic.fit(X_train,y_train)
#predicting the test data

y_pred= logistic.predict(X_test)
# calculating the  accuracy, precision,recall and f1-score for logistic regression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
accuracy_score(y_test,y_pred)
classification_report(y_test,y_pred)
#calculating the ROC and AUC  for the logistics regression

from sklearn.metrics import roc_curve,roc_auc_score
roc_curve(y_test,y_pred)
roc_auc_score(y_test,y_pred)
#now  we will make a model of random forest and  gradient boosting 

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
rand=RandomForestClassifier(n_jobs=10, random_state=40)

gb=GradientBoostingClassifier(random_state=50)
rand.fit(X_train,y_train)

gb.fit(X_train,y_train)
# predicting the test sample for randomforest and gradient boosting 

rand_pred=rand.predict(X_test)
gb_pred=gb.predict(X_test)
# checking accuracy, precision,recall and f1-score for data

accuracy_score(y_test,rand_pred)
accuracy_score(y_test,gb_pred)
classification_report(y_test,rand_pred)
classification_report(y_test,gb_pred)
roc_auc_score(y_test,rand_pred)
roc_auc_score(y_test,gb_pred)
#creating  confusion matrix for logistic reression,random forest and gradient boosting

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
confusion_matrix(y_test,rand_pred)
confusion_matrix(y_test,gb_pred)