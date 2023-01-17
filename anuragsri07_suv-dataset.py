import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# reading dataset

suv_data = pd.read_csv("../input/suv-data/suv_data.csv")
sns.countplot('Gender', data=suv_data)
#Plotting data about how many Male and Female are able to purchased the SUV

sns.countplot('Purchased', hue='Gender', data=suv_data)
# Plotting Age to check how old are the people in dataset

suv_data['Age'].plot.hist()
#checking head of the dataset

suv_data.head()
#Dropping User ID as we don't need for futher process

suv_data.drop('User ID', inplace=True, axis=1)
#checking head of the dataset after dropping of User ID

suv_data.head()
# using get_dummies to convert Gender's String value in binary

gender = pd.get_dummies(suv_data['Gender'], drop_first=True)

gender.head()
suv_data = pd.concat([suv_data,gender], axis=1)

suv_data.head()
#dropping Gender as we already converted it's data

suv_data.drop('Gender',axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
X = suv_data.drop('Purchased',axis=1)

y = suv_data['Purchased']
# Chossing data for training and Testing #test_size is use to choose for the testing and random_state is used to ensure the same result.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
# Fitting data to the Logistic Regression Model

log_model = LogisticRegression(solver='lbfgs')

log_model.fit(X_train, y_train)

# Predicting data to test the our trained data

predictions = log_model.predict(X_test)
#Checking Classification report

print(classification_report(y_test, predictions))
#checking confusion matrix

confusion_matrix(y_test, predictions)
# Checking prediction accuracy score in percentage

accuracy_score(y_test, predictions)*100