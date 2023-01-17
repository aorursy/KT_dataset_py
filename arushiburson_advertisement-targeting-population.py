#importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#reading dataframe

ad_data = pd.read_csv('../input/advertising.csv')

ad_data.head()
#performing exploratory analysis

sns.set_style('darkgrid')

sns.distplot(ad_data['Age'], kde=False, bins=40)
sns.jointplot(x='Age', y='Area Income', data=ad_data)
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde', color='maroon')
sns.jointplot(x='Age', y='Daily Internet Usage', data=ad_data, color='darkgreen')
sns.pairplot(ad_data, hue='Clicked on Ad')
#splitting data into train and test

from sklearn.model_selection import train_test_split

y = ad_data['Clicked on Ad']

X = ad_data.iloc[:,[0,1,2,3,6]]

#X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#performing logistic regression

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='liblinear')

logmodel.fit(X_train, y_train)
#predicting values for test data

predictions = logmodel.predict(X_test)
#creating classification report for the model to determine precision

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))