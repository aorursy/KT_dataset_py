# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# Import data

data = pd.read_csv('../input/advertising.csv')
data.head()
data.shape
# Checking the info on our data

data.info()
data.describe()
from scipy.stats import norm

sns.distplot(data['Age'],hist=False, color='r', rug=True, fit=norm)
f, ax = plt.subplots(figsize=(10, 10))  

sns.kdeplot(data.Age, data['Daily Time Spent on Site'], color="b", ax=ax) 

sns.rugplot(data.Age, color="r", ax=ax)  

sns.rugplot(data['Daily Time Spent on Site'], vertical=True, ax=ax)  
f, ax = plt.subplots(figsize=(8, 8))  

cmap = sns.cubehelix_palette(as_cmap=True, start=0, dark=0, light=3, reverse=True)  

sns.kdeplot(data["Daily Time Spent on Site"], data['Daily Internet Usage'],  

    cmap=cmap, n_levels=100, shade=True);
from pandas.plotting import scatter_matrix  

scatter_matrix(data[['Daily Time Spent on Site', 'Age','Area Income', 'Daily Internet Usage']],  

 alpha=0.3, figsize=(10,10));

object_variables = ['Ad Topic Line', 'City', 'Country']  

data[object_variables].describe(include=['O'])  
pd.crosstab(index=data['Country'], columns='count').sort_values(['count'], ascending=False).head(20) 
data = data.drop(['Ad Topic Line', 'City', 'Country'], axis=1)  
data['Timestamp'] = pd.to_datetime(data['Timestamp'])



data['Month'] = data['Timestamp'].dt.month

data['Day of month'] = data['Timestamp'].dt.day

data['Day of week'] = data['Timestamp'].dt.dayofweek

data['Hour'] = data['Timestamp'].dt.hour  

data = data.drop(['Timestamp'], axis=1)



data.head()  
data.columns
from sklearn.model_selection import train_test_split



X = data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage',  

    'Male', 'Month', 'Day of month' ,'Day of week']]

y = data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
model1 = LogisticRegression(solver='lbfgs')

model1.fit(X_train, y_train)

predictions_LR = model1.predict(X_test)



print('\nLogistic regression accuracy:', accuracy_score(predictions_LR, y_test))

print('\nConfusion Matrix:')

print(confusion_matrix(predictions_LR, y_test))

from sklearn.tree import DecisionTreeClassifier



model2 = DecisionTreeClassifier()

model2.fit(X_train, y_train)

predictions_DT = model2.predict(X_test)



print('\nLogistic regression accuracy:', accuracy_score(predictions_LR, y_test))

print('\nConfusion Matrix:')

print(confusion_matrix(predictions_LR, y_test))