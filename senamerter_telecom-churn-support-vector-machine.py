import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



data = pd.read_csv(r"../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#first few rows

data.head()
# Uploading the packages we need:

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

data.head()

data.describe()



# Take a look at Tenure dimension using distibution, dimensions can be changed.

sns.distplot(data['tenure'])
# Correlation check

data.corr()

sns.heatmap(data.corr(),cmap='BuGn')

# Excluding the ID column, it won't be useful.

data = data.drop(['customerID'], axis= 1)

data.head()
# Converting categorical data to numeric 

char_cols = data.dtypes.pipe(lambda x: x[x == 'object']).index

for c in char_cols:

    data[c] = pd.factorize(data[c])[0]

data.head()
# Define the target variable (dependent variable) 

y = data.Churn 

data = data.drop(['Churn'], axis= 1)
# Splitting training and testing data

from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.20)  

# Applying Support Vector Machine algorithm

from sklearn.svm import SVC  

svclassifier = SVC(kernel='linear', degree=8)  

svclassifier.fit(X_train, y_train)  
# Predicting part, applying the model to predict

y_pred = svclassifier.predict(X_test)  
# Evaluating model performance

from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test,y_pred))  

print(classification_report(y_test,y_pred))  