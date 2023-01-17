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
#Import packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/bank-customer-retirement/Bank_Customer_retirement.csv')

data.head()
data.tail()
data.info()
#Retirement

print("The total number of customers in this dataset is: ", len(data))

print("Out of those customers, the number of retired customers is: ", len(data[data['Retire']==1]))

print("Out of those customers, the number of not retired customers is: ", len(data[data['Retire']==0]))
#Distribution of age

sns.distplot(data[data['Retire']==1]['Age'], label = 'Retire == 1')

sns.distplot(data[data['Retire']==0]['Age'], label = 'Retire == 0')

plt.legend()

plt.show()
#Distribution of 401k savings

sns.distplot(data[data['Retire']==1]['401K Savings'], label = 'Retire == 1')

sns.distplot(data[data['Retire']==0]['401K Savings'], label = 'Retire == 0')

plt.legend()

plt.show()
#Scatterplot

sns.scatterplot(data = data, x = '401K Savings', y = 'Age', hue = 'Retire')
#Drop customer ID

data = data.drop(['Customer ID'], axis = 1)

data.head()
#Split into X and y

X = data.drop(['Retire'], axis = 1)

y = data['Retire']
#Normalize X

X_min = X.min()

X_range = (X - X_min).max()

X_scaled = (X - X_min)/X_range
#Inspect normalization

sns.scatterplot(data = X, x = 'Age', y = '401K Savings')
sns.scatterplot(data = X_scaled, x = 'Age', y = '401K Savings')
#Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.25, random_state = 42)
#Fit model

from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(X_train, y_train)
#Evaluate model

y_pred = svc_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = 'd')

plt.xlabel('Actual Value')

plt.ylabel('Predicted Value')
from sklearn.model_selection import GridSearchCV
#specify parameters and instantiate grid search cv

params = {'C': [0.1, 1, 10, 100],

         'gamma': [0.1, 1, 10, 100],

         'kernel': ['linear', 'rbf']}

gs = GridSearchCV(SVC(), params)
#fit gs

gs.fit(X_train, y_train)
gs.best_params_
#move the range of C leftwards to check validity

params_1 = {'C': [0.001, 0.01, 0.1, 1],

         'gamma': [0.1, 1, 10, 100],

         'kernel': ['linear', 'rbf']}

gs_1 = GridSearchCV(SVC(), params_1)
gs_1.fit(X_train, y_train)

gs_1.best_params_
#predict based on gs

y_pred_gs1= gs_1.predict(X_test)
#evaluate gs results

print(classification_report(y_test, y_pred_gs1))
sns.heatmap(confusion_matrix(y_test, y_pred_gs1), annot = True)

plt.xlabel('Actual Value')

plt.ylabel('Predicted Value')