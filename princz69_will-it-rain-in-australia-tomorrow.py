# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the file

df = pd.read_csv('../input/weatherAUS.csv', index_col='Date', parse_dates=True)

# Check dataset content

print(df.head(5))

# Check total null values for all columns

print(df.isnull().sum())

# Handling null or missing values and unneccesary column(s)

df.dropna(axis=0, subset=['Rainfall'], inplace=True)

df.drop(['Location','Evaporation','Sunshine', 'Cloud9am','Cloud3pm','WindGustDir',

        'WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm', 'RISK_MM'], axis=1, inplace=True)

df.fillna(method='ffill', inplace=True)

# Handling Yes/No values

df.RainTomorrow.replace({'No':'It will not Rain Tomorrow', 'Yes':'It will Rain Tomorrow'}, inplace=True)

df.RainToday.replace({'No':0, 'Yes':1}, inplace=True)

# Check if any more null or missing values left

print(df.isnull().any())

# Check our dataset info

print(df.info())

# Define MinMaxScaler

scaler = preprocessing.MinMaxScaler()

# Declare array of columns need to be scaled

columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 

           'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday']

# Scale values for chosen columns

df[columns] = scaler.fit_transform(df[columns])

# Check dataset content after scaling

print(df.head(5))
from sklearn.neighbors import KNeighborsClassifier # K Neighbors Classifier Algo.

from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier Algo.

from sklearn.linear_model import LogisticRegression # Logistic Regression Classifier Algo.

from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier Algo.

from sklearn.naive_bayes import MultinomialNB # Naive Bayes Classifer Algo.

from sklearn.svm import SVC # Support Vector Classifier Algo.



from sklearn.metrics import confusion_matrix, accuracy_score, classification_report # To get models info.

from sklearn.model_selection import train_test_split # To split data
X = df[columns]

y= df['RainTomorrow']
# Splitting  up data, seting 80% for train and 20% for test.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf_knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,

                               n_jobs=None, n_neighbors=5, p=2, weights='uniform')

clf_knn.fit(X_train, y_train)

y_pred_knn = clf_knn.predict(X_test)

print(accuracy_score(y_test, y_pred_knn))

print(confusion_matrix(y_test, y_pred_knn))

print(classification_report(y_test, y_pred_knn))
clf_rf = RandomForestClassifier(criterion='gini', max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=40)

clf_rf.fit(X_train, y_train)

y_pred_rf = clf_rf.predict(X_test)

print(accuracy_score(y_test, y_pred_rf))

print(confusion_matrix(y_test, y_pred_rf))

print(classification_report(y_test, y_pred_rf))
clf_lr = LogisticRegression(C=2.0, max_iter=200, multi_class='warn', penalty='l1', solver='warn', tol=0.0001)

clf_lr.fit(X_train, y_train)

y_pred_lr = clf_lr.predict(X_test)

print(accuracy_score(y_test, y_pred_lr))

print(confusion_matrix(y_test, y_pred_lr))

print(classification_report(y_test, y_pred_lr))
clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=8, splitter='best')

clf_dt.fit(X_train, y_train)

y_pred_dt = clf_dt.predict(X_test)

print(accuracy_score(y_test, y_pred_dt))

print(confusion_matrix(y_test, y_pred_dt))

print(classification_report(y_test, y_pred_dt))
clf_nb = MultinomialNB()

clf_nb.fit(X_train, y_train)

y_pred_nb = clf_nb.predict(X_test)

print(accuracy_score(y_test, y_pred_nb))

print(confusion_matrix(y_test, y_pred_nb))

print(classification_report(y_test, y_pred_nb))