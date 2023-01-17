# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics

from sklearn.model_selection import train_test_split
df1 = pd.read_csv('../input/class.csv')

df2 = pd.read_csv('../input/zoo.csv')

print(df1.describe())

print(df2.describe())
df1.head()
df2.head()
#Classifiying data and target

X = df2.iloc[:,1:17].values   # Not considering the name of the animal - placing it 

y = df2.iloc[:,17].values     # Class number to be assigned (labels)



# Separating into test and training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
b = df2.columns.values

index = [0,17]

feature_names = np.delete(b,index)

feature_names
target_names = np.array(df1['Class_Type'])

target_names
#Standardizing data



from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X_train)
from sklearn.metrics import accuracy_score

from sklearn import svm

model = svm.LinearSVC()

model.fit(X_train, y_train)



# make predictions

expected = y_test

predicted = model.predict(X_test)



print ('The accurancy of SVM  is ' + str(accuracy_score(expected, predicted)))

# summarize the fit of the model

print(metrics.classification_report(expected, predicted))
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()

model.fit(X_train,y_train)



# make predictions

expected = y_test

predicted = model.predict(X_test)



print ('The accurancy of Decision Tree Classifier  is ' + str(accuracy_score(expected, predicted)))

# summarize the fit of the model

print(metrics.classification_report(expected, predicted))
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()



model.fit(X_train,y_train)



# make predictions

expected = y_test

predicted = model.predict(X_test)



print ('The accurancy of Linear Discriminant Analysis  is ' + str(accuracy_score(expected, predicted)))

# summarize the fit of the model

print(metrics.classification_report(expected, predicted))