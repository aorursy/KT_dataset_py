#imports

#numpy for array calculations

import numpy as np

#pandas for working with data

import pandas as pd

#matplotlib for graphs

import matplotlib.pyplot as plt
#read csv file and make it as dataframe

df = pd.read_csv("../input/heart-disease-uci/heart.csv")

#print information of the data

df.info()
#print first 5 data

df.head()
#print first 10 data

df.head(10)
#printing the total number of peoples without heart disease and with heart disease

df.target.value_counts()
#Our features = age - thal

features = df.iloc[:, 0:13]

label = df['target']
features
#this is the codebase for generating graph, you don't need to understand it right now. assume it as a block

pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex')

plt.ylabel('Frequency')

plt.show()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
#scikit learn is a ready to go machine learning library 

from sklearn import ensemble

clf_rand = ensemble.RandomForestClassifier()
from sklearn import linear_model

clf_log = linear_model.LogisticRegression()
#we need to shuffle the data then we need split the dataset as training data and testing data

from sklearn.model_selection import train_test_split
#random state is a calculated randomness

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.1, random_state=2)

X_train
#fitting the training dataset to the classifier 

clf_rand.fit(X_train, y_train)

clf_log.fit(X_train, y_train)
#predicting the test dataset through classifier

pred_rand = clf_rand.predict(X_test)

pred_log = clf_log.predict(X_test)
print(pred_rand)

print(pred_log)
#confusion matrix for random forest

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred_rand)

print(cm)
#confusion matrix for logistic regression

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred_log)

print(cm)
#how accurately our classifier predicts those test data

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred_rand)
#how accurately our classifier predicts those test data

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred_log)
#predicting with the new value

clf_rand.predict([[39, 0, 1, 135, 208, 0, 0, 171, 0, 1.5, 2, 0, 2]])
#to save the model

from joblib import dump, load
#saving the classifier, it can be used by apps and websites 

dump(clf_rand, 'model_rand.joblib')

dump(clf_log, 'model_log.joblib')
#loading the model 

loaded_model = load('model_rand.joblib')
#prediction using loaded model

pred = loaded_model.predict(X_test)

pred