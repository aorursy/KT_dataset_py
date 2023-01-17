# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
# importing the heart disease dataset

heart = pd.read_csv('../input/heart-disease-uci/heart.csv')
heart.info()
sns.lmplot(data=heart,x='age',y='trestbps')

# Data shows resting blood pressure higher in older patients
sns.lmplot(data=heart,x='age',y='thalach')

# Data shows maximum heart rate acheived lower in older patients
# importing the classifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
# set a few different parameters to check which fits bets into the model

paramgrid = {'C': [1,10,100,1000,10000], 'gamma': [1,0.1,0.01,0.001,0.0001]}
model = GridSearchCV(estimator=SVC(),param_grid=paramgrid, verbose=3)
from sklearn.model_selection import train_test_split
# split the data in training and test data

X_train,X_test,y_train,y_test = train_test_split(heart.drop('target',axis=1),heart['target'],test_size=0.33)
model.fit(X_train,y_train)
# CVgridsearch chooses the best parameters to go ahead with

model.best_params_
model.best_estimator_
model.best_score_
predictions = model.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))

print('')

print(classification_report(y_test,predictions))