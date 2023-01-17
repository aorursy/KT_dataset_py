## Importing required libraries
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
## Reading the data
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.head(2)
## Id & Unnamed32 are unnessary columns
## Checking Null values 
fig, ax = plt.subplots(figsize=(12,12)) # increase the figure size
sns.heatmap(df.isnull()) 
# Unnamed32 have null values
# Dropping the ID column
df.drop('id', inplace = True, axis =1)
# Dropping the UNNAMED32 column
df.drop('Unnamed: 32', inplace = True, axis =1)
# Count check on Malignant & Benign Catagory
sns.countplot(df['diagnosis'])
# seems like Benign is more in number but not that much
from sklearn.model_selection import train_test_split
## y is diagnosis
## x is everthing except diagnosis columns

x = df.drop('diagnosis',axis =1)
y = df['diagnosis']

x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size = 0.3,random_state = 101)
# importing support vector classifer 
from sklearn.svm import SVC
svm_model = SVC()
## Fitting the model to training data
svm_model.fit(x_train,y_train)
## predicting on our test data
svm_predict = svm_model.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report
## The classificatiion report and confusion matrix 
print(classification_report(svm_predict,y_test))
print('/n')
print(confusion_matrix(svm_predict,y_test))
# Accuracy = 92%
## Grid Search is one of the technique used
from sklearn.model_selection import GridSearchCV
## Giving the parameter grid lower and upper values 
param_grid = {'C':[0.1,1,10,100,100],
             'gamma':[1,0.1,0.01,0.001,0.0001]}
## Creating a grid model with SVC Model
grid = GridSearchCV(SVC(),param_grid,verbose=3)
## Fitting the grid model on our training data
grid.fit(x_train,y_train)
## The best fit parameter is given by:
grid.best_params_
grid.best_estimator_
## Checking the best score possible on Grid SVC model
grid.best_score_
## Predicting on the test data
grid_prediction = grid.predict(x_test)
## Classification report and confusion matrix
print(classification_report(y_test,grid_prediction))
print('/n')
print(confusion_matrix(y_test,grid_prediction))
# Accuracy = 94%