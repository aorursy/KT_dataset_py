#imports Pandas package

import pandas as pd
#Create's DataFrame 'train' which reads from the train.csv file

train = pd.read_csv("../input/train.csv")
#Displays the first 5 rows of Data from the train.csv

train.head()
#Creates a Matrix that scikit-learn will learn from

feature_cols = ["Pclass", "Parch"]
#Creates feature Matrix with all rows, and feature_cols columns

X = train.loc[:, feature_cols]
#Displays the size of the matrix (row x column)

X.shape
#Declares the target vector being the 'survived' series

y = train.Survived
#Displays the size of the target vector series

y.shape
#Creates classification model via Sklearn's LogisticRegression

from sklearn.linear_model import LogisticRegression

#Declares variable logreg as classification model

logreg = LogisticRegression()

#Fits the model to training data

logreg.fit(X, y)
#Reads in test data for predictions to be made

test = pd.read_csv("../input/test.csv")
#Displays the first 5 rows of data 

test.head()
#Creates a new matrix based on test.csv data 

X_new = test.loc[:, feature_cols]
#Displays the amount of predictions that need to be made

X_new.shape
#Creates a new prediction class using Logistic Regression

new_pred_class = logreg.predict(X_new)
#Displays the prediction results

new_pred_class
#Creates a csv file with the results

pd.DataFrame({'PassengerID':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerID').to_csv('sub.csv')