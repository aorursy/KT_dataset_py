# import all packages

import pandas as pd

from numpy import loadtxt

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



#Loading dataset and to retrieve first 5 rows

dataset = pd.read_csv('../input/diabetes.csv')

dataset.head()



#Split the data into X and Y

X = dataset.iloc[:,0:8]

Y = dataset.iloc[:,8]



# split data into train and test sets with pre-defined "seed=7" for future reference

seed = 7

test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)



# Creating XGBClassifier and fitting into training set of data 

# Used Scikit-learn to fit the model

model = XGBClassifier()

model.fit(X_train, y_train)



# Use Fitted training model to make predictions on test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



#Evaluate the performance of the developed model after predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))