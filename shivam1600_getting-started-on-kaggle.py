# importing libraries

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split # For Splitting testing and training data
from sklearn.metrics import mean_absolute_error # For calculating mean absolute error
from sklearn.tree import DecisionTreeRegressor # Model builder
from sklearn.preprocessing import Imputer # To fill missing values
# importing train and test data
titanic_train_file_path = '../input/train.csv'
titanic_train_data = pd.read_csv(titanic_train_file_path)

titanic_test_file_path = '../input/test.csv'
titanic_test_data = pd.read_csv(titanic_test_file_path)
# Taking a look at train data
print(titanic_train_data.shape)
titanic_train_data.head()
# Taking a look at test data
print(titanic_test_data.shape)
titanic_test_data.head()
# Choose target and predictors
y = titanic_train_data.Survived
survival_predictors = ['Pclass','Age', 'SibSp', 
                        'Parch', 'Fare']
X = titanic_train_data[survival_predictors]
# Dealing with missing values
# In this case we are filling missing value with most frequent value of the column
imp_train = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
X = imp_train.fit_transform(X=X)
# Split data into training and validation set
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
titanic_model_decision_tree = DecisionTreeRegressor()
# Fit model
titanic_model_decision_tree.fit(train_X, train_y)
# get predicted survivals on validation data
val_predictions = titanic_model_decision_tree.predict(val_X)

val_y_list = val_y.tolist()
val_predictions_list = [int(x) for x in val_predictions.tolist()]

val_compare = [ x == y for (x,y) in zip(val_y_list, val_predictions_list)]

print("Accuracy = ",sum(val_compare)/len(val_compare))
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
mod_test = imp.fit_transform(X=titanic_test_data[survival_predictors])
final_prediction = titanic_model_decision_tree.predict(mod_test)
final_prediction_list = [int(x) for x in final_prediction.tolist()]
# create data frame for submission
my_submission = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': final_prediction_list})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)