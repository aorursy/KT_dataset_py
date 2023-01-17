# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
val_data = pd.read_csv('../input/home-data-for-ml-course/test.csv')

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *


print(val_data.head())
print("Setup Complete")
# print the list of columns in the dataset to find the name of the prediction target
home_data.columns

y = home_data['SalePrice']

# Check your answer
step_1.check()
# The lines below will show you a hint or the solution.
# step_1.hint() 
# step_1.solution()
# Create the list of features below
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF', 'FullBath', 'BedroomAbvGr','TotRmsAbvGrd']
 
# Select data corresponding to features in feature_names
X = home_data[feature_names]
X_val = val_data[feature_names]
print(val_data.columns)
# Check your answer
step_2.check()
# step_2.hint()
# step_2.solution()
# Review data
# print description or statistics from X
#print(_)
from sklearn.model_selection import train_test_split
# print the top few lines
#print(_)
from sklearn.ensemble import RandomForestClassifier
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = RandomForestClassifier(n_estimators=10)

# Fit the model
iowa_model.fit(X,y)

 
# step_3.hint()
# step_3.solution()
predictions = iowa_model.predict(X_val)
print(predictions)
print(len(predictions))
val_data['SalePrice'] = predictions
export_preds = ['Id', 'SalePrice']
val_data[export_preds].to_csv('prediction.csv') 
# step_4.hint()
# step_4.solution()
# You can write code in this cell
y.head() , predictions[0:5]