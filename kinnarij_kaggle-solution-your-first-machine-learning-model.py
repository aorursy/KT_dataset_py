# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

df = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")
# print the list of columns in the dataset to find the name of the prediction target
df.columns
Y = df['SalePrice']

# Check your answer
step_1.check()
# The lines below will show you a hint or the solution.
# step_1.hint() 
# step_1.solution()
# Create the list of features below
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select data corresponding to features in feature_names
X = df[feature_names]

# Check your answer
step_2.check()
# step_2.hint()
# step_2.solution()
# Review data
# print description or statistics from X
print(X)
print("----------------------------------")
print(Y)
# print the top few lines
#print(_)
from sklearn import svm, datasets, tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=0, max_depth=5)

# Fit the model
iowa_model.fit(X,Y)

# Check your answer
step_3.check()
# step_3.hint()
# step_3.solution()
predictions = iowa_model.predict(X)
print(predictions)

# Check your answer
step_4.check()
# step_4.hint()
# step_4.solution()
# You can write code in this cell
#print("Decision Tree",accuracy_score(Y,predictions)*100,"%")