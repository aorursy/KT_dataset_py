!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@fix-correct-message
import sys

sys.path.append('/kaggle/working')
# Code you have previously used to load data

import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex3 import *



print("Setup Complete")
# print the list of columns in the dataset to find the name of the prediction target

y = ____



step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()

#%%RM_IF(PROD)%%



y = home_data.SalePrice

step_1.assert_check_passed()
# Create the list of features below

feature_names = ___



# select data corresponding to features in feature_names

X = ____



step_2.check()
# step_2.hint()

# step_2.solution()
#%%RM_IF(PROD)%%



# Create the list of features below

feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF",

                      "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]



# select data corresponding to features in feature_names

X = home_data[feature_names]

print("Correct")

step_2.assert_check_passed()
# Review data

# print description or statistics from X

#print(_)



# print the top few lines

#print(_)

# from _ import _

#specify the model. 

#For model reproducibility, set a numeric value for random_state when specifying the model

iowa_model = ____



# Fit the model

____



step_3.check()
# step_3.hint()

# step_3.solution()
#%%RM_IF(PROD)%%



from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)

print("Incorrect")

step_3.assert_check_failed()
#%%RM_IF(PROD)%%



iowa_model.fit(X,y)

step_3.check()
#%%RM_IF(PROD)%%

step_3.hint()
predictions = ____

print(predictions)

step_4.check()
# step_4.hint()

# step_4.solution()
#%%RM_IF(PROD)%%

predictions = iowa_model.predict(X)

step_4.assert_check_passed()
# You can write code in this cell
