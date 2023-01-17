# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split

# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

# Check your answer
step_1.check()
# The lines below will show you a hint or the solution.
step_1.hint() 
step_1.solution()

# You imported DecisionTreeRegressor in your last exercise
# and that code has been copied to the setup code above. So, no need to
# import it again

# Specify the model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data

iowa_model.fit(train_X,train_y)

# Check your answer
step_2.check()
step_2.hint()
step_2.solution()
# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

# Check your answer
step_3.check()
step_3.hint()
step_3.solution()
# print the top few validation predictions
print(val_predictions[0:9])
      
# print the top few actual prices from validation data
print(val_y[0:9])
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_predictions,val_y)

# uncomment following line to see the validation_mae
print(val_mae)

# Check your answer
step_4.check()
step_4.hint()
step_4.solution()
#importing the plot function using matplotlib

from matplotlib.pyplot import plot

#training data plotting
training_difference_vector = train_y - iowa_model.predict(train_X)
training_diff_plot = plot(training_difference_vector, animated = True, c="Orange", ls= "-.")

training_diff_plot

#testing data plotting

testing_difference_vector = val_y - val_predictions
testing_diff_plot = plot(testing_difference_vector, animated = True, c="Purple", ls= "-.")
print(testing_diff_plot)

#absolute value vector creations

absolute_train_diffs= abs(training_difference_vector)
absolute_test_diffs= abs(testing_difference_vector)

max_abs_train_diff= max(absolute_train_diffs)
max_abs_test_diff= max(absolute_test_diffs)
min_abs_train_diff= min(absolute_train_diffs)
min_abs_test_diff= min(absolute_test_diffs)

print(absolute_train_diffs.describe())
print(plot(absolute_train_diffs))
print("The maximum value difference takes in training data is " + str(max_abs_train_diff) +", and the minimum value it takes in the training data is "+ str(min_abs_train_diff))
print (absolute_test_diffs.describe())

print("The maximum value difference takes in validation data is " + str(max_abs_test_diff) +", and the minimum value it takes in the validation data is "+ str(min_abs_test_diff))
from seaborn import distplot

distplot(absolute_train_diffs, hist=True, kde=True, color="grey", rug= True, vertical= True)
distplot(absolute_test_diffs, hist=True, kde=True, color="red", rug= True, vertical= True, axlabel="difference")