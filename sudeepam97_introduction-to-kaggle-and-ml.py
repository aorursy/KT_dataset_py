# Let us explore our environment. Check the name of the current directory.

%pwd
# List the files in the current directory

%ls

# Visit the parent directory and explore it.

%cd ..

%ls

# Go Back to the original directory

%cd working
# We need the `pandas` module to move around and manipulate our data. Import it.

import pandas as pd



# Input data files are available in the "../input/" directory. Specify this path in a variable.

data_file_path = "../input/train.csv"



# Load the data into a pandas dataframe.

home_data = pd.read_csv(data_file_path)



# Print summary statistics of the data.

home_data.describe()
# Print the first 7 lines of the data. Default is 5 lines if no argument is passed.

home_data.head(7)
# Specify the prediction target.

y = home_data.SalePrice



# Specify the input features. Only numerical data and no missing values for now.

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



# select data corresponding to features in `feature_names`

X = home_data[feature_names]



# Check the data types of X and y

print(type(y))

print(type(X))
# print description statistics of X

print(X.describe())



#  OR

# ----

# X.describe()

# display(X.describe())
# print a few lines of the input data

print(X[1:5])



#  OR

# ----

# print(X.head(5))

# X.head(5)

# display(X[1:5])

# display(X.head(5))
# Now we will create a few decision tree based models and choose the best one.

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

import numpy as np



# Split the data into training and validation sets.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1, train_size = 0.8, test_size = 0.2)



def find_best_model(num_lef_nodes):

    # Specify the model.

    model = DecisionTreeRegressor(max_leaf_nodes = num_leaf_nodes, random_state = 1)



    # Fit the model with the training data.

    model.fit(train_X, train_y)



    # Make predictions on the validation set.

    val_predictions = model.predict(val_X)



    # print the first few predictions.

    print(f"First 5 predictions: {np.round(val_predictions[0:5])}")



    # print the corrosponding ground truth values.

    print(f"Corrosponding Truth Values: {val_y.head().tolist()}")



    # Calculate te Mean Absolute Error.

    val_mae = mean_absolute_error(val_y, val_predictions)

    print(f"Mean Absolute Error for {num_leaf_nodes} leaf nodes is: {round(val_mae)}")

    print("\n")



candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

for num_leaf_nodes in candidate_max_leaf_nodes:

    find_best_model(num_leaf_nodes)
# Create the optimal decision tree model using all the available data.

final_model = DecisionTreeRegressor(random_state = 1, max_leaf_nodes = 100)

final_model.fit(X, y)

predictions = final_model.predict(X)

mae = mean_absolute_error(y, predictions)

print("MAE of the best decision tree model on train.csv is: " + str(round(mae)))
# path to file containing the test data

test_data_path = '../input/test.csv'



# read the test data

test_data = pd.read_csv(test_data_path)



# select the features that we have trained our model on

test_X = test_data[feature_names]



# make predictions 

test_preds = final_model.predict(test_X)



# Export in the specified format

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

output.to_csv('decision_tree_submission.csv', index=False)

print ("Done!")
%reset
import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor



# Read the training data

data_file_path = '../input/train.csv'

home_data = pd.read_csv(data_file_path)



# Specify the prediction target.

y = home_data.SalePrice



# Specify the input features.

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



# Select the data corresponding to features in feature_names

X = home_data[feature_names]



# Now, you would split the data into validation and training sets and calculate MAEs

# for various values of hyperparameters to get the best model. For now, we skip this

# step and create a straightforward Random Forest model to sumbit to the competition.
from sklearn.model_selection import train_test_split



# Split the data into training and validation sets.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, train_size = 0.8, test_size = 0.2)



def find_best_model():

    # Define various models

    model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

    model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

    model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

    model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

    model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

    

    # Iterate through these models to find the best one

    models = [model_1, model_2, model_3, model_4, model_5]

    for i, model in enumerate(models):

        # Fit the model with the training data.

        model.fit(train_X, train_y)



        # Make predictions on the validation set.

        val_predictions = model.predict(val_X)



        # Calculate te Mean Absolute Error.

        val_mae = mean_absolute_error(val_y, val_predictions)

        print(f"Mean Absolute Error for model_{i} is: {round(val_mae)}")

    

find_best_model()
# Specify the Model

rf_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)



# Fit the model on full data

rf_model.fit(X, y)



# path to file containing the test data

test_data_path = '../input/test.csv'



# read the test data

test_data = pd.read_csv(test_data_path)



# select the features that we have trained our model on

test_X = test_data[feature_names]



# make predictions 

test_preds = rf_model.predict(test_X)



# Export in the specified format

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

output.to_csv('random_forest_submission.csv', index=False)

print ("Done!")