# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

#save filepath to variable for easier access

asteroid_file_path = '../input/prediction-of-asteroid-diameter/Asteroid.csv'



#read the data and store data in DataFrame titled asteroid_data

asteroid_data = pd.read_csv(asteroid_file_path)



# Now, let's remove the columns we know we won't be using: condition code, n_obs_used, i, data_arc, H, extent, spec_b

asteroid_data.drop(columns=['condition_code', 'i', 'H', 'extent', 'data_arc', 'n_obs_used', 'spec_B', 'spec_T', 'pha', 'neo', 'IR', 'GM'], inplace=True)



print(asteroid_data.head(40))
# Now we remove any NA rows, because we only want useful data:

asteroid_data.dropna(axis=0, inplace=True)

print(asteroid_data.head(40))

print(asteroid_data.shape)
asteroid_features = ['a','e','G','om','w','q','ad','per_y','albedo','rot_per','BV','UB','moid']



X = asteroid_data[asteroid_features]

Y = asteroid_data['diameter']



print(X.describe())

print(X.head(100))
#Define model. Specify a number for random_state to ensure same results each run

asteroid_model = DecisionTreeRegressor(random_state=1)



#Fit model

asteroid_model.fit(X, Y)
print("Making predictions for the following 5 asteroids:")

print(X.head())

print("The predictions are")

print(asteroid_model.predict(X.head()))

print("The real diameters are")

print(Y.head())



predicted_diameters = asteroid_model.predict(X)

mean_squared_error(Y, predicted_diameters)



# split data into training and validation data, for both features and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state = 0)

# # Define model

asteroid_model = DecisionTreeRegressor()

# # Fit model

asteroid_model.fit(train_X, train_y)
# get predicted prices on validation data

val_predictions = asteroid_model.predict(val_X)

print(mean_squared_error(val_y, val_predictions))









def get_mse(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mse = mean_squared_error(val_y, preds_val)

    return(mse)



for max_leaf_nodes in [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]:

     my_mse = get_mse(max_leaf_nodes, train_X, val_X, train_y, val_y)

     print("Max leaf nodes: %d  \t\t Mean Squared Error:  %d" %(max_leaf_nodes, my_mse))



best_tree_size = 3



# Fill in argument to make optimal size and uncomment

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)



#fit the final model and uncomment the next two lines

final_model.fit(train_X, train_y)

val_predictions = final_model.predict(val_X)

print(mean_squared_error(val_y, val_predictions))





mse_model = DecisionTreeRegressor(criterion='mse', random_state=1)

mse_model.fit(train_X, train_y)

val_predictions = mse_model.predict(val_X)

print(mean_squared_error(val_y, val_predictions))
