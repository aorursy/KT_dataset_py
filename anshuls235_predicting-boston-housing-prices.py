# Import libraries necessary for this project

import numpy as np

import pandas as pd

from sklearn.model_selection import ShuffleSplit



#To ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Import supplementary visualizations code visuals.py

import visuals as vs



# Pretty display for notebooks

%matplotlib inline



# Load the Boston housing dataset

data = pd.read_csv('../input/bostonhoustingmlnd/housing.csv')

prices = data['MEDV']

features = data.drop('MEDV', axis = 1)

    

# Success

print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
# TODO: Minimum price of the data

minimum_price = np.min(prices)



# TODO: Maximum price of the data

maximum_price = np.max(prices)



# TODO: Mean price of the data

mean_price = np.mean(prices)



# TODO: Median price of the data

median_price = np.median(prices)



# TODO: Standard deviation of prices of the data

std_price = np.std(prices)



# Show the calculated statistics

print("Statistics for Boston housing dataset:\n")

print("Minimum price: ${}".format(minimum_price)) 

print("Maximum price: ${}".format(maximum_price))

print("Mean price: ${}".format(mean_price))

print("Median price ${}".format(median_price))

print("Standard deviation of prices: ${}".format(std_price))
# TODO: Import 'r2_score'

from sklearn.metrics import r2_score



def performance_metric(y_true, y_predict):

    """ Calculates and returns the performance score between 

        true and predicted values based on the metric chosen. """

    

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'

    score = r2_score(y_true, y_predict)

    

    # Return the score

    return score
# Calculate the performance of this model

score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])

print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))
# TODO: Import 'train_test_split'

from sklearn.model_selection import train_test_split



# TODO: Shuffle and split the data into training and testing subsets

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 42)



# Success

print("Training and testing split was successful.")
# Produce learning curves for varying training set sizes and maximum depths

vs.ModelLearning(features, prices)
vs.ModelComplexity(X_train, y_train)
# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

from sklearn.metrics import make_scorer

from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor



def fit_model(X, y):

    """ Performs grid search over the 'max_depth' parameter for a 

        decision tree regressor trained on the input data [X, y]. """

    

    # Create cross-validation sets from the training data

    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)

    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)

    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)



    # TODO: Create a decision tree regressor object

    regressor = DecisionTreeRegressor()



    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10

    params = {'max_depth': np.arange(1,11)}



    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 

    scoring_fnc = make_scorer(r2_score)



    # TODO: Create the grid search cv object --> GridSearchCV()

    # Make sure to include the right parameters in the object:

    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.

    grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv = cv_sets)



    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(X, y)



    # Return the optimal model after fitting the data

    return grid.best_estimator_
# Fit the training data to the model using grid search

reg = fit_model(X_train, y_train)



# Produce the value for 'max_depth'

print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
# Produce a matrix for client data

client_data = [[5, 17, 15], # Client 1

               [4, 32, 22], # Client 2

               [8, 3, 12]]  # Client 3



# Show predictions

for i, price in enumerate(reg.predict(client_data)):

    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
from matplotlib import pyplot as plt



clients = np.transpose(client_data)

pred = reg.predict(client_data)

for i, feature in enumerate(['RM','LSTAT','PTRATIO']):

    plt.scatter(features[feature], prices, alpha=0.2, c='blue')

    plt.scatter(clients[i], pred, c='black', marker='s')

    plt.xlabel(feature)

    plt.ylabel('MEDV')

    plt.show()
vs.PredictTrials(features, prices, fit_model, client_data)