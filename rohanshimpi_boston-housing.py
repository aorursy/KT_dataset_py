# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



from sklearn.cross_validation import ShuffleSplit

from sklearn.cross_validation import train_test_split

import sklearn.learning_curve as curves

from sklearn.tree import DecisionTreeRegressor

from sklearn.cross_validation import ShuffleSplit

from sklearn.metrics import make_scorer

from sklearn.grid_search import GridSearchCV

# Import supplementary visualizations code visuals.py

#import visuals as vs



# Pretty display for notebooks

%matplotlib inline

import matplotlib.pyplot as pl

# Any results you write to the current directory are saved as output.



print(check_output(["ls", "../input"]).decode("utf8"))
# Load the Boston housing dataset

data = pd.read_csv('../input/housing.csv')

prices = data['MEDV']

features = data.drop('MEDV', axis = 1)



# Success

print('Boston housing dataset has {} data points with {} variables each.'.format(*data.shape))
minimum_price = np.min(prices)

maximum_price = np.max(prices)

mean_price = np.mean(prices)

median_price = np.median(prices)

std_price = np.std(prices)



# Show the calculated statistics

print("Statistics for Boston housing dataset:\n")

print("Minimum price: ${:,.2f}".format(minimum_price))

print("Maximum price: ${:,.2f}".format(maximum_price))

print("Mean price: ${:,.2f}".format(mean_price))

print("Median price ${:,.2f}".format(median_price))

print("Standard deviation of prices: ${:,.2f}".format(std_price))
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
# TODO: Shuffle and split the data into training and testing subsets

X_train, X_test, y_train, y_test = train_test_split(features, prices, random_state=0, test_size=0.2)
def ModelLearning(X, y):

    """ Calculates the performance of several models with varying sizes of training data.

        The learning and testing scores for each model are then plotted. """

    

    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)



    # Generate the training set sizes increasing by 50

    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)



    # Create the figure window

    fig = pl.figure(figsize=(10,7))



    # Create three different models based on max_depth

    for k, depth in enumerate([1,3,6,10]):

        

        # Create a Decision tree regressor at max_depth = depth

        regressor = DecisionTreeRegressor(max_depth = depth)



        # Calculate the training and testing scores

        sizes, train_scores, test_scores = curves.learning_curve(regressor, X, y, \

            cv = cv, train_sizes = train_sizes, scoring = 'r2')

        

        # Find the mean and standard deviation for smoothing

        train_std = np.std(train_scores, axis = 1)

        train_mean = np.mean(train_scores, axis = 1)

        test_std = np.std(test_scores, axis = 1)

        test_mean = np.mean(test_scores, axis = 1)



        # Subplot the learning curve 

        ax = fig.add_subplot(2, 2, k+1)

        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')

        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')

        ax.fill_between(sizes, train_mean - train_std, \

            train_mean + train_std, alpha = 0.15, color = 'r')

        ax.fill_between(sizes, test_mean - test_std, \

            test_mean + test_std, alpha = 0.15, color = 'g')

        

        # Labels

        ax.set_title('max_depth = %s'%(depth))

        ax.set_xlabel('Number of Training Points')

        ax.set_ylabel('Score')

        ax.set_xlim([0, X.shape[0]*0.8])

        ax.set_ylim([-0.05, 1.05])

    

    # Visual aesthetics

    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)

    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)

    fig.tight_layout()

    fig.show()
def ModelComplexity(X, y):

    """ Calculates the performance of the model as model complexity increases.

        The learning and testing errors rates are then plotted. """

    

    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)



    # Vary the max_depth parameter from 1 to 10

    max_depth = np.arange(1,11)



    # Calculate the training and testing scores

    train_scores, test_scores = curves.validation_curve(DecisionTreeRegressor(), X, y, \

        param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')



    # Find the mean and standard deviation for smoothing

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)



    # Plot the validation curve

    pl.figure(figsize=(7, 5))

    pl.title('Decision Tree Regressor Complexity Performance')

    pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')

    pl.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')

    pl.fill_between(max_depth, train_mean - train_std, \

        train_mean + train_std, alpha = 0.15, color = 'r')

    pl.fill_between(max_depth, test_mean - test_std, \

        test_mean + test_std, alpha = 0.15, color = 'g')

    

    # Visual aesthetics

    pl.legend(loc = 'lower right')

    pl.xlabel('Maximum Depth')

    pl.ylabel('Score')

    pl.ylim([-0.05,1.05])

    pl.show()

def PredictTrials(X, y, fitter, data):

    """ Performs trials of fitting and predicting data. """



    # Store the predicted prices

    prices = []



    for k in range(10):

        # Split the data

        X_train, X_test, y_train, y_test = train_test_split(X, y, \

            test_size = 0.2, random_state = k)

        

        # Fit the data

        reg = fitter(X_train, y_train)

        

        # Make a prediction

        pred = reg.predict([data[0]])[0]

        prices.append(pred)

        

        # Result

        print("Trial {}: ${:,.2f}".format(k+1, pred))



    # Display price range

    print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))
# Produce learning curves for varying training set sizes and maximum depths

ModelLearning(features, prices)
ModelComplexity(X_train, y_train)
def fit_model(X, y):

    """ Performs grid search over the 'max_depth' parameter for a 

        decision tree regressor trained on the input data [X, y]. """

    

    # Create cross-validation sets from the training data

    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)

    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)

    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)



    # TODO: Create a decision tree regressor object

    regressor = DecisionTreeRegressor()



    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10

    params = {'max_depth':np.arange(1,11)}



    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 

    scoring_fnc = make_scorer(performance_metric, greater_is_better=True)



    # TODO: Create the grid search cv object --> GridSearchCV()

    # Make sure to include the right parameters in the object:

    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.

    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc,cv=cv_sets)



    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(X, y)



    # Return the optimal model after fitting the data

    return grid.best_estimator_
# Fit the training data to the model using grid search

reg = fit_model(X_train, y_train)



# Produce the value for 'max_depth'

print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))