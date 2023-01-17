# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('/kaggle/input/boston-housepredict/boston_train.csv')
# Correlation

plt.subplots(figsize=(20,15))

correlation_matrix = df_train.corr().round(2)

sns_plot=sns.heatmap(data=correlation_matrix, annot=True)
df_train.head()
prices=df_train['medv']

features=df_train.drop(['medv'],axis=1)
print("BASIC STATS FOR OUR THE BOSTON HOUSING DATASET  \n")

MAX_PRICE=np.max(prices)

MIN_PRICE=np.min(prices)

MEAN_PRICE=np.mean(prices)

MEDIAN_PRICE=np.median(prices)

STD_PRICE=np.std(prices)

print("Max Price in USD 1000’s = ${:,.2f}".format(MAX_PRICE))

print("Min Price in USD 1000’s = ${:,.2f}".format(MIN_PRICE))

print("Mean Price in USD 1000’s = ${:,.2f}".format(MEAN_PRICE))

print("Median Price in USD 1000’s = ${:,.2f}".format(MEDIAN_PRICE))

print("Standard Dev Price in USD 1000’s = ${:,.2f}".format(STD_PRICE))
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):

    """ Calculates and returns the performance score between 

        true and predicted values based on the metric chosen. """

    

    # Calculate the performance score between 'y_true' and 'y_predict'

    score = r2_score(y_true, y_predict)

    

    # Return the score

    return score
from sklearn.model_selection import train_test_split



# Shuffle and split the data into training and testing subsets

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=0)



# Success

print ("Training and testing split was successful.")
from sklearn.metrics import make_scorer

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV 

from sklearn.model_selection import ShuffleSplit



def fit_model(X, y):

    """ Performs grid search over the 'max_depth' parameter for a 

        decision tree regressor trained on the input data [X, y]. """

    

   

    cv_sets = ShuffleSplit(X.shape[0],  test_size = 0.20, random_state = 0)



    # Create a decision tree regressor object

    

    regressor = DecisionTreeRegressor()



    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10

    params = {'max_depth':range(1,11)}



    # Transform 'performance_metric' into a scoring function using 'make_scorer' 

    scoring_fnc = make_scorer(performance_metric)



    # Make sure to include the right parameters in the object:

    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.

    grid = GridSearchCV(regressor,params,scoring=scoring_fnc,cv=cv_sets)



    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(X, y)



    # Return the optimal model after fitting the data

    return grid.best_estimator_
reg = fit_model(X_train, y_train)



#Print the value for 'max_depth'

print ("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

reg = fit_model(X_train, y_train)

pred = reg.predict(X_test)

score = performance_metric(y_test,pred)

print("R Squared Value: " + str(score))