%matplotlib inline



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import sqlite3

import numpy as np

from numpy import random

from time import time
#load data (make sure you have downloaded database.sqlite)

with sqlite3.connect('../input/database.sqlite') as con:

    Player_Attributes = pd.read_sql_query("SELECT * from Player_Attributes", con)

Player_Attributes.shape
#select relevant fields

Player_Attributes.dropna(inplace=True)

Player_Attributes.drop(['id', 'player_fifa_api_id', 'player_api_id', 'date'], axis = 1, inplace = True)

overall_rating = Player_Attributes['overall_rating']

features = Player_Attributes.drop('overall_rating', axis = 1)

features.head()
# Use encoding to convert catogarial values to numerical data using LabelEncoder()

from sklearn import preprocessing



le_sex = preprocessing.LabelEncoder()



#to convert into numbers



features.preferred_foot = le_sex.fit_transform(features.preferred_foot)

features.attacking_work_rate = le_sex.fit_transform(features.attacking_work_rate)

features.defensive_work_rate = le_sex.fit_transform(features.defensive_work_rate)

features.head()



# to convert back

# train.Sex = le_sex.inverse_transform(train.Sex)
list(features.columns)
# Feature scaling using MinMaxScaler

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

scaled_features = min_max_scaler.fit_transform(features)

scaled_features
# PCA to reduce feature reduction and to improve model speed

from sklearn.decomposition import PCA

pca = PCA(n_components = 6)

pca_features = pca.fit_transform(scaled_features)

print (pca.components_)
# Feature selection using sklearn SelectKBest

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

feature_reg = SelectKBest(f_regression, k=6)

X_new = feature_reg.fit_transform(scaled_features, overall_rating)

X_new
# Train and predict model on Decision tree

from sklearn import tree

from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.cross_validation import train_test_split

from time import time



reg1 = tree.DecisionTreeClassifier()

reg2 = linear_model.SGDRegressor()



regs = {reg1:"Decision Tree", reg2:"SGDRegressor"}



for key in regs:

    t0 = time()

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, overall_rating, test_size=0.25, random_state=0)



    print ("--------------------")

    print (regs[key])

    print ("--------------------")

    print ("Time taken to split the data: {}".format(time()-t0))



    t1 = time()

    key.fit(X_train, y_train)

    print ("Time taken to train the model: {}".format(time()-t1))



    t2 = time()

    pred = key.predict(X_test)

    print ("Time taken to predict the model: {}".format(time()-t2))



    t3 = time()

    print ("r2 score of this model is: {}".format(r2_score(y_test, pred)))

    print ("Time taken to find the accuracy of model: {}".format(time()-t3))
# Use GridSearch to tune the model

def fit_model(X, y):

    """ Performs grid search over the 'max_depth' parameter for a 

        decision tree regressor trained on the input data [X, y]. """

    

    # Create cross-validation sets from the training data

    from sklearn.cross_validation import ShuffleSplit

    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)



    # TODO: Create a decision tree regressor object

    from sklearn.tree import DecisionTreeRegressor

    from sklearn.svm import SVC

    from sklearn import linear_model

    

    

    regressor1 = DecisionTreeRegressor()

    regressor2 = SVC()

    regressor3 = linear_model.SGDRegressor()



    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10

    tree_params = {'max_depth' : [3, 6, 9, 20, 100], 'min_samples_split':[2, 3, 4, 5]}

    svm_params = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

    sgd_params = {'loss':['squared_loss', 'huber'], 'penalty': ['none', 'l2', 'l1', 'elasticnet'], 'n_iter':[10, 75, 100, 500]}

    

    

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 

    from sklearn.metrics import make_scorer

    scoring_fnc = make_scorer(performance_metric)



    # TODO: Create the grid search object

    from sklearn.grid_search import GridSearchCV

    

    # Updated cv_sets and scoring parameter

    grid = GridSearchCV(regressor3, sgd_params, scoring = scoring_fnc, cv = cv_sets)



    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(X, y)



    # Return the optimal model after fitting the data

    return grid.best_estimator_



def performance_metric(y_true, y_predict):

    

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'

    from sklearn.metrics import r2_score

    

    score = r2_score(y_true, y_predict)

    # Return the score

    return score
'''from time import time

t0 = time()

grid_reg = fit_model(pca_features, overall_rating)

print (grid_reg.score)

# grid_pred = grid_reg()

print ("Time taken to train and predict using GridSearch: {}".format(time() - t0))

print ("Best parameters are: {}".format(grid_reg.get_params()))'''