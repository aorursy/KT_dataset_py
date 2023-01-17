# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting correlation

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import pandas as pd

mvp_votings = pd.read_csv("../input/nba-mvp-votings-through-history/mvp_votings.csv")

test_data = pd.read_csv("../input/nba-mvp-votings-through-history/test_data.csv")
mvp_votings.head()
test_data.head()
print(mvp_votings.columns)

print(test_data.columns)
#The ever-important scoring metric: points per game

plt.plot(mvp_votings['pts_per_g'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Points Per Game')
plt.plot(mvp_votings['ast_per_g'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Assists Per Game')
plt.plot(mvp_votings['trb_per_g'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Total Rebounds Per Game')
plt.plot(mvp_votings['stl_per_g'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Steals Per Game')
plt.plot(mvp_votings['blk_per_g'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Blocks Per Game')
plt.plot(mvp_votings['win_pct'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Win Percentage')
plt.figure(1)

plt.plot(mvp_votings['fga'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Field Goal Attempts')

plt.figure(2)

plt.plot(mvp_votings['fg3a'], mvp_votings['points_won'], 'bo', label = '2')

plt.ylabel('MVP Votes Won')

plt.xlabel('3-Point Field Goal Attempts')

plt.figure(3)

plt.plot(mvp_votings['fta'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Free Throw Attempts')
plt.figure(1)

plt.plot(mvp_votings['fg_pct'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Field Goal Percentage')

plt.figure(2)

plt.plot(mvp_votings['fg3_pct'], mvp_votings['points_won'], 'bo', label = '2')

plt.ylabel('MVP Votes Won')

plt.xlabel('3-Point Field Goal Percentage')

plt.figure(3)

plt.plot(mvp_votings['ft_pct'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Free Throw Percentage')
#Data exploration, what correlatess best with the target (votes)

plt.plot(mvp_votings['per'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Player Efficiency Rating')
plt.plot(mvp_votings['ws'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Win Shares')
plt.plot(mvp_votings['ws_per_48'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Win Shares Per 48')
plt.plot(mvp_votings['usg_pct'], mvp_votings['points_won'], 'bo')

plt.ylabel('MVP Votes Won')

plt.xlabel('Usage Percentage')
y = mvp_votings.points_won

y_2 = mvp_votings.award_share

feature_names = ['fga', 'fg3a', 'fta', 'per', 'ts_pct', 'usg_pct', 'bpm', 'win_pct', 'g', 'mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct', 'ws_per_48']

X = mvp_votings[feature_names]
def mse_random_forest(estimators, data_X, data_y, val_X, val_y):

    test_model = RandomForestRegressor(n_estimators = estimators, random_state = 1)

    test_model.fit(data_X, data_y)

    predictions_y = test_model.predict(val_X)

    return mean_squared_error(val_y, predictions_y)



def mse_multi_layer_perceptron(hidden_layer_size_test, alpha_given, data_X, data_y, val_X, val_y):

    test_model = MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = hidden_layer_size_test, alpha = alpha_given, random_state = 1) 

    #The LBFGS optimizer is being used due to dataset size

    test_model.fit(data_X, data_y)

    predictions_y = test_model.predict(val_X)

    return mean_squared_error(val_y, predictions_y)



def mse_sgd(iter_test, alpha_given, data_X, data_y, val_X, val_y):

    test_model = SGDRegressor(max_iter = iter_test, alpha = alpha_given)

    test_model.fit(data_X, data_y)

    predictions_y = test_model.predict(val_X)

    return mean_squared_error(val_y, predictions_y)
#Simply testing as many regression models as possible, both for award share and points won

#This code cell is dedicated to predicting POINTS WON, and nothing else. The next cell will deal with award share. 

from sklearn import svm #Support vector machine

from sklearn.linear_model import SGDRegressor #Stochastic Gradient Descent 

from sklearn.linear_model import Ridge #Ridge regression

from sklearn.ensemble import RandomForestRegressor #Random forest, w/ multiple decision trees

from sklearn.neural_network import MLPRegressor #Multi-layer perceptron network

from sklearn.metrics import mean_squared_error #Metric used for selecting best model

from sklearn.model_selection import train_test_split #Utility function for creating training and validation data

from sklearn import preprocessing #Scaling

#Most of these models benefit strongly from scaling, to reduce data input to have a mean of 0 and variance 1

X_scaled = preprocessing.scale(X)

#Data is split here, for training and cross-validation sets

train_X, val_X, train_y, val_y = train_test_split(X_scaled, y, random_state = 0)

train_X2, val_X2, train_y2, val_y2 = train_test_split(X_scaled, y_2, random_state = 1)

#Testing for which combination of characteristics helps produce the lowest validation error

#RANDOM FOREST REGRESSION

estimator_count = [50, 100, 200, 500, 1000]

print("RANDOM FOREST OPTIONS")

for estimator in estimator_count:

    print("Estimator count:", estimator, "| Mean Squared Error:", mse_random_forest(estimator, train_X, train_y, val_X, val_y))

mvp_model = RandomForestRegressor(n_estimators = 500, random_state = 1)#500 was found to be the best estimator count, a tad high but it should serve its purpose

print('-----------------------------------------------------------------------------------------------------------')

#MULTI-LAYER PERCEPTRON

hidden_layer_sizes_test = [50, 100, 150, 200, 250]

alpha = [0.0001, 0.0003, 0.001, 0.003]

print("MULTI LAYER PERCEPTRON OPTIONS")

for layer_size in hidden_layer_sizes_test:

    for alp_test in alpha:

        print("Layer size: ", layer_size, "| Alpha: ", alp_test, "| Mean Squared Error: ", mse_multi_layer_perceptron(layer_size, alp_test, train_X, train_y, val_X, val_y))

mvp_modelMLP = MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = 150, alpha = 0.003, random_state = 1) #Minimum error was found with a layer size of 150 and alpha of 0.03

print('-----------------------------------------------------------------------------------------------------------')

#STOCHASTIC GRADIENT DESCENT

print("STOCHASTIC GRADIENT DESCENT OPTIONS")

max_iters = [500, 1000, 1500, 2000, 2500]

for iter_test in max_iters:

    for alp_test in alpha:

        print("Maximum iterations: ", iter_test, "| Alpha: ", alp_test, "| Mean Squared Error: ", mse_sgd(iter_test, alp_test, train_X, train_y, val_X, val_y))

mvp_modelSGD = SGDRegressor(max_iter = 2500, alpha = 0.0003) #Maximum iterations of 2500, alpha 3 times default

#Ridge Regression

mvp_modelRidge = Ridge()

#Support Vector Regression 

#SVR and Ridge will be left with no options tweaked, simply to keep 2 models default

mvp_modelSVR = svm.SVR()

#Model fitting

mvp_model.fit(train_X, train_y)

mvp_modelMLP.fit(train_X, train_y)

mvp_modelSGD.fit(train_X, train_y)

mvp_modelRidge.fit(train_X, train_y)

mvp_modelSVR.fit(train_X, train_y)
#TO-DO LIST

#Graphing to see whether or not it matches the original trendline

#Analyzing why each player got such a high vote-count

#SHAD Values to assess the main predictors of MVP voting as evaluated by models

#Final predictions for the 18-19 season

test_data['fg3_pct'] = test_data['fg3_pct'].fillna(value = 0)

mvp_preds = mvp_model.predict(preprocessing.scale(test_data[feature_names]))

mvp_predsMLP = mvp_modelMLP.predict(preprocessing.scale(test_data[feature_names]))

mvp_predsSGD = mvp_modelSGD.predict(preprocessing.scale(test_data[feature_names]))

mvp_predsRidge = mvp_modelRidge.predict(preprocessing.scale(test_data[feature_names]))

mvp_predsSVR = mvp_modelSVR.predict(preprocessing.scale(test_data[feature_names]))

test_data['Predicted MVP Voting Random Forest'] = mvp_preds

test_data['Predicted MVP Voting MLP'] = mvp_predsMLP

test_data['Predicted MVP Voting SGD'] = mvp_predsSGD

test_data['Predicted MVP Voting Ridge'] = mvp_predsRidge

test_data['Predicted MVP Voting SVR'] = mvp_predsSVR

print(test_data[['player', 'Predicted MVP Voting Random Forest', 'Predicted MVP Voting MLP', 'Predicted MVP Voting SGD']].sort_values('Predicted MVP Voting Random Forest', ascending = False))

print(test_data[['player', 'Predicted MVP Voting Ridge', 'Predicted MVP Voting SVR']].sort_values('Predicted MVP Voting Ridge', ascending = False))
#AWARD SHARE PREDICTIONS

print("RANDOM FOREST OPTIONS")

for estimator in estimator_count:

    print("Estimator count:", estimator, "| Mean Squared Error:", mse_random_forest(estimator, train_X2, train_y2, val_X2, val_y2))

award_model = RandomForestRegressor(n_estimators = 1000, random_state = 1)#500 was found to be the best estimator count, a tad high but it should serve its purpose

print('-----------------------------------------------------------------------------------------------------------')

#MULTI-LAYER PERCEPTRON

print("MULTI LAYER PERCEPTRON OPTIONS")

for layer_size in hidden_layer_sizes_test:

    for alp_test in alpha:

        print("Layer size: ", layer_size, "| Alpha: ", alp_test, "| Mean Squared Error: ", mse_multi_layer_perceptron(layer_size, alp_test, train_X2, train_y2, val_X2, val_y2))

award_modelMLP = MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = 150, alpha = 0.0001, random_state = 1) #Minimum error was found with a layer size of 150 and alpha of 0.03

print('-----------------------------------------------------------------------------------------------------------')

#STOCHASTIC GRADIENT DESCENT

for iter_test in max_iters:

    for alp_test in alpha:

        print("Maximum iterations: ", iter_test, "| Alpha: ", alp_test, "| Mean Squared Error: ", mse_sgd(iter_test, alp_test, train_X2, train_y2, val_X2, val_y2))

award_modelSGD = SGDRegressor(max_iter = 1000, alpha = 0.003) #Maximum iterations of 2500, alpha 3 times default

#Ridge Regression

award_modelRidge = Ridge()

#Support Vector Regression 

#SVR and Ridge will be left with no options tweaked, simply to keep 2 models default

award_modelSVR = svm.SVR()

#Model fitting

award_model.fit(train_X2, train_y2)

award_modelMLP.fit(train_X2, train_y2)

award_modelSGD.fit(train_X2, train_y2)

award_modelRidge.fit(train_X2, train_y2)

award_modelSVR.fit(train_X2, train_y2)
award_preds = award_model.predict(preprocessing.scale(test_data[feature_names]))

award_predsMLP = award_modelMLP.predict(preprocessing.scale(test_data[feature_names]))

award_predsSGD = award_modelSGD.predict(preprocessing.scale(test_data[feature_names]))

award_predsRidge = award_modelRidge.predict(preprocessing.scale(test_data[feature_names]))

award_predsSVR = award_modelSVR.predict(preprocessing.scale(test_data[feature_names]))

test_data['Predicted Award Share Random Forest'] = award_preds

test_data['Predicted Award Share MLP'] = award_predsMLP

test_data['Predicted Award Share SGD'] = award_predsSGD

test_data['Predicted Award Share Ridge'] = award_predsRidge

test_data['Predicted Award Share SVR'] = award_predsSVR

print(test_data[['player', 'Predicted Award Share Random Forest', 'Predicted Award Share MLP', 'Predicted Award Share SGD']].sort_values('Predicted Award Share Random Forest', ascending = False))

print(test_data[['player', 'Predicted Award Share Ridge', 'Predicted Award Share SVR']].sort_values('Predicted Award Share SVR', ascending = False))