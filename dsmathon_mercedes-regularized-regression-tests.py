# Importing main packages and settings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Import the relevant sklearn packages

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, f_regression

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV, LarsCV, LassoLarsCV

from sklearn.linear_model import MultiTaskElasticNetCV, MultiTaskLassoCV, OrthogonalMatchingPursuitCV

from sklearn.metrics import mean_squared_error
# Function for plotting the scores for different alphas used in Ridge regression

def display_plot(cv_scores, cv_scores_std):

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)

    ax.plot(alpha_space, cv_scores)



    std_error = cv_scores_std / np.sqrt(10)



    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)

    ax.set_ylabel('CV Score +/- Std Error')

    ax.set_xlabel('Alpha')

    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')

    ax.set_xlim([alpha_space[0], alpha_space[-1]])

    ax.set_xscale('log')

    plt.show()
# Loading the training dataset

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# turning object features into dummy variables

df_train_dummies = pd.get_dummies(df_train, drop_first=True)

df_test_dummies = pd.get_dummies(df_test, drop_first=True)



# dropping ID and the target variable

df_train_dummies = df_train_dummies.drop(['ID','y'], axis=1)

df_test_dummies = df_test_dummies.drop('ID', axis=1)



print("Clean Train DataFrame With Dummy Variables: {}".format(df_train_dummies.shape))

print("Clean Test DataFrame With Dummy Variables: {}".format(df_test_dummies.shape))
# concatenate to only include columns in both data sets

# the number should be based on the number of columns. Original is 30471. Now set to 15471 after outlier handling etc.

df_temp = pd.concat([df_train_dummies, df_test_dummies], join='inner')

df_temp_train = df_temp[:len(df_train.index)]

df_temp_test = df_temp[len(df_train.index):]



# check shapes of combined df and split out again

print(df_temp.shape)

print(df_temp_train.shape)

print(df_temp_test.shape)
# defining X and y

X = df_temp_train

test_X = df_temp_test

y = df_train['y']
# Setup the array of alphas and lists to store scores

alpha_space = np.logspace(-4, 0, 20)

ridge_scores = []

ridge_scores_std = []



# Create a ridge regressor: ridge

ridge = Ridge(normalize=True)



# Compute scores over range of alphas

for alpha in alpha_space:



    # Specify the alpha value to use: ridge.alpha

    ridge.alpha = alpha

    

    # Perform 10-fold CV: ridge_cv_scores

    ridge_cv_scores = cross_val_score(ridge, X, y, cv=5)

    

    # Append the mean of ridge_cv_scores to ridge_scores

    ridge_scores.append(np.mean(ridge_cv_scores))

    

    # Append the std of ridge_cv_scores to ridge_scores_std

    ridge_scores_std.append(np.std(ridge_cv_scores))



# Display the plot

display_plot(ridge_scores, ridge_scores_std)
ridgescores = pd.DataFrame({'alpha':alpha_space, 'score':ridge_scores})

ridgescores
# Setup the hyperparameter grid

alpha_space = np.logspace(-4, 0, 20)

param_grid = {'alpha': alpha_space}



# Instantiate a logistic regression classifier: ridge

ridge = Ridge()



# Instantiate the GridSearchCV object: ridge_cv

ridge_cv = GridSearchCV(ridge, param_grid, cv=5)



# Fit it to the data

ridge_cv.fit(X, y)



# Print the tuned parameters and score

print("Tuned Logistic Regression Parameters: {}".format(ridge_cv.best_params_)) 

print("Best score is {}".format(ridge_cv.best_score_))
# instantiating

rcv = RidgeCV()



# setting up steps for the pipeline

steps = [('RidgeCV', rcv)]



# instantiating the pipeline

pipe = Pipeline(steps)



# creating train and test sets using train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# fitting and predicting

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(pipe.score(X_test, y_test)))

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
'''

# Setup the hyperparameter grid

alpha_space = np.logspace(-4, 0, 5)

l1_l2_space = np.linspace(0,1,11)



param_grid = {'alpha': alpha_space,

             'l1_ratio': l1_l2_space}



# Instantiate a logistic regression classifier: elas

elas = ElasticNet()



# Instantiate the GridSearchCV object: elas_cv

elas_cv = GridSearchCV(elas, param_grid, cv=5)



# Fit it to the data

elas_cv.fit(X, y)



# Print the tuned parameters and score

print("Tuned Logistic Regression Parameters: {}".format(elas_cv.best_params_)) 

print("Best score is {}".format(elas_cv.best_score_))

'''
# instantiating different regressors

rcv = RidgeCV()

lcv = LassoCV()

llrcv = LassoLarsCV()

ecv = ElasticNetCV()

ompcv = OrthogonalMatchingPursuitCV()
# bad for but just for now:

import warnings

warnings.filterwarnings("ignore")



# Compute 10-fold cross-validation scores: cv_scores

cv_scores_rcv = cross_val_score(rcv, X, y, cv=10)

cv_scores_lcv = cross_val_score(lcv, X, y, cv=10)

cv_scores_llrcv = cross_val_score(llrcv, X, y, cv=10)

cv_scores_ecv = cross_val_score(ecv, X, y, cv=10)

cv_scores_ompcv = cross_val_score(ompcv, X, y, cv=10)



# Print the 10-fold cross-validation scores

print(cv_scores_rcv)

print(cv_scores_lcv)

print(cv_scores_llrcv)

print(cv_scores_ecv)

print(cv_scores_ompcv)



print("Average 10-Fold RidgeCV CV Score: {}".format(np.mean(cv_scores_rcv)))

print("Average 10-Fold LassoCV CV Score: {}".format(np.mean(cv_scores_lcv)))

print("Average 10-Fold LassoLarsCV CV Score: {}".format(np.mean(cv_scores_llrcv)))

print("Average 10-Fold ElasticNetCV CV Score: {}".format(np.mean(cv_scores_ecv)))

print("Average 10-Fold OrthogonalMatchingPursuitCV CV Score: {}".format(np.mean(cv_scores_ompcv)))