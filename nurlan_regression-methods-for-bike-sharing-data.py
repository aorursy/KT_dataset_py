##############################################################################
# The goal of this script is to compare various regression techniques using
# bike sharing demand dataset.
# Pipelines below are far away from being an optimal pipeline but i will keep
# improving them while i learn more about the nuances of feature engineering,
# optimization, model selection and meta model development.
# comments can be sent to nurlanbek.duishoev @ gmail
##############################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation  import cross_val_score
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Load training and test datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# make a copy of the data
train_orig = train.copy()
test_orig = test.copy()

# check variables
train.head()
test.head()

# drop columns from train set that do not exist in test set
train = train.drop(labels=["casual", "registered"], axis=1)

##################################################################
################## Feature Engineering

# year, month, day-of-the-week, time-of-the-day might be useful parameters for 
# prediction
train['year'] = train['datetime'].str.extract("^(.{4})")
test['year'] = test['datetime'].str.extract("^(.{4})")

train['month'] = train['datetime'].str.extract("-(.{2})-")
test['month'] = test['datetime'].str.extract("-(.{2})-")

train['day'] = train['datetime'].str.extract("(.{2}) ")
test['day'] = test['datetime'].str.extract("(.{2}) ")

train['time'] = train['datetime'].str.extract(" (.{2})")
test['time'] = test['datetime'].str.extract(" (.{2})")

# convert string to int
train[['year', 'month', 'day', 'time']] = train[['year', 'month', 'day', 'time']].astype(int)
test[['year', 'month', 'day', 'time']] = test[['year', 'month', 'day', 'time']].astype(int)

train['dayOfWeek'] = train.apply(lambda x: datetime.date(x['year'], x['month'], x['day']).weekday(), axis=1)
test['dayOfWeek'] = test.apply(lambda x: datetime.date(x['year'], x['month'], x['day']).weekday(), axis=1)

# drop datetime column since its unique for every row
train = train.drop(labels=["datetime"], axis=1)
test = test.drop(labels=["datetime"], axis=1)

# convert ordinal categorical variables into multiple dummy variables
# get dummy variables for season
train['season'].value_counts() # this will give frequency table, similar to table() in R
train = train.join(pd.get_dummies(train.season, prefix='season'))
test = test.join(pd.get_dummies(test.season, prefix='season'))

# drop season variable
train = train.drop(labels=["season"], axis=1)
test = test.drop(labels=["season"], axis=1)

# get dummy variables for weather
train['weather'].value_counts() # this will give frequency table, similar to table() in R
train = train.join(pd.get_dummies(train.weather, prefix='weather'))
test = test.join(pd.get_dummies(test.weather, prefix='weather'))

# drop weather variable
train= train.drop(labels=["weather"], axis=1)
test= test.drop(labels=["weather"], axis=1)

train.corr()
plt.matshow(train.corr())

# drop highly correlated predictors, read here for more information about
# consequences of collinear predictors: https://en.wikipedia.org/wiki/Multicollinearity
train= train.drop(labels=["atemp"], axis=1)
test= test.drop(labels=["atemp"], axis=1)

# remove first response variable from the training set
target = train['count'].values
train = train.drop(labels=["count"], axis=1)


##################################################################
################## Predictions in log space
# Regression algorithms can predict negative values. But in this competition
# it's not allowed to predict negative response. So we will convert the
# response into log scale, predict in log scale, and convert final
# predictions back to linear space again. 
target = np.log(target)


############################################################
#################### Define Helper Functions
############################################################
def bs_create_polynomial_terms(l_train, l_test, degree):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    # details http://scikit-learn.org/stable/modules/linear_model.html
    l_train_poly = poly.fit_transform(l_train) 
    l_test_poly = poly.fit_transform(l_test)
    return l_train_poly, l_test_poly
    
def bs_scale_mean_std(l_train, l_test):
    # read about data scaling here: 
    # http://quant.stackexchange.com/questions/4434/gradient-tree-boosting-do-input-attributes-need-to-be-scaled
    sc = StandardScaler()
    l_train_scaled = pd.DataFrame(sc.fit_transform(l_train))
    l_test_scaled = pd.DataFrame(sc.transform(l_test)) # careful, transform() only.
    return l_train_scaled, l_test_scaled

def bs_fit_and_save(clf, l_train, l_target, l_test, filename):
    # more about it here: http://scikit-learn.org/stable/modules/svm.html#regression
    clf.fit (l_train, l_target)
    
    # The mean square error
    predict_train = clf.predict(l_train)
    print("Residual sum of squares: %.2f" % np.mean((predict_train - l_target) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % clf.score(l_train, l_target))
    
    # Plot outputs
    plt.plot(l_target, "ro", color="red")
    plt.plot(predict_train, "ro", color="blue")
    plt.show()
        
    # convert prediction from log scale to linear space
    predict_test = clf.predict(l_test)
    predict_test = np.exp(predict_test)
    
    output = test_orig['datetime']
    output = pd.DataFrame(output)
    predict = pd.DataFrame(predict_test)
    output = output.join(predict)
    output.columns = ['datetime', 'count']
    
    output.to_csv("predictions/" + filename + ".csv", index=False)
    return clf

############################################################
###### Model Development
############################################################


############################################################################
################## Simple Linear Regression

# Since its a regression problem we will first develop simple linear regression

# Create linear regression object
clf = linear_model.LinearRegression()

# get fitted regresser
clf = bs_fit_and_save(clf, train, target, test, "output_SLR")


############################################################################
################## Simple Linear Regression with Ridge Regression

# Now we will perform Ridge Regression
# Unlike simple linear regression, ridge regularization requires scaled data
train_scaled, test_scaled = bs_scale_mean_std(train, test)

# first we will perform cross-validation to find the best alpha value
clf = linear_model.RidgeCV(alphas=[0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0])

clf = bs_fit_and_save(clf, train_scaled, target, test_scaled, "output_Ridge")

# The coefficients
print('Coefficients: \n', clf.coef_)
print('Alpha: \n', clf.alpha_)  

############################################################################
################## Simple Linear Regression with Lasso Regression

# Unlike simple linear regression, lasso regularization requires scaled data
train_scaled, test_scaled = bs_scale_mean_std(train, test)

# first we will perform cross-validation to find the best alpha value
clf = linear_model.LassoCV(alphas=[0.001, 0.1, 1.0, 10.0, 100.0, 1000.0])

# get fitted regresser
clf = bs_fit_and_save(clf, train_scaled, target, test_scaled, "output_Lasso")

# The coefficients
print('Alpha: \n', clf.alpha_)  

############################################################################
################## Simple Linear Regression with Polynomial terms

# create polynomial terms
train_poly, test_poly = bs_create_polynomial_terms(train, test, 2)

# Create linear regression object
clf = linear_model.LinearRegression()

# get fitted regresser
clf = bs_fit_and_save(clf, train_poly, target, test_poly, "output_poly_d2")

############################################################################
################## Ridge Regression with Polynomial Terms

# create polynomial terms
train_poly, test_poly = bs_create_polynomial_terms(train, test, 2)

# Ridge regularization requires scaled data
train_scaled, test_scaled = bs_scale_mean_std(train_poly, test_poly)

# Create linear regression object
clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])

# get fitted regresser
clf = bs_fit_and_save(clf, train_scaled, target, test_scaled, "output_Ridge_poly_2")

############################################################################
################## Lasso Regression with Polynomial Terms

# create polynomial terms
train_poly, test_poly = bs_create_polynomial_terms(train, test, 2)

# Lasso regularization requires scaled data
train_scaled, test_scaled = bs_scale_mean_std(train_poly, test_poly)

# Create linear regression object
clf = linear_model.LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

# get fitted regresser
clf = bs_fit_and_save(clf, train_scaled, target, test_scaled, "output_lasso_poly_2")


############################################################################
################## Support Vector Regression with Polynomial Terms

# create polynomial terms
train_poly, test_poly = bs_create_polynomial_terms(train, test, 2)

# SVR requires scaled data
train_scaled, test_scaled = bs_scale_mean_std(train_poly, test_poly)

# Create linear regression object
# ideally these parameters should be determined using cross-validation
clf = svm.SVR(kernel='rbf', C=100, gamma=0.1) 

# get fitted regresser
clf = bs_fit_and_save(clf, train_scaled, target, test_scaled, "output_svm_poly_2")

############################################################################
################## Gradient Boosting Regression with Polynomial Terms

# create polynomial terms
train_poly, test_poly = bs_create_polynomial_terms(train, test, 2)

# Unlike simple linear regression, ridge regularization requires scaled data
train_scaled, test_scaled = bs_scale_mean_std(train_poly, test_poly)

# Create linear regression object
clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, 
                                max_depth=3, loss='ls')

# get fitted regresser
clf = bs_fit_and_save(clf, train_scaled, target, test_scaled, "output_gbm_poly_2")


############################################################################
################## GBR with parameter estimation with cross-validation

# create polynomial terms
train_poly, test_poly = bs_create_polynomial_terms(train, test, 3)

# Unlike simple linear regression, ridge regularization requires scaled data
train_scaled, test_scaled = bs_scale_mean_std(train_poly, test_poly)

# Create linear regression object
t_clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, 
                                max_depth=3, loss='ls')
                                
from sklearn.grid_search import GridSearchCV
param_range_n_estimators = [500, 1000, 2000]
param_range_max_depth = [1, 3, 5]

param_grid = [{'n_estimators': param_range_n_estimators,
              'max_depth': param_range_max_depth}]

# we will not define optional 'scoring' parameter. It will use lsr for scoring
# read here for detail of scoring for classification and regression grid search
# http://scikit-learn.org/stable/modules/grid_search.html
# https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/metrics/regression.py#L370
clf = GridSearchCV(estimator=t_clf, param_grid=param_grid, cv=5, n_jobs=4)
                 
# get fitted regresser
clf = bs_fit_and_save(clf, train_scaled, target, test_scaled, "output_gbm_cv_poly_3") # this will take some time

########################################################################
################## Create meta-models out of multiple single models
## in progress...













