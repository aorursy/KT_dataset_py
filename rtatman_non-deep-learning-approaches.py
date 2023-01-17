import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# set seed for reproducability 
np.random.seed(0)

# read in data
data = pd.read_csv("../input/Admission_Predict.csv")

# for classificiton, add a column with binary accepted or not judgement
# data['admitted'] = np.where(data['Chance of Admit ']>=0.8, '1', '0')

# most methods here will not work with na's. You may want
# to impute instead of dropping.
data = data.dropna()

# clean up column names
data.columns = data.columns.\
    str.strip().\
    str.lower().\
    str.replace(' ', '_')

# split data into training & testing
train, test = train_test_split(data, shuffle=True)

# peek @ dataframe
train.head()
# imports for mixed effect libraries
import statsmodels.api as sm
import statsmodels.formula.api as smf

# model that predicts chance of admission based on GRE & TOEFL score, 
# with university rating as a random effect
md = smf.mixedlm("chance_of_admit ~ gre_score + toefl_score", # formula w/ fixed effects
                 train, # training data
                 groups=train["university_rating"]) # random effects

# fit & summerize model
fitted_model = md.fit()
print(fitted_model.summary())
# predictions & actual values, from test set
predictions = fitted_model.predict(test)
actual = test['chance_of_admit']

# plot actual vs. predicted
plt.scatter(actual, 
            predictions,  
            color='black')

# report mse & r2
print("Mean squared error: %.2f" % mean_squared_error(actual, predictions))
print('Variance score: %.2f' % r2_score(actual, predictions))
import xgboost as xgb

# split training data into inputs & outputs
X = train.drop(["chance_of_admit"], axis=1)
Y = train["chance_of_admit"]

# specify model (xgboost defaults are generally fine)
model = xgb.XGBRegressor()

# fit our model
model.fit(y=Y, X=X)
# split testing data into inputs & output
test_X = test.drop(["chance_of_admit"], axis=1)
test_Y = test["chance_of_admit"]

# predictions & actual values, from test set
predictions = model.predict(test_X)
actual = test_Y

# plot actual vs. predicted
plt.scatter(actual, 
            predictions,  
            color='black')

# report mse & r2
print("Mean squared error: %.2f" % mean_squared_error(actual, predictions))
print('Variance score: %.2f' % r2_score(actual, predictions))
from sklearn.svm import SVR

# split training data into inputs & outputs
X = train.drop(["chance_of_admit"], axis=1)
Y = train["chance_of_admit"]

# specify model (xgboost defaults are generally fine)
model = SVR(gamma='scale', C=1.0, epsilon=0.2)

# fit our model
model.fit(y=Y, X=X)

# split testing data into inputs & output
test_X = test.drop(["chance_of_admit"], axis=1)
test_Y = test["chance_of_admit"]

# predictions & actual values, from test set
predictions = model.predict(test_X)
actual = test_Y

# plot actual vs. predicted
plt.scatter(actual, 
            predictions,  
            color='black')

# report mse & r2
print("Mean squared error: %.2f" % mean_squared_error(actual, predictions))
print('Variance score: %.2f' % r2_score(actual, predictions))