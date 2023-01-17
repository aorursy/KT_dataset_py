# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
# Loading the data into data frame

df = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")
df.head()
# Droping Id,Date column

df = df.drop(['id','date'],axis=1)

df.head()
df.tail()
# Descriptive Statistic

df.describe
df.dtypes
# Checking for missing values

df.isnull().sum()
# Independent Feature and Dependent Feature

X = df.drop("price",1)

y = df['price']
# Correlation map 

#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Correlation with output variable

cor_target = abs(cor["price"])
#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.5]

relevant_features
print(df[["bathrooms","sqft_living"]].corr())

print(df[["sqft_living","grade"]].corr())

print(df[["grade","sqft_above"]].corr())

print(df[["sqft_above","sqft_living15"]].corr())
# Backward Elimination

import statsmodels.api as sm
#Adding constant column of ones, mandatory for sm.OLS model

X_1 = sm.add_constant(X)

#Fitting sm.OLS model

model = sm.OLS(y,X_1).fit()

model.pvalues
#Backward Elimination

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

    p= []

    X_1 = X[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols.remove(feature_with_p_max)

    else:

        break

selected_features_BE = cols

print(selected_features_BE)
# Recursive Feature Elimination

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
model = LinearRegression()

#Initializing RFE model

rfe = RFE(model, 7)

#Transforming data using RFE

X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)
#no of features

nof_list=np.arange(1,13)            

high_score=0

#Variable to store the optimum features

nof=0           

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

    model = LinearRegression()

    rfe = RFE(model,nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train,y_train)

    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe,y_train)

    score = model.score(X_test_rfe,y_test)

    score_list.append(score)

    if(score>high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))
cols = list(X.columns)

model = LinearRegression()

#Initializing RFE model

rfe = RFE(model, 10)             

#Transforming data using RFE

X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model

model.fit(X_rfe,y)              

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scale = scaler.fit_transform(X_train)

X_test_scale = scaler.fit_transform(X_test)
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(X_train_scale,y_train)
y_pred = reg.predict(X_test_scale)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

print("MAE: ",mean_absolute_error(y_test,y_pred))

print("MSE: ",mean_squared_error(y_test,y_pred))

print("R Squared: ",r2_score(y_test,y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
ridge_reg = linear_model.Ridge(alpha=0.5)
ridge_reg.fit(X_train_scale,y_train)
y_pred = ridge_reg.predict(X_test_scale)
print("MAE: ",mean_absolute_error(y_test,y_pred))

print("MSE: ",mean_squared_error(y_test,y_pred))

print("R Squared: ",r2_score(y_test,y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
lasso_reg = linear_model.Lasso(alpha=0.1,max_iter=2000)
lasso_reg.fit(X_train_scale,y_train)
y_pred = lasso_reg.predict(X_test_scale)
print("MAE: ",mean_absolute_error(y_test,y_pred))

print("MSE: ",mean_squared_error(y_test,y_pred))

print("R Squared: ",r2_score(y_test,y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
sgdr = linear_model.SGDRegressor()
sgdr.fit(X_train_scale,y_train)
y_pred = sgdr.predict(X_test_scale)
print("MAE: ",mean_absolute_error(y_test,y_pred))

print("MSE: ",mean_squared_error(y_test,y_pred))

print("R Squared: ",r2_score(y_test,y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
from sklearn import svm

svm_reg = svm.SVR()
svm_reg.fit(X_train_scale,y_train)
y_pred = svm_reg.predict(X_test_scale)
print("MAE: ",mean_absolute_error(y_test,y_pred))

print("MSE: ",mean_squared_error(y_test,y_pred))

print("R Squared: ",r2_score(y_test,y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
from sklearn import tree

dtr = tree.DecisionTreeRegressor()
dtr = dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
print("MAE: ",mean_absolute_error(y_test,y_pred))

print("MSE: ",mean_squared_error(y_test,y_pred))

print("R Squared: ",r2_score(y_test,y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=20)
rfr = rfr.fit(X_train,y_train)
y_pred = rfr.predict(X_test)
print("MAE: ",mean_absolute_error(y_test,y_pred))

print("MSE: ",mean_squared_error(y_test,y_pred))

print("R Squared: ",r2_score(y_test,y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=20)
gbr = gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)
print("MAE: ",mean_absolute_error(y_test,y_pred))

print("MSE: ",mean_squared_error(y_test,y_pred))

print("R Squared: ",r2_score(y_test,y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
from sklearn.ensemble import VotingRegressor

r1 = linear_model.LinearRegression()

r2 = GradientBoostingRegressor(n_estimators=20)

r3 = RandomForestRegressor(n_estimators=20)

ereg = VotingRegressor(estimators=[('gb', r1), ('rf', r2), ('lr', r3)])
ereg = ereg.fit(X_train,y_train)
y_pred = ereg.predict(X_test)
print("MAE: ",mean_absolute_error(y_test,y_pred))

print("MSE: ",mean_squared_error(y_test,y_pred))

print("R Squared: ",r2_score(y_test,y_pred))
errors = abs(y_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
print(""" Creating Cross Validation score for best regressor random forest with 10 folds.""")
from sklearn.model_selection import cross_val_score

rfr = RandomForestRegressor(n_estimators=20)

scores = cross_val_score(rfr,X,y,cv=10)
from sklearn.model_selection import cross_val_score

rfr = RandomForestRegressor(n_estimators=20)

scores = cross_val_score(rfr,X,y,cv=10)
scores
scores.mean()
rfr = RandomForestRegressor(n_estimators=20)
from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rfr.get_params())
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rfr = RandomForestRegressor()

# Random search of parameters, using 5 fold cross validation, 

# search across 50 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rfr, param_distributions = random_grid, n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)
rf_random.best_params_
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    print('Model Performance')

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    

    return accuracy
base_model = RandomForestRegressor(n_estimators = 20, random_state = 42)

base_model.fit(X_train, y_train)

base_accuracy = evaluate(base_model, X_test, y_test)
best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))