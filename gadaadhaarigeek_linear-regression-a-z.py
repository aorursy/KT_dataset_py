# What is adjusted R squared ?
import pandas as pd

import numpy as np

data = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")

data.head(2)
pair_df = [data[["Year", "Present_Price", "Kms_Driven", "Owner"]], 

           pd.get_dummies(data[["Fuel_Type", "Seller_Type", "Transmission"]], drop_first=True), data[["Selling_Price"]]]

X = pd.concat(pair_df, axis=1)

y = data[["Selling_Price"]]
# Lets have a look into processed data

X.head()
# Dependent variable

y.head()
# Correlation of features with dependent variables

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(16,8))

corrmat = X.corr()

# picking the top 10 correlated features

cols = corrmat.index

cm = np.corrcoef(X[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Let's delete the Selling_Price from X

X.drop(labels=["Selling_Price"], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
# Shape of the dataset

X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train = X_train.values

y_train = y_train.values

X_test = X_test.values

y_test = y_test.values
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)
# MSE on training data

from sklearn.metrics import mean_squared_error

mean_squared_error(y_true=y_train, y_pred=linreg.predict(X_train))
# MAE on training data 

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_true=y_train, y_pred=linreg.predict(X_train))
# R squared on training data

# This method returns R squared value

# This will first predict the values for X_train since our model is already fit and ten will calculate the R^2 value

linreg.score(X_train, y_train)
# We will get the same value with r2_score function also

from sklearn.metrics import r2_score

r2_score(y_true=y_train, y_pred=linreg.predict(X_train))
# Lets check the metrics on test data

mse = mean_squared_error(y_true=y_test, y_pred=linreg.predict(X_test))

mae = mean_absolute_error(y_true=y_test, y_pred=linreg.predict(X_test))

r2 = linreg.score(X_test, y_test)



print("MSE on test data: ", mse)

print("MAE on test data: ", mae)

print("R squared on test data: ", r2)
np.round(linreg.coef_.ravel(), 3)
X.head(2)
# Bias term in our model 

linreg.intercept_
import matplotlib.pyplot as plt

import seaborn as sns

# For training data

plt.figure(figsize=(20, 10))

plt.plot(range(0, len(y_train)), y_train, label="TrueValues", marker="*", linewidth=3)

plt.plot(range(0, len(y_train)), linreg.predict(X_train), label="PredictedValues", marker="*", linewidth=3)

plt.xlabel("Indices",fontsize=20)

plt.ylabel("Selling Price of Cars",fontsize=20)

plt.title("True Selling Price Vs. Predicted Selling Price",fontsize=20)

plt.show()
# For Test data

plt.figure(figsize=(20, 10))

plt.plot(range(0, len(y_test)), y_test, label="TrueValues", marker="*", linewidth=3)

plt.plot(range(0, len(y_test)), linreg.predict(X_test), label="PredictedValues", marker="o", linewidth=3)

plt.xlabel("Indices",fontsize=20)

plt.ylabel("Selling Price of Cars",fontsize=20)

plt.title("True Selling Price Vs. Predicted Selling Price",fontsize=20)

plt.legend()

plt.show()
def plot_prices(y, y_pred, data_string):

    plt.figure(figsize=(20, 10))

    plt.plot(range(0, len(y)), y, label="TrueValues", marker="*", linewidth=3)

    plt.plot(range(0, len(y)), y_pred, label="PredictedValues", marker="o", linewidth=3)

    plt.xlabel("Indices", fontsize=20)

    plt.ylabel("Selling Price of Cars", fontsize=20)

    plt.title("True Vs. Predicted S.P. - " + data_string, fontsize=20)

    plt.legend()
def apply_model(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    

    # Train data 

    mse = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))

    mae = mean_absolute_error(y_true=y_train, y_pred=model.predict(X_train))

    r2 = model.score(X_train, y_train)



    print("MSE on train data: ", mse)

    print("MAE on train data: ", mae)

    print("R squared on train data: ", r2)  

    

    print()

    print("#"*50)

    print()

    

    # Test data 

    mse = mean_squared_error(y_true=y_test, y_pred=linreg.predict(X_test))

    mae = mean_absolute_error(y_true=y_test, y_pred=linreg.predict(X_test))

    r2 = linreg.score(X_test, y_test)



    print("MSE on test data: ", mse)

    print("MAE on test data: ", mae)

    print("R squared on test data: ", r2)

    

    print()

    print("#"*50)

    print()

    

    plot_prices(y_train, model.predict(X_train), "TRAINING SET")

    

    plot_prices(y_test, model.predict(X_test), "TEST SET")
from sklearn.linear_model import Ridge

ridreg = Ridge()
apply_model(ridreg, X_train, y_train, X_test, y_test)
from sklearn.model_selection import GridSearchCV

params = {"alpha": [.01, .1, .5, .7, 1, 1.5, 2, 2.5, 3, 5, 10]}

ridreg = Ridge()

clf = GridSearchCV(estimator=ridreg, param_grid=params, cv=5, return_train_score=True)

clf.fit(X_train, y_train)
clf.cv_results_
clf.best_estimator_
# With best value of alpha

ridreg = Ridge(alpha=3)

apply_model(ridreg, X_train, y_train, X_test, y_test)
from sklearn.linear_model import Lasso

params = {"alpha": [.00001, .0001, .001, .005, .01, .1, 1, 5]}

lasreg = Lasso()

clf = GridSearchCV(estimator=lasreg, param_grid=params, cv=5, return_train_score=True)

clf.fit(X_train, y_train)
clf.cv_results_
clf.best_estimator_
lasreg = Lasso(alpha=.00001)

apply_model(lasreg, X_train, y_train, X_test, y_test)
np.round(lasreg.coef_, 3)
# Even with the best fitted parameters we don't have much of an improvement. I guess we need contend ourselves here 

# this only.
# Using statsmodels 

import statsmodels.api as sm

res = sm.OLS(endog=y, exog=X).fit()

res.summary()