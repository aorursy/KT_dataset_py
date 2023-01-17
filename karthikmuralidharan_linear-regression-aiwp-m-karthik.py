import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import math
# Reading the dataset

df = pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")
# Displaying the first 5 rows of the dataset

df.head()
# Shape of the dataset

df.shape
# Plotting correlation matrix

corrMatrix = df.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
# Scatter Plot of the Dataset

plt.scatter(df["YearsExperience"], df["Salary"],  color='red')
X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
# Plotting the Classifier

def clf_plot(clf, X_test, y_test):

    plt.scatter(X_test, y_test,  color='purple', marker = 'o')

    plt.plot(X_test, clf.predict(X_test), color='green', linewidth=3)
# Custom functions for calculations

def MSE(a,b):

    return  (((a - b) ** 2).mean())

def RMSE(mse):

    return math.sqrt(mse)
def fitModelFunction(X, y, split_ratio):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio)

    classify = LinearRegression()

    classify.fit(X_train, y_train)

    y_pred = classify.predict(X_test)

    print("Regression Coefficient: ", classify.coef_)

    

    # Using MSE and RMSE functions

    mse = MSE(y_test, y_pred)

    print("Mean squared error using function: %.2f" % mse)

    print("Root mean squared error using function: %.2f" % RMSE(mse))

    print("R2 Score: ", r2_score(y_test, y_pred))

    

    # Cross-checking MSE & RMSE using sklearn metrics

    mse_skl = mean_squared_error(y_test, y_pred)

    print("Mean squared error using sklearn: %.2f" % mse_skl)

    print("Root mean squared error using sklearn: %.2f" % RMSE(mse_skl))

    clf_plot(classify, X_test, y_test)

    

    return None
fitModelFunction(X, y,0.5) # 50% Train and 50% Test
fitModelFunction(X, y,0.3) # For 70% Train and 30% Test
fitModelFunction(X, y,0.2) # For 80% Train and 20% Test