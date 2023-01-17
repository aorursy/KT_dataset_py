import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import math
df = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')
df.head()
df.info()
df.describe().transpose()
df.shape
#check for null values

df.isnull().sum()
sns.set_style('darkgrid')

sns.lmplot(x = 'YearsExperience', y = 'Salary', data = df, palette = 'coolwarm')
sns.distplot(df['Salary'])
#plot correlation using seaborn heatmap

sns.heatmap(df.corr(), annot = True)
X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
#function for manual calculation of errors

def MSE(y_test, y_pred):

    return np.square(np.subtract(y_test,y_pred)).mean()



def RMSE(mse):

    return math.sqrt(mse)
#classifier plotting function

def clf_plot(clf, X_test, y_test):

    plt.scatter(X_test, y_test,  color='red', marker = 'x')

    plt.plot(X_test, clf.predict(X_test), color='green', linewidth=3)
#function for training and testing model

def fit_model(X, y, split_ratio):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio)

    clf = LinearRegression()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    

    print("Regression Coefficient: ", clf.coef_)

    

    mse_sklearn = mean_squared_error(y_test, y_pred)

    print("Mean Square Error using sklearn: ", mse_sklearn)

    print("Root Mean Square Error using sklearn:", RMSE(mse_sklearn))

    

    mse = MSE(y_test, y_pred)

    print("Mean Square Error: ", mse)

    print("Root Mean Square Error:", RMSE(mse))

    print("R2 Score: ", r2_score(y_test, y_pred))

    

    clf_plot(clf, X_test, y_test)

    

    return None
fit_model(X, y, 0.5)
fit_model(X, y, 0.3)
fit_model(X, y, 0.2)