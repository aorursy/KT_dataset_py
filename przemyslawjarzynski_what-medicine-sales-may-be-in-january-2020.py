# Pandas - Data manipulation and analysis library

import pandas as pd

# NumPy - mathematical functions on multi-dimensional arrays and matrices

import numpy as np

# Matplotlib - plotting library to create graphs and charts

import matplotlib.pyplot as plt

# Re - regular expression module for Python

import re

# Calendar - Python functions related to the calendar

import calendar



# Manipulating dates and times for Python

from datetime import datetime



# Scikit-learn algorithms and functions

from sklearn import linear_model

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.ensemble import VotingRegressor



# Settings for Matplotlib graphs and charts

from pylab import rcParams

rcParams['figure.figsize'] = 12, 8



# Display Matplotlib output inline

%matplotlib inline



# Additional configuration

np.set_printoptions(precision=2)
def scatterData(X_train, y_train, X_test, y_test, title):

    plt.title('Prediction using ' + title)

    plt.xlabel('Month sequence', fontsize=20)

    plt.ylabel('Sales', fontsize=20)



    # Use Matplotlib Scatter Plot

    plt.scatter(X_train, y_train, color='blue', label='Training observation points')

    plt.scatter(X_test, y_test, color='cyan', label='Testing observation points')
def predictLinearRegression(X_train, y_train, X_test, y_test):



    y_train = y_train.reshape(-1, 1)

    y_test = y_test.reshape(-1, 1)



    scatterData(X_train, y_train, X_test, y_test, 'Linear Regression')



    reg = linear_model.LinearRegression()

    reg.fit(X_train, y_train)

    plt.plot(X_train, reg.predict(X_train), color='red', label='Linear regressor')

    plt.legend()

    plt.show()



    # LINEAR REGRESSION - Predict/Test model

    y_predict_linear = reg.predict(X_test)



    # LINEAR REGRESSION - Predict for January 2020

    linear_predict = reg.predict([[predictFor]])

    # linear_predict = reg.predict([[predictFor]])[0]



    # LINEAR REGRESSION - Accuracy

    accuracy = reg.score(X_train, y_train)



    # LINEAR REGRESSION - Error

    # error = round(np.mean((y_predict_linear-y_test)**2), 2)

    

    # Results

    print('Linear Regression: ' + str(linear_predict) + ' (Accuracy: ' + str(round(accuracy*100)) + '%)')



    return {'regressor':reg, 'values':linear_predict}
def predictPolynomialRegression(X_train, y_train, X_test, y_test):



    y_train = y_train.reshape(-1, 1)

    y_test = y_test.reshape(-1, 1)



    scatterData(X_train, y_train, X_test, y_test, 'Polynomial Regression')

    

    poly_reg = PolynomialFeatures(degree = 2)

    X_poly = poly_reg.fit_transform(X_train)

    poly_reg_model = linear_model.LinearRegression()

    poly_reg_model.fit(X_poly, y_train)

    plt.plot(X_train, poly_reg_model.predict(poly_reg.fit_transform(X_train)), color='green', label='Polynomial regressor')

    plt.legend()

    plt.show()



    # Polynomial Regression - Predict/Test model

    y_predict_polynomial = poly_reg_model.predict(X_poly)



    # Polynomial Regression - Predict for January 2020

    polynomial_predict = poly_reg_model.predict(poly_reg.fit_transform([[predictFor]]))



    # Polynomial Regression - Accuracy

    # X_poly_test = poly_reg.fit_transform(X_test)

    accuracy = poly_reg_model.score(X_poly, y_train)



    # Polynomial Regression - Error

    # error = round(np.mean((y_predict_polynomial-y_train)**2), 2)



    # Result

    print('Polynomial Regression: ' + str(polynomial_predict) + ' (Accuracy: ' + str(round(accuracy*100)) + '%)')

    return {'regressor':poly_reg_model, 'values':polynomial_predict}
def predictSVR(X_train, y_train, X_test, y_test):



    y_train = y_train.reshape(-1, 1)

    y_test = y_test.reshape(-1, 1)



    scatterData(X_train, y_train, X_test, y_test, 'Simple Vector Regression (SVR)')



    svr_regressor = SVR(kernel='rbf', gamma='auto')

    svr_regressor.fit(X_train, y_train.ravel())



    # plt.scatter(X_train, y_train, color='red', label='Actual observation points')

    plt.plot(X_train, svr_regressor.predict(X_train), label='SVR regressor')

    plt.legend()

    plt.show()



    # Simple Vector Regression (SVR) - Predict/Test model

    y_predict_svr = svr_regressor.predict(X_test)



    # Simple Vector Regression (SVR) - Predict for January 2020

    svr_predict = svr_regressor.predict([[predictFor]])



    # Simple Vector Regression (SVR) - Accuracy

    accuracy = svr_regressor.score(X_train, y_train)



    # Simple Vector Regression (SVR) - Error

    # error = round(np.mean((y_predict_svr-y_train)**2), 2)

    

    # Result

    print('Simple Vector Regression (SVR): ' + str(svr_predict) + ' (Accuracy: ' + str(round(accuracy*100)) + '%)')

    return {'regressor':svr_regressor, 'values':svr_predict}
product = 'N02BA'



# For storing all regression results

regResults = pd.DataFrame(columns=('Linear', 'Polynomial', 'SVR', 'Voting Regressor'), index=[product])



# To display a larger graph than a default with specify some additional parameters for Matplotlib library.

rcParams['figure.figsize'] = 12, 8



# We will be using monthly data for our predictions

df = pd.read_csv("/kaggle/input/pharma-sales-data/salesmonthly.csv")



# We will use monthly sales data from 2014-2019.

df = df.loc[df['datum'].str.contains("2014") | df['datum'].str.contains("2015") | df['datum'].str.contains("2016") | df['datum'].str.contains("2017") | df['datum'].str.contains("2018") | df['datum'].str.contains("2019")]

df = df.reset_index()
df.head()
df['datumNumber'] = 1

for index, row in df.iterrows():

    df.loc[index, 'datumNumber'] = index+1
# The first and the last available month is quite low which may indicate that it might be incomplete

# and skewing results so we're dropping it

df.drop(df.head(1).index,inplace=True)

df.drop(df.tail(1).index,inplace=True)
df = df[df[product] != 0]
df.head()
predictFor = len(df)+5

print('Predictions for the product ' + str(product) + ' sales in January 2020')
regValues = {}
dfSplit = df[['datumNumber', product]]



# We are going to keep 30% of the dataset in test dataset

train, test = train_test_split(dfSplit, test_size=3/10, random_state=0)



trainSorted = train.sort_values('datumNumber', ascending=True)

testSorted = test.sort_values('datumNumber', ascending=True)



X_train = trainSorted[['datumNumber']].values

y_train = trainSorted[product].values

X_test = testSorted[['datumNumber']].values

y_test = testSorted[product].values
# LINEAR REGRESSION

linearResult = predictLinearRegression(X_train, y_train, X_test, y_test)

reg = linearResult['regressor']

regValues['Linear'] = round(linearResult['values'][0][0])
# POLYNOMIAL REGRESSION

polynomialResult = predictPolynomialRegression(X_train, y_train, X_test, y_test)

polynomial_regressor = polynomialResult['regressor']

regValues['Polynomial'] = round(polynomialResult['values'][0][0])
# SIMPLE VECTOR REGRESSION (SVR)

svrResult = predictSVR(X_train, y_train, X_test, y_test)

svr_regressor = svrResult['regressor']

regValues['SVR'] = round(svrResult['values'][0])
vRegressor = VotingRegressor(estimators=[('reg', reg), ('polynomial_regressor', polynomial_regressor), ('svr_regressor', svr_regressor)])



vRegressorRes = vRegressor.fit(X_train, y_train.ravel())



# VotingRegressor - Predict for January 2020

vRegressor_predict = vRegressor.predict([[predictFor]])

regValues['Voting Regressor'] = round(vRegressor_predict[0])

print('Voting Regressor January 2020 predicted value: ' + str(round(vRegressor_predict[0])))

regResults.loc[product] = regValues
regResults