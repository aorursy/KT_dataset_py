#load all needed libraries

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

import pandas as pd

%matplotlib inline
#load data for train and test

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#view some train data

train.head()
#view some test data

test.head()
#shape of train dataset

train.shape
#shape of test dataset

test.shape
#check data type of each column

train.info()
#check missing values with 'isnull' command 

train.isnull().any()
#count, how many missing value is in this attributes

train['y'].isnull().sum()
# Here, I remove Row of with missing value

train = train.dropna()
#Now check there is missing value or not

print(train.isnull().any())



#plot

sb.heatmap(train.isnull())

plt.show()
#describe of data into statical form

train.describe()
#Histograme od train dataset

train.hist()

plt.show()
#Scatter plot

plt.scatter(x=train.x, y=train.y, c='blue')

plt.title('scatter plot')

plt.xlabel('Independent variables')

plt.ylabel('Dependent variables')

plt.show()
#BoxPlot

train.plot(kind='box', subplots=True, layout=(2, 2), figsize=(12, 8))

plt.show()
#Input Variable 

X = train.x.values

#Output Variable

y = train.y.values
#Calculate mean of list numbers with mean function

def mean(numbers):

    return sum(numbers) / float(len(numbers))



#Calculate varience of list numbers with varience function

def varience(numbers, mean):

    return sum([abs(x-mean)**2 for x in numbers])
X_mean, y_mean = mean(X), mean(y)

X_varience = varience(X, X_mean)

y_varience = varience(y, y_mean)
#Calculate the covarience of these groups

def covarience(X, X_mean, y, y_mean):

    ln = len(X)

    cov = 0.0

    for i in range(ln):

        cov += ((X[i] - X_mean) * (y[i] - y_mean))

    return cov
#Lets estimate with coefficient

def coefficients():

    m = covarience(X, X_mean, y, y_mean) / varience(X, X_mean)

    b = y_mean - (m*X_mean)

    return [m,b]
#Let's seprate the test datasets and reshape it

X_test = test['x'].values.reshape(-1, 1)

y_test = test['y'].values.reshape(-1, 1)
# simple_linear_regression() function making here to prediction

def simple_linear_regression():

    prediction = list()

    m, c = coefficients()

    for test in X_test:

        y_pred = m*test[0] + c

        prediction.append(y_pred)

    return prediction
predict = simple_linear_regression()
# Ploting Line

plt.plot(X_test, predict, c='red', label='Regression Line')

# Ploting Scatter Points

plt.scatter(X, y, label='data', c='blue')



plt.xlabel('Independent variable')

plt.ylabel('Dependent variable')

plt.legend()

plt.show()
def root_mean_sqaure_error():

    rmse = 0.0

    m, c = coefficients()

    for i in range(len(X_test)):

        yhat = m*X_test[i] + c

        rmse += (y_test[i] - yhat)**2

    rmse = np.sqrt(rmse/len(X_test))

    return rmse
#Root Mean Sqare Error

RMSE = root_mean_sqaure_error()

print(RMSE[0])
def r_sqaure():

    #sst is the total sum of squares and ssr is the total sum of squares of residuals

    sst = 0

    ssr = 0

    m, c = coefficients()

    for i in range(len(X_test)):

        ypred = m*X_test[i] + c

        ssr += (y_test[i] - ypred)**2

        sst += (y_test[i] - y_mean)**2

    return (1-(ssr/sst))
#R-Sqaure

score = r_sqaure()

print(score[0])