import sys

import pandas_datareader as web

import numpy as np

from numpy import array

import datetime

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score
window_size_ary = [1, 5, 8, 10, 14, 17, 20]
seed = 7

np.random.seed(seed)
start = datetime.datetime(2010,1,31)

end = datetime.datetime(2018,12,31)

df = web.DataReader("nvda", 'yahoo', start, end)

df.to_csv('nvda_yahoo_prices_volumes_to_csv_demo.csv')

dataset = df.values

df.head()
plt.style.use('ggplot')

df.plot(subplots=True)
# Method to calculate the mean_square_error for given dataset with respective to given window_size and prior columns

def calculate_mean_square_error(window_size, number_of_prior_columns):

    #X1, y1 = tab2seq(raw_tabular_data, window_size)

    x = dataset[:,:number_of_prior_columns]

    y = dataset[:,number_of_prior_columns]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    model = linear_model.LinearRegression()

    model.fit(x_train, y_train)

    predicted = model.predict(x_test)

    #print('Coefficients: ', model.coef_)

    #print('Varience score: %.2f'  %r2_score(y_test, predicted))

    mean = mean_squared_error(y_test, predicted)

    plt.plot(model.predict(x_train[0:1000:10]), color='blue')

    plt.plot(y_train[0:1000:10], color='red')

    plt.title('Window size: ' + str(window_size) + ", Mean Square_Error: " + str(mean))

    plt.figure()

    return mean
# Method to calculate the best window size for given dataset

def calculate_best_window_size(number_of_prior_columns):

    global_mean = sys.float_info.max 

    mean_square_errors = list()

    best_window = 0 

    for window_size in window_size_ary:

        mean = calculate_mean_square_error(window_size, number_of_prior_columns)

        mean_square_errors.append(mean)

        if mean < global_mean :

            global_mean = mean

            best_window = window_size    

        print(mean_square_errors)

    return best_window


# Calculate the best window size for prior volume column

best_window = calculate_best_window_size(5)

print('\nBest Window Size for prior volume column: ' , best_window)
# Calculate the best window size for prior open column

best_window = calculate_best_window_size(3)

print('\nBest Window Size for prior open column: ' , best_window)