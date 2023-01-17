import pandas as pd

import quandl, math, datetime

import numpy as np

from sklearn import preprocessing, model_selection, svm

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from matplotlib import style

import pickle

from sklearn.datasets import load_digits

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from pandas import DataFrame

from sklearn import linear_model

import tkinter as tk 

import statsmodels.api as sm
df = quandl.get('WIKI/GOOGL')

df.head()
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]



df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100



df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]



df.fillna(-99999, inplace=True)  # Filling outliers inplace of nan

df.head()
forecast_col = 'Adj. Close'

forecast_out = int(math.ceil(0.01*len(df)))  # ciel will round up the decimal value to the nearest whole 

print("Our algorithm will be predicting accuracy true for", forecast_out, "days")



df['label'] = df[forecast_col].shift(-forecast_out)   # Shifting columns negatively. the label column for each row will be Adj. Close price for 10 days into the future



df.dropna(inplace=True)

df.tail()

df.head()
X = np.array(df.drop(['label', 'Adj. Close'], 1))   # Besides label and Adj. Close, everything is a feature in our defined dataset



X = preprocessing.scale(X)   # Scaling is normalising data with other data points. So, in order to properly scale it, you would have to include the training data.



X_lately = X[-forecast_out:]  # predict against

#X = X[:-forecast_out]



df.dropna(inplace=True)



y = np.array(df['label'])

print(len(X), len(y))
X_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
clf = LinearRegression(n_jobs=10) # n_jobs means algo can be threaded. How many jobs are you willing to run at any given time. default for regression is 1. -1 means as many jobs possible by the processor

clf.fit(X_train, y_train)  # train

accuracy = clf.score(x_test, y_test)  # test



accuracy
with open('linearregression.pickle', 'wb') as f:

    pickle.dump(clf, f)

    

pickle_in = open('linearregression.pickle', 'rb')

clf = pickle.load(pickle_in)
clf2 = svm.SVR()

clf2.fit(X_train, y_train)  # train

accuracy2 = clf2.score(x_test, y_test)  # test



accuracy2
clf3 = svm.SVR(kernel='poly')

clf3.fit(X_train, y_train)  # train

accuracy3 = clf3.score(x_test, y_test)  # test



accuracy3
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy * 100, forecast_out)
style.use('ggplot')



df['Forecast'] = np.nan



last_date = df.iloc[-1].name

last_unix = last_date.timestamp()

one_day = 86400

next_unix = last_unix + one_day



for i in forecast_set:

    next_date = datetime.datetime.fromtimestamp(next_unix)

    next_unix += one_day

    df.loc[next_date] = [np.nan for i in range(len(df.columns)-1)] + [i]

    

print(df.tail())



df['Adj. Close'].plot()

df['Forecast'].plot()

plt.legend(loc=4)

plt.xlabel('Date')

plt.ylabel('Price')

plt.show()

digits = load_digits()

digits.keys()
print(digits.data.shape)

x = digits.data    # this is the image of digits stored in matrix form

print(x)
y = digits.target

print(y)
plt.matshow(digits.images[1], cmap = plt.cm.Accent_r)   # cm - color map
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 2)

clf = LinearSVC()

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
clf.score(X_test, y_test)
from statistics import mean

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style
xs = np.array([1,2,3,4,5,6], dtype = np.float64)

ys = np.array([5,4,6,5,6,7], dtype = np.float64)



plt.scatter(xs, ys)
def best_fit_slope_and_intercept(xs, ys):

    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) / 

         ((mean(xs)**2) - mean(xs**2)))

    b = mean(ys) - m*mean(xs)

    return m, b
m, b = best_fit_slope_and_intercept(xs, ys)

print(m, b)
regression_line = [(m*x)+b for x in xs]
plt.scatter(xs, ys)

plt.plot(xs, regression_line)
predict_x = 8

predict_y = (m*predict_x)+b



plt.scatter(xs, ys)

plt.scatter(predict_x, predict_y, color = 'g')

plt.plot(xs, regression_line)
plt.scatter(xs, ys)

plt.plot(xs, regression_line)
def squared_error(ys_orig, ys_line):

    

    return sum((ys_line - ys_orig)**2)





def coefficient_of_determination(ys_orig, ys_line):

    

    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = squared_error(ys_orig, ys_line)

    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    

    return ( 1 - squared_error_regr / squared_error_y_mean)
r_squared = coefficient_of_determination(ys, regression_line)

print(r_squared)
import random



def create_dataset(hm, variance, step = 2, correlation= False): # hm- how many datasets, variance- how variable is the dataset, step- how far on an average to step up the y value per point, correlation- positive or negative

    val = 1

    ys = []

    for i in range(hm):

        y = val + random.randrange(-variance, variance)

        ys.append(y)

        if correlation and correlation == 'pos':

            val += step

        elif correlation and correlation == 'neg':

            val -= step

    

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

xs, ys = create_dataset(40, 40, 2, correlation='pos')
m, b = best_fit_slope_and_intercept(xs, ys)

print(m, b)



regression_line = [(m*x)+b for x in xs]



predict_x = 8

predict_y = (m*predict_x)+b



plt.scatter(xs, ys)

plt.scatter(predict_x, predict_y, color = 'g')

plt.plot(xs, regression_line)

plt.show()



r_squared = coefficient_of_determination(ys, regression_line)

print(r_squared)
xs2, ys2 = create_dataset(40, 10, 2, correlation='pos')
m2, b2 = best_fit_slope_and_intercept(xs2, ys2)

print(m2, b2)



regression_line2 = [(m2*x)+b2 for x in xs2]



predict_x2 = 8

predict_y2 = (m2*predict_x2)+b2



plt.scatter(xs2, ys2)

plt.scatter(predict_x2, predict_y2, color = 'g')

plt.plot(xs2, regression_line2)

plt.show()



r_squared2 = coefficient_of_determination(ys2, regression_line2)

print(r_squared2)
xs3, ys3 = create_dataset(40, 10, 2, correlation=False)
m3, b3 = best_fit_slope_and_intercept(xs3, ys3)

print(m3, b3)



regression_line3 = [(m3*x)+b3 for x in xs3]



predict_x3 = 8

predict_y3 = (m3*predict_x3)+b3



plt.scatter(xs3, ys3)

plt.scatter(predict_x3, predict_y3, color = 'g')

plt.plot(xs3, regression_line3)

plt.show()



r_squared3 = coefficient_of_determination(ys3, regression_line3)

print(r_squared3)
def mean(values):

    return sum(values)/float(len(values))



def variance(values, mean):

    return sum([(x - mean)**2 for x in values])



dataset = [[1,1], [2,3], [4,3], [3,2], [5,5]]



x = [row[0] for row in dataset]

y = [row[1] for row in dataset]



mean_x, mean_y = mean(x), mean(y)

var_x, var_y = variance(x, mean_x), variance(y, mean_y)



print(x)

print(y)



print('x stats: mean = %.3f variance = %.3f' % (mean_x, var_x))

print('x stats: mean = %.3f variance = %.3f' % (mean_y, var_y))
# covariance = sum((x(i) - mean(x)) * (y(i) - mean(y)))



def covariance(x, mean_x, y, mean_y):

    covar = 0.0

    for i in range(len(x)):

        covar += (x[i] - mean_x) * (y[i] - mean_y)

    return covar



covar = covariance(x, mean_x, y, mean_y)

print('Covariance: %.3f' % (covar))
from math import sqrt



def rmse_metric(actual, predicted):

    sum_error = 0.0

    for i in range(len(actual)):

        prediction_error = predicted[i] - actual[i]

        sum_error += (prediction_error ** 2)

    mean_error = sum_error / float(len(actual))

    return sqrt(mean_error)

    
# Calculate coefficient

# B1 = sum((x(i) - mean(x)) * (y(i) - mean(y))) / sum( (x(i) - mean(x))^2)



def coefficient(dataset):

    x = [row[0] for row in dataset]

    y = [row[1] for row in dataset]

    x_mean, y_mean = mean(x), mean(y)

    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)

    b0 = y_mean - b1*x_mean

    return [b0, b1]



b0,b1 = coefficient(dataset)

print('coefficients: b0 = %.3f, b1 = %.3f' % (b0, b1))



# y = b0 + b1*x
def evaluate_algorithm(dataset, algorithm):

    test_set = list()

    for row in dataset:

        row_copy = list(row)

        row_copy[-1] = None

        test_set.append(row_copy)

    predicted = algorithm(dataset, test_set)

    print(predicted)

    actual = [row[-1] for row in dataset]

    rmse = rmse_metric(actual, predicted)

    return rmse, predicted

def simple_linear_regression(train, test):

    prediction = list()

    b0, b1 = coefficient(train)

    for row in test:

        yhat = b0 + b1*row[0]

        prediction.append(yhat)

    return prediction





rmse, pred = evaluate_algorithm(dataset, simple_linear_regression)

print('RMSE: %.3f' % (rmse))
Stock_Market = pd.read_csv('../input/economy.csv',

                names=['Year','Month','Interest_Rate',

                       'Unemployment_Rate',

                       'Stock_Index_Price'])

df = DataFrame(Stock_Market)

df.head()
plt.scatter(df['Interest_Rate'], df['Stock_Index_Price'], color='red')

plt.title('Stock Index Price Vs Interest Rate', fontsize=14)

plt.xlabel('Interest Rate', fontsize=14)

plt.ylabel('Stock Index Price', fontsize=14)

plt.grid(True)

plt.show()

 

plt.scatter(df['Unemployment_Rate'], df['Stock_Index_Price'], color='green')

plt.title('Stock Index Price Vs Unemployment Rate', fontsize=14)

plt.xlabel('Unemployment Rate', fontsize=14)

plt.ylabel('Stock Index Price', fontsize=14)

plt.grid(True)

plt.show()
X = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 input variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets

Y = df['Stock_Index_Price'] # output variable (what we are trying to predict)



# with sklearn

regr = linear_model.LinearRegression()

regr.fit(X, Y)



print('Intercept: \n', regr.intercept_)

print('Coefficients: \n', regr.coef_)





# with statsmodels

X = sm.add_constant(X) # adding a constant

 

model = sm.OLS(Y, X).fit()

predictions = model.predict(X) 

 





# tkinter GUI

root= tk.Tk() 

 

canvas1 = tk.Canvas(root, width = 1200, height = 450)

canvas1.pack()



# with sklearn

Intercept_result = ('Intercept: ', regr.intercept_)

label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')

canvas1.create_window(260, 220, window=label_Intercept)



# with sklearn

Coefficients_result  = ('Coefficients: ', regr.coef_)

label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')

canvas1.create_window(260, 240, window=label_Coefficients)



# with statsmodels

print_model = model.summary()

label_model = tk.Label(root, text=print_model, justify = 'center', relief = 'solid', bg='LightSkyBlue1')

canvas1.create_window(800, 220, window=label_model)





# New_Interest_Rate label and input box

label1 = tk.Label(root, text='Type Interest Rate: ')

canvas1.create_window(100, 100, window=label1)



entry1 = tk.Entry (root) # create 1st entry box

canvas1.create_window(270, 100, window=entry1)



# New_Unemployment_Rate label and input box

label2 = tk.Label(root, text=' Type Unemployment Rate: ')

canvas1.create_window(120, 120, window=label2)



entry2 = tk.Entry (root) # create 2nd entry box

canvas1.create_window(270, 120, window=entry2)





def values(): 

    global New_Interest_Rate #our 1st input variable

    New_Interest_Rate = float(entry1.get()) 

    

    global New_Unemployment_Rate #our 2nd input variable

    New_Unemployment_Rate = float(entry2.get()) 

    

    Prediction_result  = ('Predicted Stock Index Price: ', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))

    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')

    canvas1.create_window(260, 280, window=label_Prediction)

    

button1 = tk.Button (root, text='Predict Stock Index Price',command=values, bg='orange') # button to call the 'values' command above 

canvas1.create_window(270, 150, window=button1)

 



root.mainloop()