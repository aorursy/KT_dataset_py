import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#First read in the data and visually check it.

climate_df = pd.read_csv('../input/svalbard-climate-1912-2017.csv', header=0)

print("climate data frame", climate_df)

#Extract X and Y arrays.  X is year, Y is average annual temperature in the MetANN column

X=[]

Y=[]



for i in climate_df.YEAR:

    X.append([1, i])

    X.append

#I am not sure if this is the best way to replace missing data with 999.90 values

#calculate mean of the known data

mean_metANN = np.mean(climate_df.metANN[climate_df.metANN<999])

#fill in missing data with mean of known data

for i in climate_df.metANN:

    if i < 999:

        Y.append(i)

    else:

        Y.append(mean_metANN)
#convert to an np array to plot

X= np.asarray(X)

Y= np.asarray(Y)

#create solve function

def solve_w (X_s, Y_s):

    w_solve = np.linalg.solve(np.dot(X_s.T, X_s), np.dot(X_s.T, Y_s))

    Yhat_solve = np.dot(X_s, w_solve)

    return w_solve, Yhat_solve
#create plot function (probably don't really need this)

#define a function to plot data with a label

def plot_it (X, Y, lab, mark="-", col='blue'):

    plt.plot(X, Y, label = lab, linestyle=mark, color=col)
#create residual mean squared function

# determine how good the model is by computing the r-squared

def calc_r2(X, Y, Yhat):

    d1 = Y - Yhat

    d2 = Y - Y.mean()

    r2 = 1 - d1.dot(d1) / d2.dot(d2)

    return r2
#Ok, solve for Yhat (predicted value) and also calculate least square



w, Yhat = solve_w(X, Y)



#for fun, also try the lin alg least square method, I think this should be similar or the same as Yhat

A = np.vstack([X[:,1], np.ones(len(X))]).T

m, c = np.linalg.lstsq(A, Y)[0]

#now plot everything



fig = plt.figure(figsize=(8,8))

plot_it(X[:,1], Y, "Y", "-", 'blue')

plot_it(X[:,1], Yhat, "Yhat", "dashdot", 'red')

plot_it(X[:,1], X[:,1] * m + c, "Least Square", "--", 'green')

#plt.plot(X[:,1], Y, label="Y")

plt.xlabel("Year")

plt.ylabel("Temperature (C)")

plt.legend()

plt.show()
#Calcuate r-squared

r2 = calc_r2(X, Y, Yhat)

print ("the r-squared is:", r2)
#doesn't look like a very good fit

#Lets look at moving averages to see how well they can fit
#define a function for returning an array of moving averages over a period, n*2 is moving average period) 

# not sure if I got this quite correctly



def moving_average(a, n=3):

    ret = np.cumsum(a, dtype=float)

    ret[n:] = ret[n:] - ret[:-n]

    return ret[n:-n] / n



#now try with moving averages

periods = 5

Y_ma = moving_average(Y, periods)



#solve and plot

#print("shape X", X.shape, "X[periods:-periods]", X[periods:-periods].shape, "Y_ma shape", Y_ma.shape)

w, Yhat_ma = solve_w(X[periods:-periods], Y_ma)





#calculate least square

fig = plt.figure(figsize=(8,8))

plot_it(X[:,1], Y, "Y")

plot_it(X[periods:-periods,1], Y_ma, "Y moving average", "solid", "green")

plot_it(X[periods:-periods,1], Yhat_ma, "Yhat moving average prediction", "dashdot", "red")

plt.xlabel("Year")

plt.ylabel("Temperature (C)")

plt.legend()

plt.show()

r2 = calc_r2(X[periods:-periods,1], Y[periods:-periods], Y_ma)

print ("the r-squared of moving average is:", r2)

r2 = calc_r2(X[periods:-periods,1], Y[periods:-periods], Yhat_ma)

print ("the r-squared of prediction from moving average is:", r2)
#Now try fitting a polynomial

import warnings

warnings.filterwarnings("ignore")



#now with different equations

plt.figure(figsize=(8,8))

plot_it(X[:,1], Y, "Y", "solid", "blue")



r_array = []

dim_array=[]

for dim in range(1, 50, 1):

    z = np.polyfit(X[:,1], Y, dim)

    p = np.poly1d(z)

    #plot Y and predicted Y

    plot_it(X[:,1], p(X[:, 1]), dim, "solid", "green")

    r = calc_r2(X, Y, p(X[:, 1]))

    dim_array.append(dim)

    r_array.append(r)

plt.xlabel("Year")

plt.ylabel("Temperature (C)")

#plt.legend()

plt.show()
#plot r squared of polynomials and calculate minimum number of degrees

plt.xlabel("dim")

plt.ylabel("r sqared")

plot_it(dim_array, r_array, "r2")

plt.legend()

plt.show()

max_dim = r_array.index(max(r_array))

r_max = max(r_array)

print("maximum is dim: ", max_dim, "with r: ", r_max)
#It looks like a polynomial with about 36 dimensions is our best fit using linear algrebra

z = np.polyfit(X[:,1], Y, max_dim)

p = np.poly1d(z)



#plot Y and predicted Y with polynomial

plot_it(X[:,1], Y, "original annual data", 'solid', 'blue')

plot_it(X[:,1], p(X[:, 1]), "poly max dim", "dashed", "green")

plt.legend()

plt.show()

print("r squared for dimension max dim is: ", calc_r2(X, Y, p(X[:, 1])))

print("The equation coefficients are: ", p)
#test with 2017, just for fun

p(2017)
#Now try with support vector regression

from sklearn.svm import SVR

#convert to matrix

x = np.matrix(X[:,1]).T

y = Y

#SVR kernels

# #############################################################################

# Fit regression model

svr_rbf = SVR(kernel='rbf', C=100.0, gamma=0.1)

svr_lin = SVR(kernel='linear', C=100.0)

# this is very slow when I try several degrees svr_poly = SVR(kernel='poly', C=10, degree=1)

#I would like to understand why polyfit seems so much better?? So I use that instead.

y_lin = svr_lin.fit(x, y).predict(x)

y_rbf = svr_rbf.fit(x,y).predict(x)

#y_poly = svr_poly.fit(x, y).predict(x)

#plot the different models



# #############################################################################

# Look at the results

lw = 2

plt.figure(figsize=(10,5))

plt.scatter(X[:,1], y, color='darkorange', label='data')

plt.plot(X[:,1], y_rbf, color='navy', lw=lw, label='RBF model')

plt.plot(X[:,1], y_lin, color='c', lw=lw, label='Linear model')

#plt.plot(X[:,1], y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')

plt.plot(X[:,1], p(X[:, 1]), color='cornflowerblue', lw=lw, label='Polynomial model')

plt.xlabel('data')

plt.ylabel('target')

plt.title('Support Vector Regression')

plt.legend()

plt.show()

print("rsqared:  y_rbf",calc_r2(X[:,1], Y, y_rbf ))

print("rsqared:  y_lin",calc_r2(X[:,1], Y, y_lin ))

#print("rsqared:  y_poly",calc_r2(X[:,1], Y, y_poly ))

print("rsqared:  Polyfit",calc_r2(X[:,1], Y, p(X[:,1])))
#Now some fun.  Make some predictions.  (I now this is not really a valid method)

x_predict = range(2016, 2025)

y_predict = svr_rbf.fit(x,y).predict(np.matrix(x_predict).T)

#now put it together and plot - past and future

fig = plt.figure(figsize=(10,8))

plt.plot(x,y)

plt.plot(x_predict, y_predict, color="blue", linestyle='dashed')

plt.show()