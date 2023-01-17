# data description https://www.kaggle.com/sohier/calcofi#bottle.csv 

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import re

import os

import warnings

import numpy as np

import matplotlib.pyplot as plt



from pandas import DataFrame, read_csv

from scipy.stats.stats import pearsonr 
bottle = read_csv('../input/bottle.csv', usecols=['Depth_ID', 'Depthm', 'T_degC', 'Salnty', 'STheta'])

temp = [(int(i[:2]), int(i[3:5]), int(i[5:7]), i[10:12]) for i in bottle['Depth_ID']]

bottle['Century'], bottle['Year'], bottle['Month'], bottle['CastType'] = list(zip(*temp))

bottle = bottle.drop(columns="Depth_ID") 

bottle.head(5)
bottle = bottle[bottle['CastType']=='HY'][bottle['Century']==19][bottle['Year']==49][bottle['Month']==3]

bottle = bottle.drop(columns='CastType')

bottle = bottle.drop(columns='Century')

bottle = bottle.drop(columns='Year')

bottle = bottle.drop(columns='Month')
parameters = ['T_degC', 'Salnty']

objective = 'Depthm'

bottle.head(5)
x_real = bottle[parameters]

y_real = bottle[objective]
plt.scatter(x_real[parameters[0]], x_real[parameters[1]])

plt.xlabel(parameters[0])

plt.ylabel(parameters[1])
plt.figure()

plt.scatter(x_real[parameters[0]], y_real)

plt.xlabel(parameters[0])

plt.ylabel(objective)



plt.figure()

plt.scatter(x_real[parameters[1]], y_real)

plt.xlabel(parameters[1])

plt.ylabel(objective)
x1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])

y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])

x2 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])

y2 = np.array([9.14, 8.14 ,8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74])

x3 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])

y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])

x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8])

y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89])
plt.figure()

fig, axes = plt.subplots(2, 2)

axes[0, 0].scatter(x1, y1)

axes[0, 1].scatter(x2, y2)

axes[1, 0].scatter(x3, y3)

axes[1, 1].scatter(x4, y4)

for ax in axes.reshape(1, -1)[0]:

    ax.set_xlim(3, 20)

    ax.set_ylim(2, 14)
DataFrame({'mean of x': [np.mean(x1), np.mean(x2), np.mean(x3), np.mean(x4)], 

          'variance of x': [np.std(x1), np.std(x2), np.std(x3), np.std(x4)], 

          'mean of y': [np.mean(y1), np.mean(y2), np.mean(y3), np.mean(y4)], 

          'variance of y': [np.std(y1), np.std(y2), np.std(y3), np.std(y4)], 

          'correlation between x and y': [pearsonr(x1, y1)[0], pearsonr(x2, y2)[0], 

                                          pearsonr(x3, y3)[0], pearsonr(x4, y4)[0]]

          },

         index=[1, 2, 3, 4])
x1 = x1.reshape(-1, 1)

x2 = x2.reshape(-1, 1)

x3 = x3.reshape(-1, 1)

x4 = x4.reshape(-1, 1)
x = x1.copy()

y = y1.copy()

x = DataFrame([[1] * x.shape[0], list(x[:, 0])]).values.T

y = DataFrame(y).values

temp1 = np.dot(x.T, x)

temp2 = np.linalg.inv(temp1)

print(np.dot(temp1, temp2)) # check inverse of a matrix

temp3 = np.dot(temp2, x.T)

print(np.dot(temp3, y))
class linear_regression:

    def __init__(self):

        self.weights = None

        

    def fit(self, x, y):

        x = DataFrame([[1] * x.shape[0], list(x[:, 0])]).values.T

        y = DataFrame(y).values

        temp1 = np.dot(x.T, x)

        try:

            temp2 = np.linalg.inv(temp1)

        except:

            temp1 += np.diag([np.random.uniform() for i in range(temp1.shape[0])])

            temp2 = np.linalg.inv(temp1)

        temp3 = np.dot(temp2, x.T)

        self.weights = np.dot(temp3, y)

        return self

    def predict(self, x):

        x = DataFrame([[1] * x.shape[0], list(x[:, 0])]).values.T

        y = np.dot(x, self.weights)

        return y[:, 0]
mse_error = lambda true, prediction: ((true - prediction)**2).mean()

mae_error = lambda true, prediction: (abs(true - prediction)).mean()

vae_error = lambda true, prediction: (abs(true - prediction)).std()

qae_error = lambda true, prediction: np.percentile(abs(true - prediction), 75) - np.percentile(abs(true - prediction), 25)

r2_error = lambda true, prediction: 1 - ((true - prediction) ** 2).sum() / ((true - true.mean()) ** 2).sum()
model_mine = linear_regression()

model_mine.fit(x1, y1)

model_mine.predict(x1)
def plot_synthetic(model):

    plt.figure()

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].scatter(x1, y1)

    axes[0, 0].plot(x1, model.predict(x1), 'r')

    print(1, 

          ' r2=', r2_error(y1, model.predict(x1)), 

          ', mae=', mae_error(y1, model.predict(x1)))

    axes[0, 1].scatter(x2, y2)

    axes[0, 1].plot(x2, model.predict(x2), 'r')

    print(2, 

          ' r2=', r2_error(y2, model.predict(x2)), 

          ', mae=', mae_error(y2, model.predict(x2)))

    axes[1, 0].scatter(x3, y3)

    axes[1, 0].plot(x3, model.predict(x3), 'r')

    print(3, 

          ' r2=', r2_error(y3, model.predict(x3)), 

          ', mae=', mae_error(y3, model.predict(x3)))

    axes[1, 1].scatter(x4, y4)

    axes[1, 1].plot(x4, model.predict(x4), 'r')

    print(4, 

          ' r2=', r2_error(y4, model.predict(x4)), 

          ', mae=', mae_error(y4, model.predict(x4)))

    for ax in axes.reshape(1, -1)[0]:

        ax.set_xlim(3, 20)

        ax.set_ylim(2, 14)

# plot_synthetic(model_mine)
from sklearn.linear_model import LinearRegression

model_sklearn = LinearRegression(fit_intercept=True)

model_sklearn.fit(x1, y1)

plot_synthetic(model_sklearn)
model_sklearn.coef_
from sklearn.linear_model import LinearRegression

model_sklearn = LinearRegression(fit_intercept=False)

model_sklearn.fit(x1, y1)

plot_synthetic(model_sklearn)
model_sklearn.coef_
from sklearn.preprocessing import StandardScaler



plt.figure()

fig, axes = plt.subplots(2, 2)

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x1_ = StandardScaler().fit_transform(x1)

    y1_ = StandardScaler().fit_transform(y1.reshape(-1, 1))

model_sklearn = LinearRegression(fit_intercept=False)

model_sklearn.fit(x1_, y1_)

axes[0, 0].scatter(x1_, y1_)

axes[0, 0].plot(x1_, model_sklearn.predict(x1_), 'r')

print(1, 

      ' r2=', r2_error(y1_, model_sklearn.predict(x1_)), 

      ', mae=', mae_error(y1_, model_sklearn.predict(x1_)))



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x2_ = StandardScaler().fit_transform(x2)

    y2_ = StandardScaler().fit_transform(y2.reshape(-1, 1))

model_sklearn = LinearRegression(fit_intercept=False)

model_sklearn.fit(x2_, y2_)

axes[0, 1].scatter(x2_, y2_)

axes[0, 1].plot(x2_, model_sklearn.predict(x2_), 'r')

print(2, 

      ' r2=', r2_error(y2_, model_sklearn.predict(x2_)), 

      ', mae=', mae_error(y2_, model_sklearn.predict(x2_)))



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x3_ = StandardScaler().fit_transform(x3)

    y3_ = StandardScaler().fit_transform(y3.reshape(-1, 1))

model_sklearn = LinearRegression(fit_intercept=False)

model_sklearn.fit(x3_, y3_)

axes[1, 0].scatter(x3_, y3_)

axes[1, 0].plot(x3_, model_sklearn.predict(x3_), 'r')

print(3, 

      ' r2=', r2_error(y3_, model_sklearn.predict(x3_)), 

      ', mae=', mae_error(y3_, model_sklearn.predict(x3_)))



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x4_ = StandardScaler().fit_transform(x4)

    y4_ = StandardScaler().fit_transform(y4.reshape(-1, 1))

model_sklearn = LinearRegression(fit_intercept=False)

model_sklearn.fit(x4_, y4_)

axes[1, 1].scatter(x4_, y4_)

axes[1, 1].plot(x4_, model_sklearn.predict(x4_), 'r')

print(4, 

      ' r2=', r2_error(y4_, model_sklearn.predict(x4_)), 

      ', mae=', mae_error(y4_, model_sklearn.predict(x4_)))

for ax in axes.reshape(1, -1)[0]:

    ax.set_xlim((3 - np.mean(x1))/ np.std(x1), (20 - np.mean(x1))/ np.std(x1))

    ax.set_ylim((2 - np.mean(y1))/ np.std(y1), (14 - np.mean(y1))/ np.std(y1))
import statsmodels.api as sm



plt.figure()

fig, axes = plt.subplots(2, 2)

axes[0, 0].scatter(x1, y1)

model_statsmodels = sm.OLS(y1, x1).fit()

predictions = model_statsmodels.predict(x1)

axes[0, 0].plot(x1, predictions, 'r')

print(1, 

      ' r2=', r2_error(y1, predictions), 

      ', mae=', mae_error(y1, predictions))

axes[0, 1].scatter(x2, y2)

model_statsmodels = sm.OLS(y2, x2).fit()

predictions = model_statsmodels.predict(x2)

axes[0, 1].plot(x2, predictions, 'r')

print(2, 

      ' r2=', r2_error(y2, predictions), 

      ', mae=', mae_error(y2, predictions))

axes[1, 0].scatter(x3, y3)

model_statsmodels = sm.OLS(y3, x3).fit()

predictions = model_statsmodels.predict(x3)

axes[1, 0].plot(x3, predictions, 'r')

print(3, 

      ' r2=', r2_error(y3, predictions), 

      ', mae=', mae_error(y3, predictions))

axes[1, 1].scatter(x4, y4)

model_statsmodels = sm.OLS(y4, x4).fit()

predictions = model_statsmodels.predict(x4)

axes[1, 1].plot(x4, predictions, 'r')

print(4, 

      ' r2=', r2_error(y4, predictions), 

      ', mae=', mae_error(y4, predictions))

for ax in axes.reshape(1, -1)[0]:

    ax.set_xlim(3, 20)

    ax.set_ylim(2, 14)
import scipy
plt.figure()

fig, axes = plt.subplots(2, 2)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x1.T[0], y1)

prediction = intercept + slope*x1.T[0]

axes[0, 0].scatter(x1, y1)

axes[0, 0].plot(x1, prediction, 'r')

print(1, 

      ' r2=', r2_error(y1, prediction), 

      ', mae=', mae_error(y1, prediction))



slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x2.T[0], y2)

prediction = intercept + slope*x2.T[0]

axes[0, 1].scatter(x2, y2)

axes[0, 1].plot(x2, prediction, 'r')

print(2, 

      ' r2=', r2_error(y2, prediction), 

      ', mae=', mae_error(y2, prediction))



slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x3.T[0], y3)

prediction = intercept + slope*x3.T[0]

axes[1, 0].scatter(x3, y3)

axes[1, 0].plot(x3, prediction, 'r')

print(3, 

      ' r2=', r2_error(y3, prediction), 

      ', mae=', mae_error(y3, prediction))



slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x4.T[0], y4)

prediction = intercept + slope*x4.T[0]

axes[1, 1].scatter(x4, y4)

axes[1, 1].plot(x4, prediction, 'r')

print(4, 

      ' r2=', r2_error(y4, prediction), 

      ', mae=', mae_error(y4, prediction))

for ax in axes.reshape(1, -1)[0]:

    ax.set_xlim(3, 20)

    ax.set_ylim(2, 14)
import datetime
def data(size):

    xt = np.linspace(0, 1, size)

    yt = xt + [np.random.uniform(0, 0.5) for i in xt]

    return xt, yt

x, y = data(1000)

plt.scatter(x, y)
def linear_regression_time_check(sizes_range):

    time_series = []

    for size in sizes_range:

        x, y = data(size)

        x = DataFrame([[1] * len(x), list(x)]).values.T

        

        now = datetime.datetime.now()

        model_mine = linear_regression()

        model_mine.fit(x, y)

        model_mine.predict(x)



        then = datetime.datetime.now()

        delta = then - now

        time_series.append(delta.microseconds)

    return time_series
def sklearn_time_check(sizes_range):

    time_series = []

    for size in sizes_range:

        x, y = data(size)

        x = DataFrame([[1] * len(x), list(x)]).values.T

        

        now = datetime.datetime.now()

        model_mine = LinearRegression()

        model_mine.fit(x, y)

        model_mine.predict(x)



        then = datetime.datetime.now()

        delta = then - now

        time_series.append(delta.microseconds)

    return time_series
# sizes = [int(i) for i in np.logspace(0, 7, 100)]

# print(sizes)

# linear_time = linear_regression_time_check(sizes)

# sklearn_time = sklearn_time_check(sizes)

# plt.plot(sizes, linear_time, label='simple hand regression')

# plt.plot(sizes, sklearn_time, label='sklearn regression')

# plt.legend()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

model_degree = lambda d: Pipeline([('PF', PolynomialFeatures(d)), ('LR', LinearRegression(fit_intercept=False))])
from sklearn.preprocessing import StandardScaler

from pandas import Series, DataFrame



plt.figure()

fig, axes = plt.subplots(2, 2)

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x1_ = StandardScaler().fit_transform(x1)

    y1_ = StandardScaler().fit_transform(y1.reshape(-1, 1))

model_sklearn = model_degree(2)

model_sklearn.fit(x1_, y1_)

prediction = DataFrame(model_sklearn.predict(x1_), index = x1_[:, 0]).loc[np.sort(list((x1_[:, 0])))]

axes[0, 0].scatter(x1_, y1_)

axes[0, 0].plot(prediction.index, prediction, 'r')

print(1, 

      ' r2=', r2_error(y1_, model_sklearn.predict(x1_)), 

      ', mae=', mae_error(y1_, model_sklearn.predict(x1_)))



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x2_ = StandardScaler().fit_transform(x2)

    y2_ = StandardScaler().fit_transform(y2.reshape(-1, 1))

model_sklearn =  model_degree(2)

model_sklearn.fit(x2_, y2_)

prediction = DataFrame(model_sklearn.predict(x2_), index = x2_[:, 0]).loc[np.sort(list((x2_[:, 0])))]

axes[0, 1].scatter(x2_, y2_)

axes[0, 1].plot(prediction.index, prediction, 'r')

print(2, 

      ' r2=', r2_error(y2_, model_sklearn.predict(x2_)), 

      ', mae=', mae_error(y2_, model_sklearn.predict(x2_)))



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x3_ = StandardScaler().fit_transform(x3)

    y3_ = StandardScaler().fit_transform(y3.reshape(-1, 1))

model_sklearn =  model_degree(2)

model_sklearn.fit(x3_, y3_)

prediction = DataFrame(model_sklearn.predict(x3_), index = x3_[:, 0]).loc[np.sort(list((x3_[:, 0])))]

axes[1, 0].scatter(x3_, y3_)

axes[1, 0].plot(prediction.index, prediction, 'r')

print(3, 

      ' r2=', r2_error(y3_, model_sklearn.predict(x3_)), 

      ', mae=', mae_error(y3_, model_sklearn.predict(x3_)))



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x4_ = StandardScaler().fit_transform(x4)

    y4_ = StandardScaler().fit_transform(y4.reshape(-1, 1))

model_sklearn =  model_degree(2)

model_sklearn.fit(x4_, y4_)

prediction = DataFrame(model_sklearn.predict(x4_), index = x4_[:, 0]).loc[np.sort(list((x4_[:, 0])))]

axes[1, 1].scatter(x4_, y4_)

axes[1, 1].plot(prediction.index, prediction, 'r')

print(4, 

      ' r2=', r2_error(y4_, model_sklearn.predict(x4_)), 

      ', mae=', mae_error(y4_, model_sklearn.predict(x4_)))

for ax in axes.reshape(1, -1)[0]:

    ax.set_xlim((3 - np.mean(x1))/ np.std(x1), (20 - np.mean(x1))/ np.std(x1))

    ax.set_ylim((2 - np.mean(y1))/ np.std(y1), (14 - np.mean(y1))/ np.std(y1))
from sklearn.neighbors import DistanceMetric

# print((max(x1)-min(x1))/N)

def density_coefficients(x, epsilon=2):

    x = x.reshape(-1, 1)

    metric = DistanceMetric.get_metric('euclidean')

    dist_matrix = metric.pairwise(x)

    N_in_epsilon = (dist_matrix < epsilon).sum(axis=0)

    density = 1 - 1 / np.array(N_in_epsilon)

    return density



import matplotlib.colors as colors

d = density_coefficients(y4)

ax = plt.scatter(x4, y4, c=d, norm=colors.Normalize(vmin=0, vmax=1))

plt.colorbar(ax)
class polynomial_weighted_regression:

    def __init__(self, degree):

        self.weights = None

        self.degree = degree

        

    def fit(self, x, y):

        k = density_coefficients(y)

        k = k * 2

        print(k)

        x = DataFrame(np.array([list(x[:, 0]**i) for i in range(self.degree + 1)]).T).values

        y = DataFrame(y).values

        temp1 = np.dot(np.dot(x.T, np.diag(k)), x)

        try:

            temp2 = np.linalg.inv(temp1)

        except:

            temp1 += np.diag([np.random.uniform()] * temp1.shape[0])

            temp2 = np.linalg.inv(temp1)

        temp3 = np.dot(np.dot(temp2,  x.T), np.diag(k))

        self.weights = np.dot(temp3, y)

        return self

    

    def predict(self, x):

        x = DataFrame(np.array([list(x[:, 0]**i) for i in range(self.degree + 1)]).T).values

        y = np.dot(x, self.weights)

        return y[:, 0]
from sklearn.preprocessing import StandardScaler

from pandas import Series, DataFrame

degree = 4



plt.figure(figsize=(20, 20))

fig, axes = plt.subplots(2, 2)

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x1_ = StandardScaler().fit_transform(x1)

    y1_ = StandardScaler().fit_transform(y1.reshape(-1, 1))

axes[0, 0].scatter(x1_, y1_)

model_sklearn = model_degree(degree)

model_sklearn.fit(x1_, y1_)

prediction = DataFrame(model_sklearn.predict(x1_), index = x1_[:, 0]).loc[np.sort(list((x1_[:, 0])))]

axes[0, 0].plot(prediction.index, prediction, 'r')

model_sklearn = polynomial_weighted_regression(degree=degree)

model_sklearn.fit(x1_, y1_)

prediction = DataFrame(model_sklearn.predict(x1_), index = x1_[:, 0]).loc[np.sort(list((x1_[:, 0])))]

axes[0, 0].plot(prediction.index, prediction, 'g')

print(1, 'r2=', r2_error(y1_, prediction.values), ', mae=', mae_error(y1_, prediction.values))



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x2_ = StandardScaler().fit_transform(x2)

    y2_ = StandardScaler().fit_transform(y2.reshape(-1, 1))

axes[0, 1].scatter(x2_, y2_)

model_sklearn = model_degree(degree)

model_sklearn.fit(x2_, y2_)

prediction = DataFrame(model_sklearn.predict(x2_), index = x2_[:, 0]).loc[np.sort(list((x2_[:, 0])))]

axes[0, 1].plot(prediction.index, prediction, 'r')

model_sklearn =  polynomial_weighted_regression(degree=degree)

model_sklearn.fit(x2_, y2_)

prediction = DataFrame(model_sklearn.predict(x2_), index = x2_[:, 0]).loc[np.sort(list((x2_[:, 0])))]

axes[0, 1].plot(prediction.index, prediction, 'g')

print(2, ' r2=', r2_error(y2_, model_sklearn.predict(x2_)), ', mae=', mae_error(y2_, model_sklearn.predict(x2_)))



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x3_ = StandardScaler().fit_transform(x3)

    y3_ = StandardScaler().fit_transform(y3.reshape(-1, 1))

axes[1, 0].scatter(x3_, y3_)

model_sklearn = model_degree(degree)

model_sklearn.fit(x3_, y3_)

prediction = DataFrame(model_sklearn.predict(x3_), index = x3_[:, 0]).loc[np.sort(list((x3_[:, 0])))]

axes[1, 0].plot(prediction.index, prediction, 'r')

model_sklearn =  polynomial_weighted_regression(degree=degree)

model_sklearn.fit(x3_, y3_)

prediction = DataFrame(model_sklearn.predict(x3_), index = x3_[:, 0]).loc[np.sort(list((x3_[:, 0])))]

axes[1, 0].plot(prediction.index, prediction, 'g')

print(3, 

      ' r2=', r2_error(y3_, model_sklearn.predict(x3_)), 

      ', mae=', mae_error(y3_, model_sklearn.predict(x3_)))



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    x4_ = StandardScaler().fit_transform(x4)

    y4_ = StandardScaler().fit_transform(y4.reshape(-1, 1))

axes[1, 1].scatter(x4_, y4_)

model_sklearn = model_degree(degree)

model_sklearn.fit(x4_, y4_)

prediction = DataFrame(model_sklearn.predict(x4_), index = x4_[:, 0]).loc[np.sort(list((x4_[:, 0])))]

axes[1, 1].plot(prediction.index, prediction, 'r')

model_sklearn =  polynomial_weighted_regression(degree=degree)

model_sklearn.fit(x4_, y4_)

prediction = DataFrame(model_sklearn.predict(x4_), index = x4_[:, 0]).loc[np.sort(list((x4_[:, 0])))]

axes[1, 1].plot(prediction.index, prediction, 'g')

print(4, 

      ' r2=', r2_error(y4_, model_sklearn.predict(x4_)), 

      ', mae=', mae_error(y4_, model_sklearn.predict(x4_)))

for ax in axes.reshape(1, -1)[0]:

    ax.set_xlim((3 - np.mean(x1))/ np.std(x1), (20 - np.mean(x1))/ np.std(x1))

    ax.set_ylim((2 - np.mean(y1))/ np.std(y1), (14 - np.mean(y1))/ np.std(y1))
class weighted_polynomial_regression:

    def __init__(self, degree=2):

        self.coef_ = ()

        self.degree = degree

    

    def fit(self, x, y):

        k = density_coefficients(x)

        k = k / sum(k)

        x = np.array([list(x**i) for i in range(self.degree + 1)]).T

        temp = np.dot(np.dot(x.T, np.diag(k)), x)

        temp += np.diag([1e-4 * np.random.random() for i in range(self.degree + 1)])

        temp2 = np.dot(np.linalg.inv(temp), x.T)

        self.coef_ = tuple(np.dot(np.dot(temp2, np.diag(k)), y.reshape(-1, 1)))

        print(self.coef_)

        return self

    

    def predict(self, x):

        return np.array([a * x**i for i, a in enumerate(self.coef_)]).sum(axis=0)

    

    def r2_score(self, y, y_):

        return 1 - ((y  - y_) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    

    def mae(self, y, y_):

        return (abs(y - y_)).mean()
model1_ = weighted_polynomial_regression(degree=10).fit(x3, y3)

pred1_ = model1_.predict(test_x)

plt.scatter(x3, y3)

plt.scatter(test_x, test_y)

plt.plot(test_x, pred1_)

print(model1_.r2_score(test_y, pred1_))

print(model1_.mae(test_y, pred1_))