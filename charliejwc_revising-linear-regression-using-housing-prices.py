# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Image
from IPython.core.display import HTML


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
# import the data into a pandas dataframe
housing_df = pd.read_csv("../input/housingdata/housingdata.csv", header=None)
# inspect the dataframe
housing_df.head()
# add in column names
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_df.columns = housing_colnames
housing_df.info()
# import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# set seaborn style
sns.set_style("dark")
sns.set_style("ticks")
%matplotlib inline
# defining a function to plot each continuous feature against the target variable to see if there are any obvious
# trends in the data. 
def plot_features(col_list, title):
    plt.figure(figsize=(10, 14));
    i = 0
    for col in col_list:
        i += 1
        plt.subplot(6,2,i)
        plt.plot(housing_df[col], housing_df['MEDV'], marker='.', linestyle='none')
        plt.title(title % (col))
        plt.tight_layout()
plot_colnames = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
plot_features(plot_colnames, "Relationship of %s vs Median House Value")
_ = sns.pairplot(housing_df)
plt.show()
corr_matrix = np.corrcoef(housing_df[housing_colnames].values.T)
# to control figure size
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set(font_scale=1.5)
# plot heatmap of correlations
hm = sns.heatmap(corr_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=housing_colnames, 
                 xticklabels=housing_colnames)
plt.show()
plt.plot(housing_df['LSTAT'], housing_df['MEDV'], marker='.', linestyle='none')
plt.xlabel('Percentage lower status of the population')
plt.ylabel('MEDV (1000s)')
plt.title('Median house value (MEDV) vs percentage lower status of the population')
plt.show()
plt.plot(housing_df['RM'], housing_df['MEDV'], marker='.', linestyle='none')
plt.xlabel('Average number of rooms per dwelling')
plt.ylabel('MEDV (1000s)')
plt.title('Median house value (MEDV) vs Average rooms per dwelling')
plt.show()
Image("../input/math-images-for-linear-reg/image001.png")

# isolating target and predictor variables
target = housing_df['MEDV']
predictor = housing_df['RM']
Image("../input/math-images-for-linear-reg/image002.png")

Image("../input/math-images-for-linear-reg/image004.png")
Image("../input/math-images-for-linear-reg/image005.png")
Image("../input/math-images-for-linear-reg/image006.png")
def compute_cost(X, y, theta):
    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))

theta = np.zeros(2)
# adds a stack of ones to vectorize the cost function and make it useful for multiple linear regression 
X = np.column_stack((np.ones(len(predictor)), predictor))
y = target
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost:', cost)
def gradient_descent(X, y, alpha, iterations):
    """function that performs gradient descent"""
    theta = np.zeros(2)
    m = len(y)
    iteration_array = list(range(1, iterations + 1))
    cost_iter = np.zeros(iterations)
    for i in range(iterations):
        t0 = theta[0] - (alpha / m) * np.sum(np.dot(X, theta) - y)
        t1 = theta[1] - (alpha / m) * np.sum((np.dot(X, theta) - y) * X[:,1])
        theta = np.array([t0, t1])
        cost_iter[i] = compute_cost(X, y, theta)
    cost_iter = np.column_stack((iteration_array, cost_iter))
    return theta, cost_iter
iterations = 1000
alpha = 0.01

theta, cost_iter = gradient_descent(X, y, alpha, iterations)
cost = compute_cost(X, y, theta)

print("theta:", theta)
print('cost:', compute_cost(X, y, theta))
plt.plot(cost_iter[:,0], cost_iter[:,1])
plt.xlabel('Gradient descent iteration #')
plt.ylabel('Cost')
plt.show()
plt.scatter(predictor, target, marker='.', color='green')
plt.xlabel('Number of Rooms per Dwelling')
plt.ylabel('Median House Value in 1000s')
samples = np.linspace(min(X[:,1]), max(X[:,1]))
plt.plot(samples, theta[0] + theta[1] * samples, color='black')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

Xs, Ys = np.meshgrid(np.linspace(-35, 0, 50), np.linspace(0, 15, 50))
Zs = np.array([compute_cost(X, y, [t0, t1]) for t0, t1 in zip(np.ravel(Xs), np.ravel(Ys))])
Zs = np.reshape(Zs, Xs.shape)

fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection="3d")
ax.set_xlabel(r'theta 0 ', labelpad=20)
ax.set_ylabel(r'theta 1 ', labelpad=20)
ax.set_zlabel(r'cost  ', labelpad=10)
ax.view_init(elev=25, azim=40)
ax.plot_surface(Xs, Ys, Zs, cmap=cm.rainbow)
ax = plt.figure().gca()
ax.plot(theta[0], theta[1], 'r*')
plt.contour(Xs, Ys, Zs, np.logspace(-3, 3, 15))
ax.set_xlabel(r'theta 0')
ax.set_ylabel(r'theta 1')
# import packages
from sklearn import preprocessing
from sklearn import linear_model

# data
# load data
housing_df = pd.read_csv("../input/housingdata/housingdata.csv", header=None)
# add in column names
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_df.columns = housing_colnames
# select target and feature

X = housing_df[['RM']].values
# MEDV
y = housing_df['MEDV']
# instantiate the linear model
linear = linear_model.LinearRegression()
# fit the model
linear.fit(X,y)
# generate plot
plt.scatter(X, y, c='blue')
plt.plot(X, linear.predict(X), marker='.',color='red')
plt.xlabel("Number of rooms")
plt.ylabel("House value in 1000's")
plt.show()

# print fit info
print ('='* 65)
print ('%30s: %s' % ('Model R-squared', linear.score(X, y)))
print ('%30s: %s' % ('Slope', linear.coef_[0]))
print ('%30s: %s' % ('Model intercept', linear.intercept_))
print ('='* 65)
from sklearn.linear_model import RANSACRegressor

# load data
housing_df = pd.read_csv("../input/housingdata/housingdata.csv", header=None)
# add in column names
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_df.columns = housing_colnames
# select target and feature

X = housing_df[['RM']].values
# MEDV
y = housing_df['MEDV']

# instantiate the linear model
linear = linear_model.LinearRegression()

ransac = RANSACRegressor(linear, max_trials=100,
                        min_samples=50,
                        residual_threshold=5.0, random_state=0)
ransac.fit(X,y)
# plot inliers and outliers determined by RANSAC
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='.',
           label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='green', marker='.',
           label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Rooms per Dwelling')
plt.ylabel('House value in $1000s')
plt.legend(loc='lower right')
plt.show()

# print fit info
print ('='* 65)
print ('%30s: %s' % ('Model R-squared', ransac.estimator_.score(X, y)))
print ('%30s: %s' % ('Slope', ransac.estimator_.coef_[0]))
print ('%30s: %s' % ('Model intercept', ransac.estimator_.intercept_))
print ('='* 65)
from sklearn.cross_validation import train_test_split

X = housing_df[['RM']].values
y = housing_df[['MEDV']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
y_train_pred= lr.predict(X_train)
y_test_pred = lr.predict(X_test)
# plot a residual plot consisting of points where the true target values are subtracted from the predicted responses
_ = plt.scatter(y_train_pred, y_train_pred - y_train, c='red', marker='o', label='Training data')
_ = plt.scatter(y_test_pred, y_test_pred - y_test, c='green', marker='o', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='blue')
plt.xlim([-10, 50])
plt.show()
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('MSE train:', (mean_squared_error(y_train, y_train_pred)))
print('MSE test:', (mean_squared_error(y_test, y_test_pred)))
print('r-sqaured train:', (r2_score(y_train, y_train_pred)))
print('r-squared test:', (r2_score(y_test, y_test_pred)))   
# load data
housing_df = pd.read_csv("../input/housingdata/housingdata.csv", header=None)
# add in column names
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_df.columns = housing_colnames

predictors = housing_df.as_matrix(columns=['RM', 'DIS', 'NOX', 'INDUS'])
target = housing_df['MEDV']
Image("../input/math-images-for-linear-reg/image007.png")
Image("../input/math-images-for-linear-reg/image008.png")
Image("../input/math-images-for-linear-reg/image009.png")
def znormalize_features(X):
    n_features = X.shape[1]
    means = np.array([np.mean(X[:,i]) for i in range(n_features)])
    stddevs = np.array([np.std(X[:,i]) for i in range(n_features)])
    normalized = (X - means) / stddevs

    return normalized

X = znormalize_features(predictors)
X = np.column_stack((np.ones(len(X)), X))
# scale target variable
y = target / 1000
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(predictors[:,0], ax=ax1)
sns.kdeplot(predictors[:,1], ax=ax1)
sns.kdeplot(predictors[:,2], ax=ax1)
sns.kdeplot(predictors[:,3], ax=ax1)
ax2.set_title('After Scaling')
sns.kdeplot(X[:,1], ax=ax2)
sns.kdeplot(X[:,2], ax=ax2)
sns.kdeplot(X[:,3], ax=ax2)
sns.kdeplot(X[:,4], ax=ax2)
plt.show()
def multi_variate_gradient_descent(X, y, theta, alpha, iterations):
    theta = np.zeros(X.shape[1])
    m = len(X)

    for i in range(iterations):
        gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - y)
        theta = theta - alpha * gradient

    return theta
theta = multi_variate_gradient_descent(X, y, theta, alpha, iterations)
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost', cost)
Image("../input/math-images-for-linear-reg/image010.png")
from numpy.linalg import inv

def normal_eq(X, y):
    return inv(X.T.dot(X)).dot(X.T).dot(y)

theta = normal_eq(X, y)
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost:', cost)