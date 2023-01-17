import numpy as np

import pandas as pd

import math

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import learning_curve

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

%matplotlib inline
regdf = pd.read_csv('../input/regression_table.csv')

regdf.head(10)
print('The shape of the dataframe is - {}'.format(regdf.shape))

print('The number of entries - {}'.format(regdf.size))

print('-'*60)

print('The basic statistics -\n{}'.format(regdf.describe()))

print('-'*60)

print('Unique values per column -\n{}'.format(regdf.nunique()))

print('-'*60)

print('Null values per column -\n{}'.format(regdf.isnull().sum()))

print('-'*60)

print('Data types per column -\n{}'.format(regdf.dtypes))
plt.figure(figsize=(18, 9))

plt.style.use('seaborn')

plt.scatter(regdf.X, regdf.Y)

plt.title('Scatter Plot')

plt.xlabel('X Values')

plt.ylabel('Y Values')

plt.show()
degree = 20

fig, ax = plt.subplots(math.ceil(degree/2),2, figsize=(12, 20))

ax=ax.flatten()           # Converting multidimensional array into a regular "single" dimensional array.

poly_x = []

model = []

for i in range(degree):

                          # Create and store each model into "model" and polynomial features into "poly_x"

    poly_reg = PolynomialFeatures(degree=i+1)

    x = poly_reg.fit_transform(regdf[['X']])

    poly_x.append(x)

    model.append(LinearRegression())

    model[i].fit(x, regdf['Y'])

    # Plot the predicted regression curve on the scatterplot of feature and target variable

    ax[i].scatter(regdf.X, regdf.Y)

    ax[i].plot(regdf['X'], model[i].predict(x), color='r')

    ax[i].set_xlabel('X Values')

    ax[i].set_ylabel('Y Values')

    # Include r2 score in title

    ax[i].set_title('Degree {} polynomial, r2 score = {:.3f}'.format(i+1, r2_score(regdf['Y'], model[i].predict(x))))



fig.tight_layout()        # Automatically adjust subplot parameters to give specified padding
degree = 20

fig, ax = plt.subplots(math.ceil(degree/2),2, figsize=(12, 20), sharey=True)

ax=ax.flatten()

plt.style.use('seaborn')

train_sizes = [150, 300, 450, 600, 800, 950]  # Specify absolute sizes of the training sets for calculating scores

K = 20                                        # Choosing K for K-Fold Cross-Validation

estimator = LinearRegression()

rmseval = []

rmsetrain = []

for i in range(20):

    # Generate "i+1"th degree polynomial features to train model

    poly_reg = PolynomialFeatures(degree=i+1)

    x = poly_reg.fit_transform(regdf[['X']])

    # Get scores on training and validation sets at predetermined training sizes

    train_sizes, train_scores, validation_scores = learning_curve(estimator = estimator, X = x, y = regdf['Y'], cv = K, train_sizes=train_sizes, scoring = 'neg_mean_squared_error')

    # Get the mean training and validation scores of different training or validation sets for each training size

    train_scores_mean = np.sqrt(-train_scores.mean(axis = 1))           # Also MSE is converted to RMSE

    validation_scores_mean = np.sqrt(-validation_scores.mean(axis = 1)) # Also MSE is converted to RMSE

    # Store the mean training and validation scores for the final(6th) training size

    rmseval.append(validation_scores_mean[5])

    rmsetrain.append(train_scores_mean[5])

    # Display error metrics for different regression models

    print('Degree={}'.format(i+1))

    print('train sizes = {}'.format(train_sizes))

    print('train error = {}'.format(train_scores_mean))

    print('validation error = {}'.format(validation_scores_mean))

    print('-'*30)

    # Plot learning curve

    ax[i].plot(train_sizes, train_scores_mean, label = 'Training error')

    ax[i].plot(train_sizes, validation_scores_mean, label = 'Validation error')

    ax[i].set_yscale('log')

    ax[i].set_ylabel('RMSE', fontsize = 14)

    ax[i].set_xlabel('Training set size', fontsize = 14)

    ax[i].legend()

    ax[i].set_title('Learning curve for degree {} polynomial'.format(i+1))



fig.tight_layout()
n = 10

intercept = model[n-1].intercept_

coefficient = model[n-1].coef_[1:]

print('Intercept : {}\nCoefficient : {}'.format(intercept, coefficient))