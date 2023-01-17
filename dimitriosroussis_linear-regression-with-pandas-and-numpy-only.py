# Import the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Read the house data into a data frame

df = pd.read_csv('../input/kc_house_data.csv')
# Display the first five observations

df.head()
# Describe the dataset

df.describe().round(2)
# Drop the id and date columns

df = df.drop(['id', 'date'], axis=1)
# Display the number of data observations

len(df)
# Display the number of starting features

len(df.columns)
# Check the types of data

df.dtypes
# Display the number of null data observations

df.isnull().values.sum()
# Specify target and features

target = df.iloc[:, 0].name

features = df.iloc[:, 1:].columns.tolist()

features
# Correlations of features with target variable

correlations = df.corr()

correlations['price']
# Correlations with target variable

cor_target = abs(correlations['price'])



# Display features with correlation < 0.2

removed_features = cor_target[cor_target < 0.2]

removed_features
# Remove features with correlation < 0.2

df = df.drop(['sqft_lot', 'condition', 'yr_built', 'yr_renovated', 'zipcode', 'long',

              'sqft_lot15'], axis=1)
# Plot Pearson correlation matrix

fig_1 = plt.figure(figsize=(12, 10))

new_correlations = df.corr()

sns.heatmap(new_correlations, annot=True, cmap='Greens', annot_kws={'size': 8})

plt.title('Pearson Correlation Matrix')

plt.show()
# Determine the highest intercorrelations

highly_correlated_features = new_correlations[new_correlations > 0.75]

highly_correlated_features.fillna('-')
# Remove features which are highly correlated with "sqft_living"

df = df.drop(['sqft_above', 'sqft_living15'], axis=1)
# Update features and store their length

features = df.iloc[:, 1:].columns.tolist()

len_of_features = len(features)

len_of_features
# Normalize the features

df.iloc[:, 1:] = (df - df.mean())/df.std()
# Create X, y and theta

X = df.iloc[:, 1:]

ones = np.ones([len(df), 1])

X = np.concatenate((ones, X), axis=1)

y = df.iloc[:, 0:1].values

theta = np.zeros([1, len_of_features + 1])
# Store target

target = y



# Display the size of the matrices

X.shape, y.shape, theta.shape
# Define computecost function

def computecost(X, y, theta):

    H = X @ theta.T

    J = np.power((H - y), 2)

    sum = np.sum(J)/(2 * len(X))

    return sum
# Set iterations and alpha (learning rate)

alpha = 0.01

iterations = 500
# Define gradientdescent function

def gradientdescent(X, y, theta, iterations, alpha):

    cost = np.zeros(iterations)

    for i in range(iterations):

        H = X @ theta.T

        theta = theta - (alpha/len(X)) * np.sum(X * (H - y), axis=0)

        cost[i] = computecost(X, y, theta)

    return theta, cost
# Do Gradient Descent and display final theta

final_theta, cost = gradientdescent(X, y, theta, iterations, alpha)

final_theta.round(2)
# Compute and display final cost

final_cost = computecost(X, y, final_theta)

final_cost.round(2)
# Plot Iterations vs. Cost figure

fig_2, ax = plt.subplots(figsize=(10, 8))

ax.plot(np.arange(iterations), cost, 'r')

ax.set_xlabel('Iterations')

ax.set_ylabel('Cost')

ax.set_title('Iterations vs. Cost')

plt.show()
# Define rmse function

def rmse(target, final_theta):

    predictions = X @ final_theta.T

    return np.sqrt(((predictions[:, 0] - target[:, 0]) ** 2).mean())



# Compute and display Root Mean Squared Error

rmse_val = rmse(target, final_theta)

rmse_val.round(2)
# Display sample prediction for first observation

predictions = X @ final_theta.T

str(predictions[0].round(2))