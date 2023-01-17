import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



print("done importing libraries!")
dataframe = pd.read_csv("../input/train.csv")  # read file into DataFrame

print(dataframe.head())  # preview



sale_price = dataframe['SalePrice']  # holds all values of SalePrice column

sale_price.describe()  # descriptive statistics
correlation_matrix = dataframe.corr()  # correlation coefficients of all paired columns

figure, axis = plt.subplots(figsize=(12, 10))  # make the graph bigger

sns.heatmap(correlation_matrix, vmin=-0.7, vmax=0.7, cmap='RdBu', square=True)
# calculates the mean squared error as an evaluation metric

# predicted: the observed values calculated by the model

# actual: the expected values from the dataset



def calculate_loss(predicted, actual):

  squared_errors = (actual - predicted)**2

  n = len(predicted)

  return 1.0 / (2*n) * squared_errors.sum()
# returns the weighted sum of the inputs as a prediction of y

# x: rows of values for each column (inputs)

# weights: factors representing relative importance of each column



def evaluate_prediction(x, weights):

    return np.dot(x, weights)  # dot: add individual products
# returns the new weights after taking a step in the direction of the negative gradient

# x: rows of inputs

# weights: coefficients for each column in x

# y: rows of actual outputs

# learning_rate: how quickly the model should learn (how big of a step)



def gradient_descent(x, weights, y, learning_rate):

    predictions = evaluate_prediction(x, weights)

    error = predictions - y  # how bad are the current weight values

    gradient = np.dot(x.T,  error) / len(x)  # plug into derivative function

    new_weights = weights - learning_rate * gradient  # update

    return new_weights
# returns the final weights and list of losses after updating weights using repeated gradient descent

# x: rows of inputs

# y: rows of actual outputs

# iterations: number of times to update (epochs)

# learning_rate: how quickly the model should learn



def train_model(x, y, iterations, learning_rate):

    weights = np.zeros(x.shape[1])  # initially all zeros

    loss_history = []

    for i in range(iterations):  # iterations aka epochs

        prediction = evaluate_prediction(x, weights)

        current_loss = calculate_loss(prediction, y)

        loss_history.append(current_loss)

        weights = gradient_descent(x, weights, y, learning_rate)  # update

    return weights, loss_history
area = dataframe['GrLivArea']

x_train, x_test, y_train, y_test = train_test_split(area, sale_price,test_size=0.2)



std_x_train = (x_train - x_train.mean()) / x_train.std()  # calculate z-scores

std_x_train = np.c_[std_x_train, np.ones(x_train.shape[0])]  # concatenate column of 1s

std_x_test = (x_test - x_test.mean()) / x_test.std()

std_x_test = np.c_[std_x_test, np.ones(x_test.shape[0])]



weights, loss_history = train_model(std_x_train, y_train, 1000, 0.01)

print(weights)
plt.plot(loss_history)

plt.title('Loss During Training')

plt.xlabel('Iteration')

plt.ylabel('Loss')

plt.show()
plt.scatter(x_train, y_train, color='red')

plt.plot(x_train, evaluate_prediction(std_x_train, weights), color='blue')

plt.title('Sale Price vs. Living Area (Training set)')

plt.xlabel('Living Area (sq. ft.)')

plt.ylabel('Sale Price ($)')

plt.show()
plt.scatter(x_test, y_test, color='red')

plt.plot(x_test, evaluate_prediction(std_x_test, weights), color='blue')

plt.title('Sale Price vs. Living Area (Test set)')

plt.xlabel('Living Area (sq. ft.)')

plt.ylabel('Sale Price ($)')

plt.show()
multi_vars = dataframe[['GrLivArea', 'OverallQual', 'GarageCars']]



std_multi_vars = (multi_vars - multi_vars.mean()) / multi_vars.std()

std_multi_vars = np.c_[std_multi_vars, np.ones(std_multi_vars.shape[0])]



weights, loss_history = train_model(std_multi_vars, sale_price, 1000, 0.01)

print(weights)
plt.plot(loss_history)

plt.title('Loss During Training')

plt.xlabel('Iteration')

plt.ylabel('Loss')

plt.show()
from sklearn.linear_model import LinearRegression



regressor = LinearRegression()

regressor.fit(std_x_train, y_train)



print(regressor.coef_)

print(regressor.intercept_)



plt.scatter(x_train, y_train, color='red')

plt.plot(x_train, regressor.predict(std_x_train), color='blue')

plt.title('Sale Price vs. Living Area (Training set)')

plt.xlabel('Living Area (sq. ft.)')

plt.ylabel('Sale Price ($)')

plt.show()



plt.scatter(x_test, y_test, color='red')

plt.plot(x_test, regressor.predict(std_x_test), color='blue')

plt.title('Sale Price vs. Living Area (Test set)')

plt.xlabel('Living Area (sq. ft.)')

plt.ylabel('Sale Price ($)')

plt.show()