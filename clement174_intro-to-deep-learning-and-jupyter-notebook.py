import numpy as np

import matplotlib.pyplot as plt
def init_variables():

    # Creates array with two random numbers

    weights = np.random.normal(size = 2)

    bias    = 0

    return weights, bias
def get_dataset():

    row_per_class = 100

    # Creates an array of row_per_class arrays containing 2 random number each

    sick     = np.random.randn(row_per_class, 2) + np.array([1, 1])

    # Creates an array of row_per_class arrays containing 2 random number each

    healthy  = np.random.randn(row_per_class, 2) + np.array([-1, -1])

    # Concatenate the arrays => 1 array containing 2*row_per_class arrays with 2 entries each

    # Features = inputs

    features = np.vstack([sick, healthy])

    # Create an array with row_per_class 0 and row_per_class 1

    # Targets are the expected output for each patient

    targets  = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class) + 1))

    return features, targets
def pre_activation(features, weights, bias):

    # x1*w1 + x2*w2 + bias

    return np.dot(features, weights) + bias
def activation(z):

    return 1 / (1 + np.exp(-z))
def derivative_activation(z):

    return activation(z) * (1 - activation(z))
def predict(features, targets, weights, bias):

    z = pre_activation(features, weights, bias)

    # Compute activation for each value of z array and stock results in array

    y = activation(z)

    # Round values in y array

    return np.round(y)
def cost(predictions, targets):

    return np.mean((predictions - targets) ** 2)
def train(features, targets, weights, bias):

    epochs = 30

    learning_rate = 0.1

    # Print current accuracy

    predictions = predict(features, targets, weights, bias)

    print("Accuracy", np.mean(predictions == targets))

    # Plot points (Display graph)

    plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)

    plt.show()

    for epoch in range(epochs):

        if epoch % 10 == 0:

            predictions = activation(pre_activation(features, weights, bias))

            # Display cost every 10 epochs

            print("Cost: %s" % cost(predictions, targets))

        # Init gradient

        weights_gradients = np.zeros(weights.shape)

        bias_gradient = 0

        # Go through each row and corresponding target

        for feature, target, in zip(features, targets):

            # Compute prediction for each feature / row

            z = pre_activation(feature, weights, bias)

            y = activation(z)

            # Update gradients

            # With derivative partial of cost function ( (y - t)**2 )

            weights_gradients += (y - target) * derivative_activation(z) * feature

            bias_gradient     += (y - target) * derivative_activation(z)

        # Update variables

        weights = weights - learning_rate * weights_gradients

        bias    = bias - learning_rate * bias_gradient

        # Print current acuracy

        predictions = predict(features, targets, weights, bias)

        print("Accuracy", np.mean(predictions == targets))
if __name__ == '__main__':

    features, targets = get_dataset()

    weights, bias     = init_variables()

    # Train the model to find weights and bias

    train(features, targets, weights, bias)