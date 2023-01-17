import numpy as np
from math import tanh
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset to dataframe
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
X['expected_label'] = data.target

# Shuffle rows
X = X.reindex(np.random.permutation(X.index))

X.head(10)
@np.vectorize
def sigmoid(x):
    return tanh(x)

@np.vectorize
def sigmoid_prime(x):
    return 1.0 - x**2


LEAK = 0
@np.vectorize
def relu(x):
    return x if x > 0 else (LEAK * x)

@np.vectorize
def relu_prime(x):
    return 1 if x > 0 else LEAK


activation_map = {
    'sigmoid': {"f": sigmoid, "f '": sigmoid_prime},
    'relu': {"f": relu, "f '": relu_prime},
    }
## Visualize Activation Functions
import matplotlib.pyplot as plt

for activation_type in activation_map:
    X = np.arange(-4, 4, .1)
    
    for key, value in activation_map[activation_type].items():
        Y = [value(v) for v in X]
        
        plt.plot(X, Y, label=key)

    plt.title(activation_type)
    plt.legend()
    plt.show()
class MeanSquaredError:
    """
    Mean squared error -- for regression problems.
    """
    def __call__(seself, real, target):
        return (1 / len(real)) * np.sum((target - real)**2)

    def derivative(self, real, target):
        return 2 / len(real) * (real - target)


class CrossEntropyLoss:
    """
    Cross Entropy loss -- for categorical problems.
    """
    def __call__(self, real, target):
        return -real[np.where(target)] + np.log(np.sum(np.exp(real)))

    def derivative(self, real, target):
        return (1 / len(real)) * (real - target)
class NN:
    def __init__(self, layers, layer_activations, loss_prime, learning_rate=.5):
        self.layers = layers
        self.layer_activations = layer_activations
        self.learning_rate = learning_rate
        self.loss_prime = loss_prime

        assert len(self.layer_activations) == len(self.layers) - 1, "Number activations incorrect."

        self.w = []
        for i, layer in enumerate(self.layers[:-1]):
            # w[in, out]
            matrix = np.random.uniform(-.1, .1, size=(layer, self.layers[i+1]))

            self.w.append(matrix)
def forward(self, x):
    """
    Network estimate y given x.
    """
    fires = [np.copy(x)]

    for i in range(len(self.layers) - 1):
        x = np.matmul(fires[-1], self.w[i])

        fires.append(activation_map[self.layer_activations[i]]['f'](x))

    return fires[-1], fires

NN.forward = forward
def backward(self, real, target, fires):
    """
    Update weights according to directional derivative to minimize error.
    """
    ## Error for output layer
    error = self.loss_prime(fires[-1], target)
    
    delta = activation_map[self.layer_activations[-1]]["f '"](fires[-1]) * error

    deltas = [delta]

    ## Backpropogate error
    for i in range(len(self.layers) - 3, -1, -1):
        error = np.sum(deltas[0] * self.w[i+1], axis=1)

        delta = activation_map[self.layer_activations[i]]["f '"](fires[i+1]) * error

        deltas.insert(0, delta)

    for i in range(len(self.layers) - 2, -1, -1):
        self.w[i] -= self.learning_rate * deltas[i] * fires[i].reshape((-1, 1))

NN.backward = backward
def onehot(value, n_class):
    output = np.zeros(n_class)

    output[value] = 1.

    return output
## Setup Dataset
# ([in1, in2], expected)
data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
data = np.array(data)

# shuffle and split info and labels
idx = np.random.randint(0, 4, size=4000)
X, y = data[idx, 0], data[idx, 1]

## Setup NN
cost = MeanSquaredError()
network = NN([2, 2, 1], ['sigmoid', 'sigmoid'], cost.derivative)

## Train
error = 0
for i, expected in enumerate(y):
    real, fires = network.forward(X[i])

    network.backward(real, expected, fires)

    error += (1 / 2) * (expected - real)**2

    if i % 400 == 399:
        print(error)
        error = 0

## Evaluate
X, y = data[:, 0], data[:, 1]
for i, expected in enumerate(y):
    real, fires = network.forward(X[i])
    print(X[i], '->', real)
## Read Dataset
N_CLASS = 2
EPOCH = 10

import sklearn.datasets
X, y = sklearn.datasets.load_digits(n_class=N_CLASS, return_X_y=True)

## Setup
cost = CrossEntropyLoss()
network = NN([64, N_CLASS], ['sigmoid'], cost.derivative, 10**-3)

## Train
error = 0
for e in range(EPOCH):
    # shuffle dataset between epoch
    idx = [i for i in range(len(y))]
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    for i, expected in enumerate(y):
        real, fires = network.forward(X[i])

        target = onehot(expected, N_CLASS)
        network.backward(real, target, fires)

        error += cost(real, target)

    if not e % 1:
        print(error)
        error = 0

## Evaluate
confusion = {}
accuracy = 0
for i, expected in enumerate(y):
    real, fires = network.forward(X[i])

    guess = np.argmax(real)
    if expected not in confusion:
        confusion[expected] = {}
    if guess not in confusion[expected]:
        confusion[expected][guess] = 0
    confusion[expected][guess] += 1

    accuracy += int(guess == expected)

print(f"Accuracy: {accuracy / len(y)}")
print(confusion)
## Read Dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
dataset = pd.read_csv("../input/digit-recognizer/train.csv")
dataset.columns
X, y = dataset[dataset.columns[1:]].values, dataset['label'].values
## Split dataset into train and test set
split_idx = int(y.size * .8)

train_X, train_y = X[:split_idx], y[:split_idx]
test_X, test_y = X[split_idx:], y[split_idx:]
## Setup NN
N_CLASS = 10
EPOCH = 10

cost = CrossEntropyLoss()

network = NN([784, 32, N_CLASS], ['sigmoid', 'sigmoid'], cost.derivative, 10**-4)

## Train
error = 0
for e in range(EPOCH):
    # shuffle dataset between epoch
    idx = [i for i in range(len(train_y))]
    np.random.shuffle(idx)
    train_X = train_X[idx]
    train_y = train_y[idx]

    for i, expected in enumerate(train_y):
        real, fires = network.forward(train_X[i])

        target = onehot(expected, N_CLASS)
        network.backward(real, target, fires)

        error += cost(real, target)

    if not e % 1:
        print(error)
        error = 0

        
## Evaluate
n_correct = 0
for i, expected in enumerate(test_y):
    real, fires = network.forward(test_X[i])

    n_correct += np.argmax(real) == expected

print(f"Correct: {n_correct / test_y.size:.2f}")
