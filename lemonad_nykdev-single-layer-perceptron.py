b = 0.2

w1 = 0.5

w2 = -0.4

x1 = 1.0

x2 = 1.0

y = (b + x1 * w1 + x2 * w2 >= 0) * 2 - 1

print("Sum:", b + x1 * w1 + x2 * w2)

print("Output:", y)
import matplotlib.pyplot as plt

import numpy as np



def plot_hint(b, w1, w2):

    Y = lambda xi: b + xi[0] * w1 + xi[1] * w2 >= 0

    sz = 256

    xx, yy = np.meshgrid(np.linspace(-1, 1, sz), np.linspace(-1, 1, sz))

    X = np.array([xx, yy])

    X = X.reshape((2, sz * sz))

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(X[0], X[1], c=Y(X), marker=",", cmap="Blues", edgecolors="none")

    ax.axhline(y=0, xmin=-1, xmax=1, color="black")

    ax.axvline(x=0, ymin=-1, ymax=1, color="black")

    ax.set_xlabel("x1")

    ax.set_ylabel("x2")

    ax.set_aspect("equal")

    ax.set_title("Decision boundary [dark blue is +1]")



b = 0.0

w1 = 0.5

w2 = -0.4

plot_hint(b, w1, w2)
"""Jonas Nockert, NYKDEV Meetup at LearningWell, June 11, 2019."""

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image



#

# Activation functions

#

class Linear:

    def __init__(self, k=1):

        self.k = k



    def forw(self, x):

        return self.k * x



    def back(self, h):

        return np.ones(h.shape) * self.k



class Threshold:

    def __init__(self, low=0, high=1):

        self.low = low

        self.high = high



    def forw(self, x):

        return (x >= 0) * (self.high - self.low) + self.low



    def back(self, h):

        raise Exception("Threshold cant be used in back propagation")



#

# Loss functions

#

class SquaredLoss:

    def loss(self, activations, targets):

        return 0.5 * np.sum(np.square(activations - targets))



#

# Single-layer Perceptron

#

class SLP:

    def __init__(self):

        # There is no need for a learning rate in the perceptron algorithm. This is

        # because multiplying the update by any constant simply rescales the weights

        # but never changes the sign of the prediction.

        # https://datascience.stackexchange.com/questions/16843/perceptron-learning-rate

        self.learning_rate = 0.05

        self.n_outputs = 1

        self.n_inputs = 2

        self.activation = Threshold(-1, 1)

        self.output_activation = Threshold(-1, 1)

        self.loss_function = SquaredLoss()

        # Initialize weights (including bias) randomly.

        sigma = 0.1

        self.W = np.random.randn(self.n_inputs + 1, self.n_outputs) * sigma - sigma / 2



    def train(self, examples_without_bias, targets, n_epochs=100):

        n_examples = examples_without_bias.shape[1]

        assert targets.shape[0] == 1

        assert n_examples == targets.shape[1]



        # Add bias 1 to examples

        examples = np.ones((self.n_inputs + 1, n_examples))

        examples[1:, :] = examples_without_bias



        epoch_losses = []

        for epoch in range(n_epochs):

            y = np.dot(np.transpose(self.W), examples)

            a = self.activation.forw(y)

            # With threshold activation, error is ±1 if misclassified, otherwise 0.

            error = a - targets



            delta_W = -self.learning_rate * np.dot(error, np.transpose(examples))

            self.W = self.W + np.transpose(delta_W)



            loss = self.loss_function.loss(a, targets)

            epoch_losses.append(loss)

        self.epoch_losses = epoch_losses



    def classify(self, patterns):

        (nin, npat) = np.shape(patterns)

        # Add bias 1 to inputs

        X = np.ones((nin + 1, npat))

        X[1:, :] = patterns



        s = np.dot(np.transpose(self.W), X)

        y = self.output_activation.forw(s)

        return y

    

def image_from_func(f):

    width = 256

    height = 256

    M = np.zeros((height, width))

    for x in range(width):

        for y in range(height):

            x1 = x / width

            x2 = y / height

            M[y, x] = f(x1, x2) * 255

    return Image.fromarray(np.flip(M, 0))



def plot_decision_region(slp, X0, T, ax=None):

    minx, miny = np.min(X0, axis=1)

    maxx, maxy = np.max(X0, axis=1)

    sz = 256

    xx, yy = np.meshgrid(np.linspace(minx, maxx, sz), np.linspace(miny, maxy, sz))

    X = np.array([xx, yy])

    X = X.reshape((2, sz * sz))

    activations = slp.classify(X)



    if not ax:

        fig, ax = plt.subplots()

    ax.scatter(X[0], X[1], c=(-activations[0]), marker=",", cmap="Pastel2", edgecolors="none")

    XA = X0[:, T[0] >= 0]

    XB = X0[:, T[0] < 0]

    ax.scatter(XA[0], XA[1], s=4, cmap="Pastel1", label="Class A")

    ax.scatter(XB[0], XB[1], s=4, cmap="Pastel1", label="Class B")
def func2d(x1, x2):

    m1 = np.random.rand() * 2 + 1  # Between 1 and 3.

    m2 = np.random.rand() * 2 + 1  # Between 1 and 3.

    return (m1 * x1 + m2 * x2) >= 1



np.random.seed(None)  # Change this to an arbitrary integer for repeatability.

im = image_from_func(func2d)

n_training_examples = 5000

X = np.random.rand(2, n_training_examples)

T = np.array([func2d(*X) * 2 - 1])



slp = SLP()

slp.train(X, T, n_epochs=200)



fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122, aspect="equal")

ax1.plot(range(1, len(slp.epoch_losses) + 1), slp.epoch_losses)

plot_decision_region(slp, X, T, ax2)

fig.suptitle("Results from training a 2d perceptron", fontsize=16)

ax1.set_xlabel("Epochs")

ax1.set_ylabel("Error")

ax1.set_title("Training error")

ax2.set_title("Decision boundary")

ax2.legend()

plt.show()