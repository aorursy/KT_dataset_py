import matplotlib.pyplot as plt

import numpy as np



#

# Activation functions

#

class LeakyReLU:

    def __init__(self, alpha=0.01):

        self.alpha = alpha



    def forw(self, x):

        return np.maximum(self.alpha * x, x)



    def back(self, h):

        return (h >= 0) * self.alpha



class Linear:

    def __init__(self, k=1):

        self.k = k



    def forw(self, x):

        return self.k * x



    def back(self, h):

        return np.ones(h.shape) * self.k



class Sigmoid:

    def forw(self, x):

        return np.divide(1, np.add(1, np.exp(-x)))



    def back(self, h):

        return np.multiply(h, 1 - h)



class SymmetricSigmoid:

    def forw(self, x):

        return np.divide(2, np.add(1, np.exp(-x))) - 1



    def back(self, h):

        return np.multiply(1 + h, 1 - h) * 0.5



class Tanh:

    def forw(self, x):

        return np.divide(np.exp(x) - np.exp(-x), np.exp(x) + np.exp(-x))



    def back(self, h):

        return 1 - np.square(h)



class Threshold:

    def __init__(self, low=0, high=1):

        self.low = low

        self.high = high



    def forw(self, x):

        return (x >= 0) * (self.high - self.low) + self.low



    def back(self, h):

        raise Exception("Threshold cant be used in back propagation")



fig = plt.figure(figsize=(15,15))

X = np.linspace(-10, 10, 100)



f = Linear()

ax = fig.add_subplot(331)

ax.plot(X, f.forw(X))

ax.set_title("Linear")



f = Threshold(-1, 1)

ax = fig.add_subplot(332)

ax.plot(X, f.forw(X))

ax.set_title("Threshold")



f = Sigmoid()

ax = fig.add_subplot(334)

ax.plot(X, f.forw(X))

ax.set_title("Sigmoid")



f = SymmetricSigmoid()

ax = fig.add_subplot(335)

ax.plot(X, f.forw(X))

ax.set_title("Symmetric Sigmoid")



f = Tanh()

ax = fig.add_subplot(336)

ax.plot(X, f.forw(X))

ax.set_title("Tanh")



f = LeakyReLU(0)

ax = fig.add_subplot(337)

ax.plot(X, f.forw(X))

ax.set_title("ReLU")



f = LeakyReLU(0.02)

ax = fig.add_subplot(338)

ax.plot(X, f.forw(X))

ax.set_title("Leaky ReLU")



plt.show()
"""Jonas Nockert, NYKDEV Meetup at LearningWell, June 11, 2019."""

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image



#

# Loss functions

#

class SquaredLoss:

    def loss(self, activations, targets):

        return 0.5 * np.sum(np.square(activations - targets))



class CrossEntropyLoss:

    def loss(self, activations, targets):

        return -np.sum(

            targets * np.log(activations) + (1 - targets) * np.log(1 - activations)

        )



#

# Multi-layer Perceptron

#

class MLP:

    def __init__(

        self,

        n_inputs,

        n_hidden,

        activation,

        output_activation,

        learning_rate=0.001,

        momentum=0.9,

    ):

        self.learning_rate = learning_rate

        self.momentum = momentum

        self.n_inputs = n_inputs

        self.n_hidden = n_hidden

        self.n_outputs = 1

        self.activation = activation

        self.output_activation = output_activation

        self.loss_function = SquaredLoss()

        # Initialize weights including bias.

        sigma = 0.1

        self.W = np.random.randn(self.n_hidden, self.n_inputs + 1) * sigma - sigma / 2

        self.V = np.random.randn(self.n_outputs, self.n_hidden + 1) * sigma - sigma / 2

        self.dW = np.zeros((self.n_hidden, self.n_inputs + 1))

        self.dV = np.zeros((self.n_outputs, self.n_hidden + 1))



    def forward(self, examples_without_bias):

        n_examples = examples_without_bias.shape[1]

        # Add bias 1 to examples

        examples = np.ones((self.n_inputs + 1, n_examples))

        examples[1:, :] = examples_without_bias



        hin = np.dot(self.W, examples)

        hout_without_bias = self.activation.forw(hin)

        self.hout_without_bias = hout_without_bias



        hout = np.ones((self.n_hidden + 1, n_examples))

        hout[1:, :] = hout_without_bias

        oin = np.dot(self.V, hout)

        oout_without_bias = self.activation.forw(oin)

        self.oout_without_bias = oout_without_bias

        return oout_without_bias



    def backward(self, targets):

        n_examples = self.oout_without_bias.shape[1]

        delta_o = (self.oout_without_bias - targets) * self.activation.back(

            self.oout_without_bias

        )

        self.delta_o = delta_o



        fip = np.ones((self.n_hidden + 1, n_examples))

        fip[1:, :] = self.activation.back(self.hout_without_bias)

        delta_h = np.dot(np.transpose(self.V), delta_o) * fip

        self.delta_h = delta_h[1:, :]  # Remove bias.



    def update_weights(self, examples_without_bias):

        n_examples = examples_without_bias.shape[1]

        # Add bias 1 to examples

        examples = np.ones((self.n_inputs + 1, n_examples))

        examples[1:, :] = examples_without_bias



        self.dW = (self.dW * self.momentum) - (

            np.dot(self.delta_h, np.transpose(examples))

        ) * (1 - self.momentum)

        hout = np.ones((self.n_hidden + 1, n_examples))

        hout[1:, :] = self.hout_without_bias

        self.dV = self.dV * self.momentum - (

            np.dot(self.delta_o, np.transpose(hout))

        ) * (1 - self.momentum)

        self.W = self.W + self.dW * self.learning_rate

        self.V = self.V + self.dV * self.learning_rate



    def train(self, examples_without_bias, targets, n_epochs=100):

        n_examples = examples_without_bias.shape[1]

        assert targets.shape[0] == 1

        assert n_examples == targets.shape[1]



        # Add bias 1 to examples

        examples = np.ones((self.n_inputs + 1, n_examples))

        examples[1:, :] = examples_without_bias



        epoch_losses = []

        for epoch in range(n_epochs):

            y = self.forward(examples_without_bias)

            self.backward(targets)

            self.update_weights(examples_without_bias)



            loss = self.loss_function.loss(y, targets)

            epoch_losses.append(loss)

        self.epoch_losses = epoch_losses



    def classify(self, patterns):

        y = self.output_activation.forw(self.forward(patterns))

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



def plot_decision_region(mlp, X0, T, ax=None):

    minx, miny = np.min(X0, axis=1)

    maxx, maxy = np.max(X0, axis=1)

    sz = 256

    xx, yy = np.meshgrid(np.linspace(minx, maxx, sz), np.linspace(miny, maxy, sz))

    X = np.array([xx, yy])

    X = X.reshape((2, sz * sz))

    activations = mlp.classify(X)



    if not ax:

        fig, ax = plt.subplots()

    ax.scatter(X[0], X[1], c=(-activations[0]), marker=",", cmap="Pastel2", edgecolors="none")

    XA = X0[:, T[0] >= 0]

    XB = X0[:, T[0] < 0]

    ax.scatter(XA[0], XA[1], s=4, cmap="Pastel1", label="Class A")

    ax.scatter(XB[0], XB[1], s=4, cmap="Pastel1", label="Class B")
ndata = 100

mA = np.array([1.0, 0.3])

sigmaA = 0.2

mB = np.array([[0.0, -0.1]])

sigmaB = 0.2

cluster1 = np.zeros((2, ndata))

cluster1[0] = np.concatenate(

    (np.random.randn(1, round(0.5 * ndata)) * sigmaA - mA[0],

     np.random.randn(1, round(0.5 * ndata)) * sigmaA + mA[0]),

    axis=1

)

cluster1[1] = np.random.randn(1, ndata) * sigmaA + mA[1]

cluster2 = np.random.randn(2, ndata) * sigmaB + np.transpose(mB)



Ta = -np.ones((1, cluster1.shape[1]))

Tb = np.ones((1, cluster2.shape[1]))

X0 = np.concatenate((cluster1, cluster2), axis=1)

T = np.concatenate((Ta, Tb), axis=1)



# Try different combinations of these parameters.

n_hidden = 5

momentum = 0.9

learning_rate = 0.5

n_epochs = 200



mlp = MLP(2, n_hidden, SymmetricSigmoid(), Threshold(-1, 1), momentum=momentum, learning_rate=learning_rate)

mlp.train(X0, T, n_epochs=n_epochs)



fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122, aspect="equal")

ax1.plot(range(1, len(mlp.epoch_losses) + 1), mlp.epoch_losses)

plot_decision_region(mlp, X0, T, ax2)

fig.suptitle("Training results for a binary classifier", fontsize=16)

ax1.set_xlabel("Epochs")

ax1.set_ylabel("Error")

ax1.set_title("Training error")

ax2.set_title("Decision boundary")

ax2.legend()

plt.show()
def func2d_b(x1, x2):

    return np.square(x1 - 0.4) + np.square(x2 - 0.4) <= np.square(0.3)



n_training_examples = 5000

X0 = np.random.rand(2, n_training_examples)

T = -np.array([func2d_b(*X0) * 2 - 1])  # -1 for false and 1 for true.



# Try different combinations of these parameters.

n_hidden = 5

momentum = 0.9

learning_rate = 0.5

n_epochs = 200



mlp = MLP(2, n_hidden, SymmetricSigmoid(), Threshold(-1, 1), momentum=momentum, learning_rate=learning_rate)

mlp.train(X0, T, n_epochs=n_epochs)



fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122, aspect="equal")

ax1.plot(range(1, len(mlp.epoch_losses) + 1), mlp.epoch_losses)

plot_decision_region(mlp, X0, T, ax2)

fig.suptitle("Training results for a binary classifier", fontsize=16)

ax1.set_xlabel("Epochs")

ax1.set_ylabel("Error")

ax1.set_title("Training error")

ax2.set_title("Decision boundary")

ax2.legend()

plt.show()
def func2d_a(x1, x2):

    b = np.abs(x1 - 0.3 * x2 + 0.10 * np.sin(2 * np.pi * x2) - 0.25)

    return b >= 0.2

    return np.logical_and(b >= 0.2, b < 0.7, b > 0)



n_training_examples = 5000

X0 = np.random.rand(2, n_training_examples)

T = -np.array([func2d_a(*X0) * 2 - 1])  # -1 for false and 1 for true.



# Try different combinations of these parameters.

n_hidden = 5

momentum = 0.9

learning_rate = 0.003

n_epochs = 200



mlp = MLP(2, n_hidden, SymmetricSigmoid(), Threshold(-1, 1), momentum=momentum, learning_rate=learning_rate)

mlp.train(X0, T, n_epochs=n_epochs)



fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122, aspect="equal")

ax1.plot(range(1, len(mlp.epoch_losses) + 1), mlp.epoch_losses)

plot_decision_region(mlp, X0, T, ax2)

fig.suptitle("Training results for a binary classifier", fontsize=16)

ax1.set_xlabel("Epochs")

ax1.set_ylabel("Error")

ax1.set_title("Training error")

ax2.set_title("Decision boundary")

ax2.legend()

plt.show()
def func1d(x):

    return np.sin(2 * x)



n_training_examples = 1000

X0 = np.random.rand(1, n_training_examples) * 2 * np.pi

T = np.array([func1d(*X0)])



# Try different combinations of these parameters.

n_hidden = 10

momentum = 0.01

learning_rate = 0.003

n_epochs = 2000



mlp = MLP(2, n_hidden, SymmetricSigmoid(), Linear(), momentum=momentum, learning_rate=learning_rate)

mlp.train(X0, T, n_epochs=n_epochs)



# mlp = MLP(1, 100, SymmetricSigmoid(), Linear(), momentum=0.7,

#         learning_rate=0.005)

# mlp.train(X, targets, n_epochs=9000)



n_validation_examples = 1000

X = np.random.rand(1, n_validation_examples) * 2 * np.pi

Y = mlp.classify(X)



fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

ax1.plot(range(1, len(mlp.epoch_losses) + 1), mlp.epoch_losses)

ax2.scatter(X, Y, s=6, label="approximation")

ax2.scatter(X, func1d(*X), s=6, label="sin(x)")

fig.suptitle("Approximating a cycle of sin(x)", fontsize=16)

ax1.set_xlabel("Epochs")

ax1.set_ylabel("Error")

ax1.set_title("Training error")

ax2.set_title("Decision boundary")

ax2.set_xlabel("x")

ax2.set_ylabel("y")

ax2.legend()

plt.show()