import numpy as np

from matplotlib import pyplot as plt
# base class for neural network layers

class Layer():

    def __init__(self):

        pass

    

    # forward pass

    # compute output value y given input x

    def forward(self, x):

        raise NotImplementedError

    

    # backward pass

    # compute gradient for weight variables dE/dW

    # and compute gradient for input dE/dx

    # given `node_grad` which is the gradient passed from previous layers dE/dy

    def backward(self, node_grad):

        raise NotImplementedError

    

    # update weight variables based on gradients

    def update(self, learning_rate):

        raise NotImplementedError
class Sigmoid(Layer):    

    def forward(self, x):

        self.y = 1 / (1 + np.exp(-x))

        return self.y

    

    def backward(self, node_grad):

        return node_grad * (self.y * (1-self.y))

    

    # no parameters to train

    def update(self, learning_rate):

        pass
class Relu():        

    def forward(self, x):

        self.x = x

        return np.maximum(x, 0)

    

    def backward(self, node_grad):

        return node_grad * (self.x > 0)

    

    def update(self, learning_rate):

        pass
class Softmax_Cross_Entropy():

    # forward pass with softmax

    def forward(self, x):

        exps = np.exp(x - np.max(x)) # float64 is 10^308, exp with too large x will yield inf

        self.y = exps / np.sum(exps)

        return self.y

    

    # backward pass

    # compute gradient of softmax+crossentropy simultaneously

    def backward(self, label):

        self.out_grad = self.y - label

        return self.out_grad

    

    def update(self, learning_rate):

        pass
class Linear(Layer):

    def __init__(self, size_in, size_out, with_bias):

        self.size_in = size_in

        self.size_out = size_out

        self.with_bias = with_bias

        self.W = self.initialize_weight()

        if with_bias:

            self.b = np.zeros(size_out)

    

    # initialization is important

    # dangerous if set to 0

    def initialize_weight(self):

        epsilon = np.sqrt(2.0 / (self.size_in + self.size_out))

        return epsilon * (np.random.rand(self.size_in, self.size_out) * 2 - 1) 

        

    def forward(self, x):

        self.x = x

        self.y = x @ self.W + self.b  # @ is shortcut for matrix multiplication

        return self.y

    

    def backward(self, node_grad):

        self.G_W = np.outer(self.x, node_grad)

        if self.with_bias:

            self.G_b = node_grad

        return node_grad @ self.W.T

    

    # update weight parameters, perform gradient descent

    def update(self, learning_rate):

        self.W -= learning_rate * self.G_W

        if self.with_bias:

            self.b -= learning_rate * self.G_b
class MLP():

    def __init__(self, layer_size, with_bias=True, activation="sigmoid", learning_rate=1):

        assert len(layer_size) >= 2

        self.layer_size = layer_size

        self.with_bias = with_bias

        if activation == "sigmoid":

            self.activation = Sigmoid

        elif activation == "relu":

            self.activation = Relu

        else:

            raise Exception("activation not implemented")

        self.learning_rate = learning_rate

        self.build_model()

        

    def build_model(self):

        self.layers = []

        

        size_in = self.layer_size[0]

        for hu in self.layer_size[1:-1]:

            self.layers.append(Linear(size_in, hu, self.with_bias))

            self.layers.append(self.activation())

            size_in = hu

            

        # final layer uses softmax+crossentropy

        self.layers.append(Linear(size_in, self.layer_size[-1], self.with_bias))

        self.layers.append(Softmax_Cross_Entropy())

        

    def forward(self, x):

        for layer in self.layers:

            x = layer.forward(x)

        return x

    

    def backward(self, label):

        node_grad = label

        for layer in reversed(self.layers):

            node_grad = layer.backward(node_grad)

            

    def update(self, learning_rate):

        for layer in self.layers:

            layer.update(learning_rate)

            

    def train(self, x, label):

        y = self.forward(x)

        self.backward(label)

        self.update(self.learning_rate)

    

    def predict(self, x):

        for layer in self.layers:

            x = layer.forward(x)

        return np.argmax(x)

    

    def loss(self, x, label):

        y = self.forward(x)

        return -np.log(y) @ label # cross entropy loss
X = np.array([

    [0, 0],

    [0, 1],

    [1, 0],

    [1, 1]

])

Y = np.array([

    [0, 1],

    [1, 0],

    [1, 0],

    [0, 1]

])
np.random.seed(1007) # set random seed

EPOCH = 1000 # set training epochs

N = X.shape[0] # get number of training data



# craete a MLP with layer size [2,4,2] and learning_rate 0.1

mlp = MLP([2, 4, 2], learning_rate=.1, activation="relu")



loss = np.zeros(EPOCH) # store losses

for epoch in range(EPOCH):

    # train on each training data

    for i in range(N):

        mlp.train(X[i], Y[i])

        

    # compute loss

    for i in range(N):

        loss[epoch] += mlp.loss(X[i], Y[i])

        

    loss[epoch] /= N

    

# plot loss curve

plt.figure()

ix = np.arange(EPOCH)

plt.plot(ix, loss)
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD



model = Sequential()

model.add(Dense(8, input_dim=2, activation="relu"))

model.add(Dense(2, activation="softmax"))



sgd = SGD(lr=.1)



model.compile(optimizer=sgd,

              loss='categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit(X, Y, epochs=200)