import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles, make_classification
# Model parametres

input_size = 2  # no_of_features

layers = [4, 3]  # no. of neurons in 1st and 2nd layer

output_class = 2
def softmax(a):

    e_pa = np.exp(a)

    ans = e_pa / np.sum(e_pa, axis=1, keepdims=True)

    return ans
a = np.array([[10, 20]])

result = softmax(a)

print(result)
class NeuralNetwork:

    def __init__(self, input_size, layers, output_size):

        np.random.seed(0)



        model = {}  #Dictionary



        #First Layer

        model['W1'] = np.random.randn(input_size, layers[0])

        model['b1'] = np.zeros((1, layers[0]))



        #Second Layer

        model['W2'] = np.random.randn(layers[0], layers[1])

        model['b2'] = np.zeros((1, layers[1]))



        #Third/Output Layer

        model['W3'] = np.random.randn(layers[1], output_size)

        model['b3'] = np.zeros((1, output_size))



        self.model = model

        self.activation_outputs = None



    def forward(self, x):



        W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']

        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']



        z1 = np.dot(x, W1) + b1

        a1 = np.tanh(z1)



        z2 = np.dot(a1, W2) + b2

        a2 = np.tanh(z2)



        z3 = np.dot(a2, W3) + b3

        y_ = softmax(z3)



        self.activation_outputs = (a1, a2, y_)

        return y_



    def backward(self, x, y, learning_rate=0.001):

        W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']

        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']

        m = x.shape[0]



        a1, a2, y_ = self.activation_outputs



        delta3 = y_ - y

        dw3 = np.dot(a2.T, delta3)

        db3 = np.sum(delta3, axis=0)



        delta2 = (1 - np.square(a2)) * np.dot(delta3, W3.T)

        dw2 = np.dot(a1.T, delta2)

        db2 = np.sum(delta2, axis=0)



        delta1 = (1 - np.square(a1)) * np.dot(delta2, W2.T)

        dw1 = np.dot(X.T, delta1)

        db1 = np.sum(delta1, axis=0)



        #Update the Model Parameters using Gradient Descent

        self.model["W1"] -= learning_rate * dw1

        self.model['b1'] -= learning_rate * db1



        self.model["W2"] -= learning_rate * dw2

        self.model['b2'] -= learning_rate * db2



        self.model["W3"] -= learning_rate * dw3

        self.model['b3'] -= learning_rate * db3



    def predict(self, x):

        y_out = self.forward(x)

        return np.argmax(y_out, axis=1)



    def summary(self):

        W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']

        a1, a2, y_ = self.activation_outputs



        print("W1 ", W1.shape)

        print("A1 ", a1.shape)



        print("W2 ", W2.shape)

        print("A2 ", a2.shape)



        print("W3 ", W3.shape)

        print("Y_ ", y_.shape)
def loss(y_oht, p):

    l = -np.mean(y_oht * np.log(p))

    return l





def one_hot(y, depth):



    m = y.shape[0]

    y_oht = np.zeros((m, depth))

    y_oht[np.arange(m), y] = 1

    return y_oht
def train(X, Y, model, epochs, learning_rate, logs=True):

    training_loss = []



    classes = 2

    Y_OHT = one_hot(Y, classes)



    for ix in range(epochs):



        Y_ = model.forward(X)

        l = loss(Y_OHT, Y_)

        training_loss.append(l)

        model.backward(X, Y_OHT, learning_rate)



        if (logs and ix%50==0):

            print("Epoch %d Loss %.4f" % (ix, l))



    return training_loss
def plot_decision_boundary(model, X, y, cmap=plt.cm.jet):



    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = 0.01



    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))



    Z = model(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    plt.ylabel('x2')

    plt.xlabel('x1')

    plt.style.use("seaborn")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.jet)

    plt.show()
X, Y = make_circles(n_samples=500, shuffle=True, noise=0.2, random_state=1, factor=0.2)
plt.style.use('seaborn')

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Accent)

plt.show()
model = NeuralNetwork(input_size=2, layers=[4, 3], output_size=2)
model.forward([X]).shape
model.summary()
losses = train(X, Y, model, 500, 0.001)

# print(losses)
plt.plot(losses)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()
fig = plt.figure(figsize=(5, 5))

ax = fig.add_axes([0, 0, 1, 1])

labels1 = ['zeros', 'ones']

lab_colors = ['blue', 'orange']

lab_counts = np.unique(model.predict(X), return_counts=True)[1]

ax.bar(labels1, lab_counts, color=lab_colors)

plt.show()
plot_decision_boundary(lambda x: model.predict(x), X, Y)
np.unique(model.predict(X), return_counts=True)
outputs = model.predict(X)

training_accuracy = np.sum(outputs == Y) / Y.shape[0]

print(f"Training accuracy is {training_accuracy*100}%")
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

Y = np.array([0, 1, 1, 0])
losses = train(X, Y, model, 500, 0.001)

print(losses)
plt.plot(losses)

plt.show()
plot_decision_boundary(lambda x: model.predict(x), X, Y)
outputs = model.predict(X)

training_accuracy = np.sum(outputs == Y) / Y.shape[0]

print(f"Training accuracy is {training_accuracy*100}%")
def load_dataset(dataset):

    if dataset == 'moons':

        X, Y = make_moons(n_samples=500, noise=0.2,

                          random_state=1)  #Perceptron

    elif dataset == 'circles':

        X, Y = make_circles(n_samples=500,

                            shuffle=True,

                            noise=0.2,

                            random_state=1,

                            factor=0.2)

    elif dataset == 'classification':

        X, Y = make_classification(n_samples=500,

                                   n_classes=2,

                                   n_features=2,

                                   n_informative=2,

                                   n_redundant=0,

                                   random_state=1)

    else:

        #Create XOR Dataset

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        Y = np.array([0, 1, 1, 0])



    return X, Y
datasets = ["xor", "classification", "moons", "circles"]



for d in datasets:

    model = NeuralNetwork(input_size=2, layers=[4, 3], output_size=2)

    X, Y = load_dataset(d)

#     losses = train(X, Y, model, 500, 0.001, logs=False)

    train(X, Y, model, 1000, 0.001, logs=False)

    outputs = model.predict(X)

    training_accuracy = np.sum(outputs == Y) / Y.shape[0]



    print(f"Training accuracy : {training_accuracy*100}%")

    plt.title(f"Dataset | {d}")

    plot_decision_boundary(lambda x: model.predict(x), X, Y)

#     plt.plot(losses)

    plt.show()