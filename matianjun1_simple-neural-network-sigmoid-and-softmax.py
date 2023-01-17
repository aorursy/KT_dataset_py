import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
images_labels = pd.read_csv('../input/train.csv', dtype="float")

images_labels.head()
images = images_labels.iloc[:, 1:]

images.head()
labels = images_labels.iloc[:, :1]

labels.head()
images.shape
num_data = images.shape[0]

indexes = np.random.choice(range(num_data), num_data)

train_percent = 0.8

train_images = images.as_matrix()[indexes[:int(num_data * train_percent)]]

train_labels = labels.as_matrix().astype("int")[indexes[:int(num_data * train_percent)]]

train_labels = train_labels.reshape(train_labels.shape[0])

test_images = images.as_matrix()[indexes[int(num_data * train_percent):]]

test_labels = labels.as_matrix().astype("int")[indexes[int(num_data * train_percent):]]

test_labels = test_labels.reshape(test_labels.shape[0])

print(train_images.shape)

print(train_labels.shape)

print(test_images.shape)

print(test_labels.shape)
i = 1

img = train_images[i]

img = img.reshape((28,28))

plt.imshow(img, cmap='gray')

plt.title(train_labels[i])
plt.hist(train_images[i])
class ThreeLayerNeuralNetwork():

    def __init__(self):

        self.W1 = None

        self.W2 = None

    def train(self, X, y):

        learning_rate = 1e-3

        max_iter = 10000

        batch_size = 200

        # TODO init parameters

        N = X.shape[0]

        D1 = X.shape[1]

        D2 = 256 # hidden size

        Y = 10 # catalog

        if self.W1 is None:

            self.W1 = 0.01*np.random.randn(D1, D2)

        if self.W2 is None:

            self.W2 = 0.01*np.random.randn(D2, Y)

        

        for i in range(max_iter):

            # choice batch train set

            choices = np.random.choice(range(N), batch_size)

            X_batch = X[choices]

            y_batch = y[choices]

            

            # caculate loss

            loss, grad = self.loss(X_batch, y_batch)

            if (i + 1) % 100 == 0:

                print("{:d}/{:d} loss: {:2f}".format(i+1, max_iter, loss))

            # update parameters

            self.W1 -= learning_rate * grad["W1"]

            self.W2 -= learning_rate * grad["W2"]

        

        

    def loss(self, X, y):

        # Caculate y_pred

        # h = sigmoid(X*W1)

        h = 1 / (1 + np.exp(-X.dot(self.W1)))

        # y_pred = softmax(h*W2)

        s = np.exp(h.dot(self.W2))

        y_pred = (s.T / np.sum(s, axis=1)).T

        

        # Caculate loss

        loss = -np.sum(np.log(y_pred[range(X.shape[0]), y])) / X.shape[0]

        

        # Caculate gradient

        grad = {}

        y_pred[range(X.shape[0]), y] -= 1

        grad["W2"] = h.T.dot(y_pred) / X.shape[0]

        grad["W1"] = X.T.dot(h * (1-h)) * np.sum(y_pred.dot(self.W2.T), axis=0) / X.shape[0]

        

        # TODO regularzation

        # No need to regularize

        

        return loss, grad

    

    def predict(self, X):

        h = 1 / (1 + np.exp(-X.dot(self.W1)))

        s = np.exp(h.dot(self.W2))

        y_pred = (s.T / np.sum(s, axis=1)).T

        return np.argmax(y_pred, axis=1)

    
nn = ThreeLayerNeuralNetwork()

nn.train(train_images, train_labels)
y_pred = nn.predict(test_images)

print((y_pred == test_labels).mean())
y_pred = nn.predict(train_images)

print((y_pred == train_labels).mean())
test_data = pd.read_csv('../input/test.csv').as_matrix()

results = nn.predict(test_data)

results
df = pd.DataFrame(results)

df.index += 1

df.index.name = 'ImageId'

df.columns = ['Label']

df.to_csv('../input/sample_submission.csv', header=True)