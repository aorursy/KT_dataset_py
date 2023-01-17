import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for drawing

import seaborn as sns # for drawing as well

from scipy.optimize import minimize # will use for training

from sklearn.metrics import accuracy_score, f1_score # some metrics

from functools import partial # better google what it is

from tqdm import tqdm as tqdm # nice trackbar



import torch # this is the best library for neural networks (IMHO)

import torch.nn as nn # high-level API, useful for deep learning



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

os.listdir('/kaggle/input/digit-recognizer')
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



train.head()
pixels = np.array(train.iloc[0]) # get first line

pixels = pixels[1:] # remove label

pixels = pixels.reshape((28,28)) # reshape to normal image shape



plt.imshow(pixels, cmap='gray');
# max pixel value is 255, but let's make it from 0 to 1

pixels01 = pixels / 255



# make plot a bit bigger

plt.figure(figsize=(14,11))



sns.heatmap(np.round(pixels01, 1), annot=True, cmap='gray');
X_train = np.array(train.drop('label', 1)) / 255

y_train = np.array(train['label'])



X_test = np.array(test) / 255



print('X_train.shape', X_train.shape)

print('y_train.shape', y_train.shape)

print('X_test.shape', X_train.shape)
# Let's make a random neuron



neuron_r = np.random.normal(size=(784,))

neuron_r /= np.sqrt(784) # less variance



def sigmoid(x):

    ex = np.exp(x)

    return ex / (1 + ex)
# Let's apply a random neuron to some image:

x = X_train[0]

output = sigmoid(x @ neuron_r)

print(output)
# Let's make a new training set

y_train1 = (y_train == 1).astype(int)



print(y_train1[:20])
predictions = sigmoid(X_train @ neuron_r)

print('Predictions:', predictions[:10])

print('Truth:', y_train1[:10])
def cross_entropy(y_true, y_pred):

    out1 = y_true * np.log(y_pred + 1e-12) # small number for numerical stability

    out2 = (1 - y_true) * np.log(1 - y_pred + 1e-12)

    return -(out1 + out2).mean()



print(cross_entropy(y_train1, predictions))
%%time



def _loss(neuron, X, y):

    predictions = sigmoid(X @ neuron)

    error = cross_entropy(y, predictions)

    return error



# let's use only 500 examples, because otherwise it takes really long (later will solve it)

error = partial(_loss, X=X_train[:500], y=y_train1[:500])



neuron_1 = minimize(error, x0=neuron_r).x
predictions = sigmoid(X_train @ neuron_1)

print('Error:', cross_entropy(y_train1, predictions))
print('Accuracy:', accuracy_score(y_train1, predictions > 0.5))

print('F1:', f1_score(y_train1, predictions > 0.5))
# this is array of weights for each pixel

neuron_1



# let's print it

neuron_image = neuron_1.reshape(28, 28)

sns.heatmap(neuron_image);
%%time



neurons = []



for digit in range(10):

    print('Training neuron', digit)

    y = y_train == digit

    error = partial(_loss, X=X_train[:100], y=y[:100])

    neuron = minimize(error, x0=neuron_r).x

    neurons.append(neuron)
fig, axes = plt.subplots(1, 10, figsize=(30,30))



for i in range(10):

    axes[i].set_title(i)

    neuron_img = neurons[i].reshape(28, 28)

    axes[i].imshow(neuron_img)
digit_probabilities = sigmoid(X_train @ np.array(neurons).T)

print('digit_probabilities.shape:', digit_probabilities.shape)

# Now we have probabilities for each image for each digit



# Let's take maximum as prediction

predictions = np.argmax(digit_probabilities, axis=1)

print('predictions.shape:', predictions.shape)

print('\nTrain accuracy:', accuracy_score(y_train, predictions))
# Let's make test predictions

digit_probabilities = sigmoid(X_test @ np.array(neurons).T)

test_predictions = np.argmax(digit_probabilities, axis=1)



submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.Label = test_predictions



# Save it as csv file

submission.to_csv('submission_10n.csv', index=False)



submission.head()
# Let's look at the example

A = torch.tensor(10)

A
B = torch.tensor([1, 2, 3, 5])

print(B)

C = torch.tensor([0, 1, 1, 2])

print(C)

D = B + C

print(D)

print(D.shape, D.dtype)
# Let's write some strange function

f = lambda x: (x**2 + 4*x + 6) * (x<0) + (x**2 - 2*x + 6) * (x>0)



x = np.linspace(-4,6, 100)

plt.plot(x, f(x));
# Let's say, parameter x with default value 5

x = torch.tensor(5.0, requires_grad=True)



optimizer = torch.optim.SGD(params=[x], lr=0.1) # SGD = stochastic gradient descent



# Will record x values here

x_history = [x.data.numpy().copy()]



# Let's make gradient descent steps

for i in range(30):

    # Compute the function (forward pass)

    loss = f(x)

    # Compute gradients (backward pass of backpropagation)

    optimizer.zero_grad()

    loss.backward()

    # Change x a bit

    optimizer.step()

    # Add to history

    x_history.append(x.data.numpy().copy())



x_history = np.array(x_history)

print('x =', x.data.numpy())
plt.figure(figsize=(14,10))

x = np.linspace(-4,6, 100)

plt.plot(x, f(x), label='f(x)')

plt.grid()

plt.scatter(x_history, f(x_history), color='orange', label='GD steps')

plt.legend();
%%time

# Weights of all neurons

x0 = np.random.normal(size=(784, 10)) / np.sqrt(784)

neurons = torch.tensor(x0, requires_grad=True)



# We will need our dataset as torch.tensor

X = torch.tensor(X_train)

y = torch.tensor(y_train)



optimizer = torch.optim.SGD(params=[neurons], lr=1) # SGD = stochastic gradient descent



loss_history = []



# Let's make gradient descent steps

for i in range(100):

    # Compute the function (forward pass)

    predictions = X @ neurons

    # Compute loss

    loss = torch.nn.functional.cross_entropy(predictions, y) # it already includes activation function

    # Compute gradients (backward pass of backpropagation)

    optimizer.zero_grad()

    loss.backward()

    # Change neurons a bit

    optimizer.step()

    loss_history.append(loss.data.numpy())
plt.title('Training history')

plt.ylabel('Loss')

plt.xlabel('step')

plt.plot(loss_history);
fig, axes = plt.subplots(1, 10, figsize=(30,30))



for i in range(10):

    axes[i].set_title(i)

    neuron_img = neurons[:,i].data.numpy().reshape(28, 28)

    axes[i].imshow(neuron_img)
# The same code as before



digit_probabilities = sigmoid((X @ neurons).data.numpy())

print('digit_probabilities.shape:', digit_probabilities.shape)

# Now we have probabilities for each image for each digit



# Let's take maximum as prediction

predictions = np.argmax(digit_probabilities, axis=1)

print('predictions.shape:', predictions.shape)

print('\nTrain accuracy:', accuracy_score(y_train, predictions))
# Let's make test predictions

digit_probabilities = sigmoid(X_test @ neurons.data.numpy())

test_predictions = np.argmax(digit_probabilities, axis=1)



submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.Label = test_predictions



# Save it as csv file

submission.to_csv('submission_10n_torch.csv', index=False)



submission.head()
# Will use torch's high level API: nn

# When we create a neural network in pytorch, we usually create a class which derives from torch.nn.Module

# It is easy to train and further use network defined in such way

# This network has 210 neurons



class MyFirstNN(nn.Module):

    def __init__(self, n_hidden_neurons=200):

        super().__init__()

        

        # Here we define the model's trainable paramerers

        

        # First layer weights

        init_1 = np.random.normal(size=(784, n_hidden_neurons)) / np.sqrt(784) # google for "xavier initialization"

        self.neurons_layer1 = torch.tensor(init_1, requires_grad=True) # weights for the first layer of neurons

        self.neurons_layer1 = nn.Parameter(self.neurons_layer1)

        

        # Second layer weights

        init_2 = np.random.normal(size=(n_hidden_neurons, 10)) / np.sqrt(n_hidden_neurons)

        self.neurons_layer2 = torch.tensor(init_2, requires_grad=True)

        self.neurons_layer2 = nn.Parameter(self.neurons_layer2)

        

    def forward(self, x):

        # Here we do all the computations

        

        # First layer

        h = x @ self.neurons_layer1

        # Activation function of a hidden layer

        h = torch.relu(h)

        # Output layer

        out = h @ self.neurons_layer2

        return out



# People use more high-level features, such as nn.Linear, to make the code shorter

# This is a simple and explicit example
model = MyFirstNN()



optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
N_EPOCHS = 501



# We will need our dataset as torch.tensor

X_tr = torch.tensor(X_train[1000:])

y_tr = torch.tensor(y_train[1000:])

X_dev = torch.tensor(X_train[:1000])

y_dev = torch.tensor(y_train[:1000])



train_loss_history = []

dev_loss_history = []



# Let's make gradient descent steps

for i in tqdm(range(N_EPOCHS)):

    # Compute the function (forward pass)

    predictions = model(X_tr)

    # Compute loss

    loss = torch.nn.functional.cross_entropy(predictions, y_tr) # it already includes activation function

    # Compute gradients (backward pass of backpropagation)

    optimizer.zero_grad()

    loss.backward()

    # Change neurons a bit

    optimizer.step()

    train_loss_history.append(loss.data.numpy())

    

    # Evaluate the model

    if i % 10 == 0:

        predictions = model(X_dev)

        loss = torch.nn.functional.cross_entropy(predictions, y_dev)

        dev_loss_history.append(loss.data.numpy())
plt.figure(figsize=(16, 8))

plt.title('Training history')

plt.ylabel('Loss')

plt.xlabel('step')

plt.plot(range(N_EPOCHS), train_loss_history, label='train loss')

plt.plot(range(N_EPOCHS)[::10], dev_loss_history, label='dev loss')

plt.legend();
# Evaluate accuracy on dev set



# Same code as before (but use dev set)

digit_probabilities = model(X_dev).data.numpy()

print('digit_probabilities.shape:', digit_probabilities.shape)

# Now we have probabilities for each image for each digit



# Let's take maximum as prediction

predictions = np.argmax(digit_probabilities, axis=1)

print('predictions.shape:', predictions.shape)

print('\Dev accuracy:', accuracy_score(y_train[:1000], predictions))
# Let's make test predictions

digit_probabilities = model(torch.tensor(X_test)).data.numpy()

test_predictions = np.argmax(digit_probabilities, axis=1)



submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.Label = test_predictions



# Save it as csv file

submission.to_csv('submission_NN.csv', index=False)



submission.head()