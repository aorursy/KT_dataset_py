import numpy as np

X = np.array([-2, -1,  0,  1,  2,  3,  4,  5], dtype=np.float)
Y = np.array([-3, -1,  1,  3,  5,  7,  9, 11], dtype=np.float)
import tensorflow as tf

tf.__version__
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X, Y, epochs=500)
model.predict([12.0])

np.random.seed(2020) # to make reproducible results
W = 0.01 * np.random.randn(1) # In deep learning it is important to initialize the weights randomly. For our example, 0 would suffice.
b = 0
def forward(X, W, b):
    yhat = W * X + b
    return yhat 
yhat = forward(12.0, W, b)
yhat[0]
def loss(yhat, Y):
    m = len(yhat)
    loss = 1/m * np.sum(yhat - Y)**2
    return loss
loss(yhat, 12)
def backward(X, Y, yhat):
    m = len(yhat)
    dW = 1/m * np.sum( -2 * X * (Y - yhat))
    db = 1/m * np.sum( -2 * (Y - yhat))
    return (dW, db)
(dW, db) = backward(12.0, 25, yhat)
(dW, db)
def update(W, b, dW, db, learning_rate = 0.01):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)
(W, b) = update(W, b, dW, db)
(W, b)
num_iterations = 500

# a reset for W and b
np.random.seed(2020)
W = 0.01 * np.random.randn(1)
b = 0

for i in range(num_iterations):
    yhat = forward(X, W, b)
    l = loss(Y, yhat)
    dW, db = backward(X, Y, yhat)
    W, b = update(W, b, dW, db)
    if i % 100 == 0:
        print(l)
forward(12.0, W, b)
import pandas as pd
df = pd.read_csv('../input/titanic//train_data.csv')
df.shape
df.info()
df = df.drop(['Unnamed: 0', 'PassengerId'], axis=1)
df.sample(5)
Y = df['Survived'].to_numpy()
X = df.iloc[:,1:].to_numpy()
Y.shape
X.shape
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[14], activation='sigmoid'),
])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
train_history = model.fit(X, Y, epochs=500)
W_tf, b_tf = [x.numpy() for x in model.weights]
W_tf, b_tf
import matplotlib.pyplot as plt
plt.xkcd()
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(np.arange(500), train_history.history['loss'], 'b-', label='loss')
xlab, ylab = ax.set_xlabel('epoch'), ax.set_ylabel('loss')
X = X.T
X.shape
Xtry = X[ :, :2]
Ytry = Y[:2]
Xtry.shape
Ytry
np.random.seed(2020)
W = 0.01 * np.random.randn(14)
b = 0
W.shape
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

sigmoid(np.array([-100, 0, 0.1, 1000]))
def forward(X, W, b):
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    return A

A = forward(Xtry, W, b)
A
def loss(A, Y, epsilon = 1e-15):
    m = len(A)
    l = -1/m * np.sum( Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
    return l

loss(A, Ytry)
def backward(X, Y, A):
    m = len(yhat)
    dW = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y) 
    return (dW, db)

(dW, db) = backward(Xtry, Ytry, A)
dW, db
def update(W, b, dW, db, learning_rate = 0.01):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

update(W, b, dW, db)
def roundValue(A):
    return np.uint8( A > 0.5)

yhat = roundValue(A)
yhat
def accuracy(yhat, Y):
    return round(np.sum(yhat==Y) / len(yhat) * 1000) / 10
num_iterations = 8000
lr = 0.01

# Lets just reset W and b
np.random.seed(2020)
W = 0.01 * np.random.randn(14)
b = 0

losses, acces = [], []
for i in range(num_iterations):
    A = forward(X, W, b)
    l = loss(Y, A)
    yhat = roundValue(A)
    acc = accuracy(yhat, Y)
    dW, db = backward(X, Y, A)
    W, b = update(W, b, dW, db, learning_rate=lr)
    losses.append(l)
    acces.append(acc)
    if i % 1000 == 0:
        print('loss:', l, f'\taccuracy: {accuracy(yhat, Y)}%') 
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(np.arange(len(losses)), losses, 'b-', label='loss')
xlab, ylab = ax.set_xlabel('epoch'), ax.set_ylabel('loss')
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(np.arange(len(acces)), acces, 'b-', label='accuracy')
xlab, ylab = ax.set_xlabel('epoch'), ax.set_ylabel('accuracy')
