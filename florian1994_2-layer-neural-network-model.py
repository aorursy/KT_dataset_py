import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/train.csv")
# Extract data from dataframe
data = df.as_matrix()
# Split data into training set and validation set
y = data[:, 0]
X = data[:, 1:].astype(np.float64)
train_num = 41000
val_num = 1000
X_train, y_train = X[:train_num], y[:train_num]
X_val, y_val = X[train_num:], y[train_num:]
print(X_train.shape, y_train.shape, X_train.dtype, y_train.dtype)
print(X_val.shape, y_val.shape, X_val.dtype, y_val.dtype)
# Data Preprocessing
mean_pixel = X_train.mean(axis=0)
X_train -= mean_pixel
X_val -= mean_pixel
# An simple 2-layers full-connected neural network model
# Note we only use numpy 

# Initialize our nn
def initialize_global_weights():
    global W1, b1, W2, b2
    N, D = train_num, 784
    H, C = 500, 10
    W1 = 0.001 * np.random.rand(D, H)
    b1 = np.zeros(H)
    W2 = 0.001 * np.random.rand(H, C)
    b2 = np.zeros(C)

initialize_global_weights()
# Implement our train function
def train_or_evaluate(X, y=None, loss_fn=None, lr=1e-3, reg=0.0):
    global W1, W2, b1, b2
    # forward pass
    a = X.dot(W1) + b1
    scores = a.dot(W2) + b2
    if y is None:
        return scores
    loss, dscores = loss_fn(scores, y)
    print('loss: %f' % loss)
    # backward pass
    dW2 = np.dot(a.T, dscores) + reg * W2
    db2 = np.sum(dscores, axis=0)
    da = np.dot(dscores, W2.T)
    db1 = np.sum(da, axis=0)
    dW1 = np.dot(X.T, da) + reg * W1
    # update params
    W1 += - lr * dW1
    W2 += - lr * dW2
    b1 += - lr * db1
    b2 += - lr * db2
    return loss
# Implement our softmax loss function
def softmax(scores, y):
    N = scores.shape[0]
    scores = scores.copy()
    scores -= np.max(scores, axis=1)[:, None]
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=1)[:, None]
    loss = np.sum(-np.log(probs[np.arange(N), y])) / N
    
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    
    return loss, dscores
# Use initialized weight to checkout train accuracy
scores = train_or_evaluate(X_train)
print((np.argmax(scores, axis=1) == y_train).mean())
# Training our 2-layer model
num_iters = 50
initialize_global_weights()
for i in range(num_iters):
    loss = train_or_evaluate(X_train, y_train, softmax, lr=1e-7, reg=1e-5)
    if np.isinf(loss):
        break
# Use trained weight to checkout train accuracy and val accuracy
train_scores = train_or_evaluate(X_train)
train_acc = (np.argmax(train_scores, axis=1) == y_train).mean()
val_scores = train_or_evaluate(X_val)
val_acc = (np.argmax(val_scores, axis=1) == y_val).mean() 
print(train_acc, val_acc)
