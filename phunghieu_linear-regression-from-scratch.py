from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
DATA_ROOT = Path('/kaggle/input/house-prices-advanced-regression-techniques/')

TRAIN_CSV = DATA_ROOT / 'train.csv'

TEST_CSV = DATA_ROOT / 'test.csv'



LR = 1e-2

EPOCHS = 10000
train_df = pd.read_csv(TRAIN_CSV)

train_df.head()
data = train_df[['LotFrontage', 'SalePrice']]

data = data.dropna()

X = data.iloc[:, 0].tolist()

y = data.iloc[:, 1].tolist()
fig, ax = plt.subplots(figsize=(16, 8))



ax.scatter(X, y)

plt.show()
min_X = min(X)

max_X = max(X)



numerator = [X_i - min_X for X_i in X]

denominator = max_X - min_X



X = [item / denominator for item in numerator]
def infer(X, W):

    y_hat = W[0] * X + W[1]

    

    return y_hat





def compute_loss(X, y, W):

    bs = len(X)

    E = 0

    

    for X_i, y_i in zip(X, y):

        y_hat_i = infer(X_i, W)

        

        err = (y_hat_i - y_i) ** 2

        E += err



    L = (1 / (2 * bs)) * E



    return L





def update_weights(X, y, W, lr, epochs):

    bs = len(X)

    L_history = []



    for epoch in tqdm(range(epochs)):

        grad_w0 = 0

        grad_w1 = 0

        

        for X_i, y_i in zip(X, y):

            y_hat_i = infer(X_i, W)

            grad_w0_i = (y_hat_i - y_i) * X_i

            grad_w1_i = y_hat_i - y_i

            

            grad_w0 += grad_w0_i

            grad_w1 += grad_w1_i

            

        W[0] -= lr * (1 / bs) * grad_w0

        W[1] -= lr * (1 / bs) * grad_w1

        

        L = compute_loss(X, y, W)

        

        L_history.append(L)

        

    return W, L_history
W = [0, 0]
W, L_history = update_weights(X, y, W, LR, EPOCHS)
print(f'New weights: {W}')
fig, ax = plt.subplots(figsize=(16, 8))



ax.plot(L_history)

plt.show()
fig, ax = plt.subplots(figsize=(16, 8))



ax.scatter(X, y)

ax.plot(W, 'r--')

plt.show()