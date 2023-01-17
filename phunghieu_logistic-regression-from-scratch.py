from pathlib import Path

import math



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
DATA_ROOT = Path('/kaggle/input/graduate-admissions/')

TRAIN_CSV = DATA_ROOT / 'Admission_Predict.csv'



LR = 3e-2

EPOCHS = 50000
train_df = pd.read_csv(TRAIN_CSV)

train_df.head()
data = train_df[['CGPA', 'Research']]

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
def sigmoid(z):

    return 1 / (1 + math.e ** -z)



def infer(X, W):

    y_hat = sigmoid(W[0] * X + W[1])

    

    return y_hat





def compute_loss(X, y, W):

    bs = len(X)

    E = 0

    

    for X_i, y_i in zip(X, y):

        y_hat_i = infer(X_i, W)

        

        err = (y_hat_i - y_i) ** 2

        err = y_i * math.log(y_hat_i) + (1 - y_i) * math.log(1 - y_hat_i)

        E += err



    L = -(1 / bs) * E



    return L





def update_weights(X, y, W, lr, epochs):

    bs = len(X)

    L_history = []



    for epoch in tqdm(range(epochs)):

        grad_w0 = 0

        grad_w1 = 0

        

        for X_i, y_i in zip(X, y):

            y_hat_i = infer(X_i, W)

            grad_w0_i = (y_hat_i * (1 - y_i) - y_i * (1 - y_hat_i)) * X_i

            grad_w1_i = y_hat_i * (1 - y_i) - y_i * (1 - y_hat_i)

            

            grad_w0 += grad_w0_i

            grad_w1 += grad_w1_i

            

        W[0] -= lr * (1 / bs) * grad_w0

        W[1] -= lr * (1 / bs) * grad_w1

        

        L = compute_loss(X, y, W)

        

        L_history.append(L)

        

    return W, L_history





def calculate_prec_recall_f1(preds, labels, threshold=0.5, epsilon=1e-7):

    labels = np.array(labels, dtype=np.uint8)

    preds = (np.array(preds) >= threshold).astype(np.uint8)

    tp = np.count_nonzero(np.logical_and(labels, preds))

    tn = np.count_nonzero(np.logical_not(np.logical_or(labels, preds)))

    fp = np.count_nonzero(np.logical_not(labels)) - tn

    fn = np.count_nonzero(labels) - tp

    precision = tp / (tp + fp + epsilon)

    recall = tp / (tp + fn + epsilon)

    f1 = (2 * precision * recall) / (precision + recall + epsilon)

    

    return precision, recall, f1





def evaluate(X, y, W):

    y_hat = []

    for X_i, y_i in zip(X, y):

        y_hat_i = infer(X_i, W)

        y_hat.append(y_hat_i)

    

    prec, recall, f1 = calculate_prec_recall_f1(y_hat, y)

    print(f'Precision: {prec}')

    print(f'Recall: {recall}')

    print(f'F1: {f1}')
W = [0, 0]
W, L_history = update_weights(X, y, W, LR, EPOCHS)
print(f'New weights: {W}')
# Compute metrics for training set

evaluate(X, y, W)
fig, ax = plt.subplots(figsize=(16, 8))



ax.plot(L_history)

plt.show()
fig, ax = plt.subplots(figsize=(16, 8))



ax.scatter(X, y)

tmp_X = [item / 100 for item in range(101)]

tmp_y = [infer(item, W) for item in tmp_X]

ax.plot(tmp_X, tmp_y, 'r--')

plt.show()