import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import numpy as np



def loss_1(y_true, y_pred):

    return np.log(np.abs(y_true - y_pred))



def loss(y_true, y_pred, delta_pred):

    delta_true = np.abs(y_true - y_pred)

    return delta_true / delta_pred + np.log(delta_pred)



def loss_2(delta_true, delta_pred):

    x = delta_true / delta_pred

    return x - np.log(x)
def check_correctness(n=100, eps=1e-6):

    y_true = np.random.rand(n)

    y_pred = np.random.rand(n)



    delta_true = np.abs(y_true - y_pred)

    delta_pred = np.random.rand(n)

    

    l1 = loss_1(y_true, y_pred)

    l2 = loss_2(delta_true, delta_pred)

    l = loss(y_true, y_pred, delta_pred)

    

    assert np.all(np.abs(l - l1 - l2) < eps)
for _ in range(1000):

    check_correctness()

print("OK!")
y_preds = np.arange(70, 800)

y_trues = [70, 100, 200, 300, 500, 1000]

colors = ["blue", "red", "green", "brown", "black"]



plt.figure(figsize=(16, 8))

plt.title("Loss_1")

for y_true, color in zip(y_trues, colors):

    plt.plot(y_preds, loss_1(y_true, y_preds), label="delta_true={}".format(y_true), color=color)

plt.xlabel("y_pred")

plt.ylabel("loss value")

_ = plt.legend()
delta_preds = np.arange(70, 1000)

delta_trues = [70, 100, 300, 500, 1000, 2000]

colors = ["blue", "green", "red", "brown", "black"]

plt.figure(figsize=(16, 8))



plt.title("Loss_2")

for delta_true, color in zip(delta_trues, colors):

    plt.plot(delta_preds, loss_2(delta_true, delta_preds), label="delta_true={}".format(delta_true), color=color)

    plt.scatter(delta_true, loss_2(delta_true, delta_true), color=color)

plt.xlabel("delta_pred")

plt.ylabel("value")

_ = plt.legend()