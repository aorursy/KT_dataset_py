# Import libs

import math

import numpy as np

import matplotlib.pyplot as plt



from pylab import rcParams
def logloss(true_label, predicted, eps=1e-15):

    p = np.clip(predicted, eps, 1 - eps)

    if true_label == 1:

        return -math.log(p)

    else:

        return -math.log(1 - p)
def acc2logloss(acc, size, e):

    p = int(acc * size)

    true_l = [1 for i in range(size)]

    pred_l = [1 for i in range(p)] + [0 for i in range(size - p)]

    ll = 0

    for t, p in zip(true_l, pred_l):

        ll += logloss(t, p, eps=e)

    return ll / size
def plot_logloss(acc, size=4000):

    eps = np.arange(0.01, 0.2, 0.01)

    res = []

    for e in eps:

        res.append(acc2logloss(acc, size, e))

    min_logloss_ind = np.argmin(res)

    plt.plot(eps, res, label=str(acc))

    plt.plot(eps[min_logloss_ind], res[min_logloss_ind], marker='*')

    plt.text(eps[min_logloss_ind], res[min_logloss_ind], str((np.round(eps[min_logloss_ind], 2), np.round(res[min_logloss_ind], 2))), size=15, color="black")

    plt.xlabel("Epsilon")

    plt.ylabel("Logloss")
rcParams['figure.figsize'] = 10,10

accs = np.arange(0.8, 0.9, 0.01)

for acc in accs:

    plot_logloss(np.round(acc, 2))

plt.legend()

plt.title("0.80-0.90")

plt.show()



accs = np.arange(0.9, 1.0, 0.01)

for acc in accs:

    plot_logloss(np.round(acc, 2))

plt.legend()

plt.title("0.90-1.00")

plt.show()