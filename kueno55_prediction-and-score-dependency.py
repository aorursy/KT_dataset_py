import argparse

import sys

import math

import numpy as np

import random

import matplotlib.pyplot as plt

from tqdm import tqdm



from multiprocessing import Pool
def logloss(true_label, predicted, eps=1e-15):

  p = np.clip(predicted, eps, 1 - eps)

  if true_label == 1:

    return -math.log(p)

  else:

    return -math.log(1 - p)
def get_loss_uniform(w, size):

    gt = (0, 1)

    true_labels = [random.choice(gt) for _ in range(size)]

    pred_labels = [random.uniform(0.5 - w, 0.5 + w) for _ in range(size)]

    res = 0

    for t, p in zip(true_labels, pred_labels):

        res += logloss(t, p)

    return res / size



def create_hist(w, size):

    trial = 1000

    with Pool(processes=4) as pool:

        ret = pool.starmap(get_loss_uniform, [(w, size) for _ in range(trial)])

    plt.hist(ret, bins=50)

    plt.savefig("hist_Range{}_{}_uniform.png".format(w, size))

    plt.title("hist_Range{}_{}_uniform".format(w, size))

    plt.show()

    plt.close("all")

    

size = 4000

create_hist(0.5, size)

create_hist(0.1, size)

create_hist(0.01, size)
def get_extreme_score(w, size, b="best"):

    gt = 1

    if b!="best":

        gt = 0

    true_labels = [gt for _ in range(size)]

    pred_labels = [0.5 + w for _ in range(size)]

    res = 0

    for t, p in zip(true_labels, pred_labels):

        res += logloss(t, p)

    return res / size



ws = np.arange(0.01, 0.5, 0.01)

bests = []

worsts = []

for w in ws:

    bests.append(get_extreme_score(w, size, b="best"))

    worsts.append(get_extreme_score(w, size, b="worst"))

# Plot both

plt.title("Best and Worst")

plt.xlabel("w (range of prediction from 0.5)")

plt.ylabel("logloss")

plt.plot(list(ws), bests, label="best")

plt.plot(list(ws), (np.array(bests) + np.array(worsts))/2, label="middle")

plt.plot(list(ws), worsts, label="worst")

plt.legend()

plt.show()

plt.close("all")



# Plot Best only

plt.title("Best only")

plt.xlabel("w (range of prediction from 0.5)")

plt.ylabel("logloss")

plt.plot(list(ws), bests, label="best")

plt.show()



# Plot Middle only

plt.title("Middle only")

plt.xlabel("w (range of prediction from 0.5)")

plt.ylabel("logloss")

plt.plot(list(ws), (np.array(bests) + np.array(worsts))/2, label="middle")

plt.show()