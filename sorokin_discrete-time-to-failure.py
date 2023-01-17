import os

import csv

import numpy as np

from tqdm import tqdm

from collections import Counter

from operator import itemgetter

import matplotlib.pyplot as plt

%matplotlib inline
N = 629145480

y = np.empty(N, dtype=np.float32)



with open('../input/train.csv') as f:

    reader = csv.reader(f)

    for header in reader:

        break

    for i, row in enumerate(tqdm(reader, total=N)):

        y[i] = float(row[1])
deltas = y[1:] - y[:-1]

episodes = {0, len(y)}

episodes.update(np.arange(0, len(y) - 1)[deltas > 0] + 1)

episodes = sorted(list(episodes))

episodes = list(zip(episodes[:-1], episodes[1:]))

print('Episodes:', len(episodes))
deltas = []

rewards = [] # non-zero deltas



for start, end in episodes:

    t = y[start:end]

    d = t[1:] - t[:-1]

    d = np.round(d, 10)

    deltas.append(d)

    rewards.extend(d[d != 0])



counts = Counter(rewards)



classes = dict()

for value, numbers in sorted(counts.items(), key=itemgetter(0)):

    print('%2d % .10f %7d' % (len(classes), value, numbers))

    classes[value] = len(classes)
targets = [] # labels will store in a reverse order

for i, d in enumerate(deltas[::-1]):

    c = 0

    for r in tqdm(d[::-1], desc='Episode %d' % i):

        if r != 0:

            c = classes[r] if r in classes else 0

        targets.append(c)

    targets.append(c)

# reverse again to match an original order

targets = np.array(targets[::-1], dtype=np.int8)
counts = np.bincount(targets, minlength=52)

plt.bar(range(52), counts)

plt.xlabel("label")

plt.ylabel("number of samples")

plt.show()
n_rewards = 1

mapping = {-1: 0}



for i, c in enumerate(counts):

    if c < 10_000_000:

        mapping[i] = 0

        continue

    print(n_rewards, i, c)

    mapping[i] = n_rewards

    n_rewards += 1



targets_balanced = targets.copy()

for k, v in mapping.items():

    targets_balanced[targets == k] = v
x = np.arange(0, len(y))



_x = x[::1000]

_y = y[::1000]

_c = targets_balanced[::1000]



sc = plt.scatter(_x, _y, c=_c, s=50, cmap='jet')

plt.colorbar(sc)

plt.xlabel("time")

plt.ylabel("time_to_failure")

plt.show()