import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import random
df = pd.read_csv("../input/Ads_Optimisation.csv")
df.head()
df.shape
N = 10000

d = 10

total = 0

selected = []

for n in range(0,N):

    ad = random.randrange(d)

    selected.append(ad)

    prize = df.values[n,ad]

    total = total + prize
total
plt.hist(selected)

plt.show()
import math

N = 10000

d = 10

awards = [0] * d

total = 0

clicks = [0] * d

selected = []

for n in range(1,N):

    ad = 0

    max_ucb = 0

    for i in range(0,d):

        if(clicks[i]>0):

            mean = awards[i] / clicks[i]

            delta = math.sqrt(3/2*math.log(n)/clicks[i])

            ucb = mean + delta

        else:

            ucb = N*10

        if max_ucb < ucb:

            max_ucb = ucb

            ad = i

    selected.append(ad)

    clicks[ad] = clicks[ad] + 1

    prize = df.values[n,ad]

    awards[ad] = awards[ad] + prize

    total = total + prize

print("Total Prize : ", total)
plt.hist(selected)

plt.show()
import math

N = 10000

d = 10

awards = [0] * d

total = 0

clicks = [0] * d

selected = []

ones = [0] * d

zeros = [0] * d

for n in range(1,N):

    ad = 0

    max_th = 0

    for i in range(0,d):

        randomBeta = random.betavariate(ones[i] + 1, zeros[i]+1)

        if randomBeta > max_th:

            max_th = randomBeta

            ad = i

        selected.append(ad)

        prize = df.values[n,ad]

        if prize == 1:

            ones[ad] = ones[ad] + 1

        else:

            zeros[ad] = zeros[ad] + 1

        total = total + prize

print("toplam odul : ", total)
plt.hist(selected)

plt.show()