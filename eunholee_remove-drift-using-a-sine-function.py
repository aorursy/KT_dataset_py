import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.optimize import curve_fit

import os



df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

df_test  = pd.read_csv("../input/liverpool-ion-switching/test.csv")
def make_batches(df, dataset="train"):

    batches = []

    batch_size = [500000, 100000]

    if dataset == "train":

        for idx in range(10):

            batches.append(df[idx * batch_size[0]: (idx + 1) * batch_size[0]])

    else:

        for idx in range(10):

            batches.append(df[idx * batch_size[1]: (idx + 1) * batch_size[1]])

        for idx in range(2):

            base = 10 * batch_size[1]

            batches.append(df[base + idx * batch_size[0]: base + (idx + 1) * batch_size[0]])

    return batches



df_train = make_batches(df_train, "train")

df_test = make_batches(df_test, "test")
def plot_all(train, test, suffix=""):

    plt.figure(figsize=(25, 5))

    plt.subplot("211")

    plt.title("Train " + suffix)

    plt.ylabel("Signal")

    plt.xticks(np.arange(0, 501, 50))

    for x in train:

        plt.plot(x['time'], x['signal'], linewidth=.1)

    plt.grid()

    plt.subplot("212")

    plt.title("Test " + suffix)

    plt.ylabel("Signal")

    plt.xticks(np.arange(500, 701, 10))

    for x in test:

        plt.plot(x['time'], x['signal'], linewidth=.1)

    plt.grid()



plot_all(df_train, df_test, "Original")
linear_train_idx = [1]

linear_test_idx = [0, 1, 4, 6, 7, 8]



plt.figure(figsize=(30, 4))

plt.subplot("171")

plt.title("Train 1 (part)")

plt.ylabel("Signal", fontsize=8)

plt.plot(df_train[1]['time'][0:100000], df_train[1]['signal'][0:100000], linewidth=.1)

plt.grid()

plt.ylim([np.min(df_train[1]['signal'][0:100000]), np.min(df_train[1]['signal'][0:100000]) + 15])

for n, idx in enumerate(linear_test_idx):

    plt.subplot("17" + str(n + 2))

    plt.title("Test " + str(idx))

    plt.ylabel("Signal", fontsize=8)

    plt.ylim([np.min(df_test[idx]['signal']), np.min(df_test[idx]['signal']) + 15])

    plt.plot(df_test[idx]['time'], df_test[idx]['signal'], linewidth=.1)

    plt.grid()
def poly1(x, a, b):

    return a*(x - b)





def linear_drift_fit(data):

    x = data['time']

    y = data['signal']

    popt, _ = curve_fit(poly1, x, y)

    print(popt)

    return popt

    



linear_params = []

linear_params.append(linear_drift_fit(df_train[linear_train_idx[0]][0:100000]))

for idx in linear_test_idx:

    linear_params.append(linear_drift_fit(df_test[idx]))

    

plt.figure(figsize=(30, 4))

plt.subplot("171")

plt.title("Train 1 (part)")

plt.ylabel("Signal", fontsize=8)

plt.plot(df_train[1]['time'][0:100000], df_train[1]['signal'][0:100000], linewidth=.1)

plt.plot(df_train[1]['time'][0:100000], poly1(df_train[1]['time'][0:100000], *linear_params[0]), 'y')

plt.grid()

plt.ylim([np.min(df_train[1]['signal'][0:100000]), np.min(df_train[1]['signal'][0:100000]) + 15])

for n, idx in enumerate(linear_test_idx):

    plt.subplot("17" + str(n + 2))

    plt.title("Test " + str(idx))

    plt.ylabel("Signal", fontsize=8)

    plt.ylim([np.min(df_test[idx]['signal']), np.min(df_test[idx]['signal']) + 15])

    plt.plot(df_test[idx]['time'], df_test[idx]['signal'], linewidth=.1)

    plt.plot(df_test[idx]['time'], poly1(df_test[idx]['time'], *linear_params[1 + n]), 'y')

    plt.grid()
def linear_drift(x, x0):

    return 0.3 * (x - x0)





def remove_linear_drift(data, dataset="train"):

    if dataset == "train":

        data[1].loc[data[1].index[0:100000], 'signal'] = data[1].signal[0:100000].values - linear_drift(data[1].time[0:100000].values, data[1].time[0:1].values)

    else:

        for idx in linear_test_idx:

            data[idx].loc[data[idx].index[0:100000], 'signal'] = data[idx].signal[0:100000].values - linear_drift(data[idx].time[0:100000].values, data[idx].time[0:1].values)

            

    return data



df_train = remove_linear_drift(df_train, "train")

df_test = remove_linear_drift(df_test, "test")
plot_all(df_train, df_test, "- Linear Drift Removed")
parabola_train_idx = [6, 7, 8, 9]

parabola_test_idx = [10]



plt.figure(figsize=(30, 4))

for n, idx in enumerate(parabola_train_idx):

    plt.subplot("15" + str(n + 1))

    plt.title("Train " + str(idx))

    plt.ylabel("Signal", fontsize=8)

    plt.plot(df_train[idx]['time'], df_train[idx]['signal'], linewidth=.1)

    plt.grid()

    plt.ylim([np.min(df_train[idx]['signal']), np.min(df_train[idx]['signal']) + 18])

plt.subplot("155")

plt.title("Test 10")

plt.ylabel("Signal", fontsize=8)

plt.ylim([np.min(df_test[10]['signal']), np.min(df_test[10]['signal']) + 18])

plt.plot(df_test[10]['time'], df_test[10]['signal'], linewidth=.1)

plt.grid()
def my_sin(x, A, ph, d):

    frequency = 0.01

    omega = 2 * np.pi * frequency

    return A * np.sin(omega * x + ph) + d





def parabolic_drift_fit(data):

    x = data['time']

    y = data['signal']



    frequency = 0.01

    omega = 2 * np.pi * frequency

    M = np.array([[np.sin(omega * t), np.cos(omega * t), 1] for t in x])

    y = np.array(y).reshape(len(y), 1)



    (theta, _, _, _) = np.linalg.lstsq(M, y)

    

    A = np.sqrt(theta[0,0]**2 + theta[1,0]**2)

    ph = math.atan2(theta[1,0], theta[0,0])

    d = theta[2,0]



    popt = [A, ph, d]

    print(popt)

    return popt





parabola_params = []

for idx in parabola_train_idx:

    parabola_params.append(parabolic_drift_fit(df_train[idx]))

parabola_params.append(parabolic_drift_fit(df_test[parabola_test_idx[0]]))    

    

plt.figure(figsize=(30, 4))

for n, idx in enumerate(parabola_train_idx):

    plt.subplot("15" + str(n + 1))

    plt.title("Train " + str(idx))

    plt.ylabel("Signal", fontsize=8)

    plt.plot(df_train[idx]['time'], df_train[idx]['signal'], linewidth=.1)

    plt.plot(df_train[idx]['time'], my_sin(df_train[idx]['time'], *parabola_params[n]), 'y')

    plt.grid()

    plt.ylim([np.min(df_train[idx]['signal']), np.min(df_train[idx]['signal']) + 18])

plt.subplot("155")

plt.title("Test 10")

plt.ylabel("Signal", fontsize=8)

plt.ylim([np.min(df_test[10]['signal']), np.min(df_test[10]['signal']) + 18])

plt.plot(df_test[10]['time'], df_test[10]['signal'], linewidth=.1)

plt.plot(df_test[10]['time'], my_sin(df_test[10]['time'], *parabola_params[-1]), 'y')

plt.grid()

def parabolic_drift(x, t=0):

    f = 0.01

    omega = 2 * np.pi * f

    return 5 * np.sin(omega * x + t * np.pi)





def remove_parabolic_drift(data, dataset="train"):

    if dataset == "train":

        for idx in parabola_train_idx:

            data[idx].loc[data[idx].index[0:500000], 'signal'] = data[idx].signal[0:500000].values - parabolic_drift(data[idx].time[0:500000].values, (idx % 2))

    else:

        data[10].loc[data[10].index[0:500000], 'signal'] = data[10].signal[0:500000].values - parabolic_drift(data[10].time[0:500000].values)

            

    return data



df_train = remove_parabolic_drift(df_train, "train")

df_test = remove_parabolic_drift(df_test, "test")
plot_all(df_train, df_test, "- Without Drift")
def plot_dist(data, labels, m):

    plt.title("Signal Distribution Model " + str(m))

    for i, x in enumerate(data):

        x = x['signal']

        sns.distplot(x, label=labels[i], kde=True, bins=np.arange(np.min(x), np.max(x), 0.01))

#         sns.distplot(x, label=labels[i], kde=True)

    plt.xlabel("signal value")

    plt.ylabel("frequency")

    plt.legend(loc="best")    

    



M = [[df_train[0], df_train[1], df_test[0], df_test[3], df_test[8], df_test[10], df_test[11]],

     [df_train[2], df_train[6], df_test[4]],

     [df_train[3], df_train[7], df_test[1], df_test[9]],

     [df_train[4], df_train[9], df_test[5], df_test[7]],

     [df_train[5], df_train[8], df_test[2], df_test[6]]]

labels = [["train 0", "train 1 (line)", "test 0 (line)", "test 3", "test 8 (line)", "test 10 (sine)", "test 11"],

          ["train 2", "train 6 (sine)", "test 4 (line)"],

          ["train 3", "train 7 (sine)", "test 1 (line)", "test 9"],

          ["train 4", "train 9 (sine)", "test 5", "test 7 (line)"],

          ["train 5", "train 8 (sine)", "test 2", "test 6 (line)"]]



plt.figure(figsize=(25, 8))

for i in range(5):

    plt.subplot("15" + str(i + 1))

    plot_dist(M[i], labels[i], i)
df_train_clean = df_train[0]

df_test_clean = df_test[0]

for df in df_train[1:]:

    df_train_clean = pd.concat([df_train_clean, df], ignore_index=True)

for df in df_test[1:]:

    df_test_clean = pd.concat([df_test_clean, df], ignore_index=True)



df_train_clean.to_csv("train_wo_drift.csv", index=False, float_format="%.4f")

df_test_clean.to_csv("test_wo_drift.csv", index=False, float_format="%.4f")