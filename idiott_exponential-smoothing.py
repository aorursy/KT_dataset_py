import numpy as np

prefix = np.loadtxt("../input/data_with_prefix.csv", delimiter=",", usecols=(9, 10, 11, 12, 13, 14, 15, 16), unpack=True)

from matplotlib import pyplot as plt

import math



def to_difference(x):

    a = x.copy()

    b = x.copy()

    a = np.concatenate((a, np.array([0])))

    b = np.concatenate((np.array([0]), b))

    out = a - b

    out = out[0:-1]

    return out



def to_prefix_sum(x):

    out = np.zeros(x.shape[0] + 1)

    for i in range(1, out.shape[0]):

        out[i] = out[i - 1] + x[i - 1]

    return out

        

def exp_smoothing(Y, alpha):

    deltaY = to_difference(Y)

    Y_p = np.zeros_like(Y)

    deltaY_p = np.zeros_like(deltaY)

    assert Y.shape == deltaY.shape

    

    for i in range(Y.size - 1):

        deltaY_p[i + 1] = alpha * deltaY[i] + (1 - alpha) * deltaY_p[i]

        Y_p[i + 1] = deltaY_p[i + 1] + Y[i]

    

    return Y_p







threshold = 4.5

alpha = 1.7

sum_ = 0

plt.figure(figsize=(20,10))

print("when alpha =", alpha)

for i in range(8):

    plt.title("Region 1~8")

    Y = prefix[i]

    Y_p = exp_smoothing(prefix[i], alpha)

    diff = np.abs(Y - Y_p)

    points_X = np.arange(diff.size)[diff > threshold]

    points_Y = Y_p[points_X]

    X = np.arange(0, 100)

    plt.plot(X, Y, label='R' + str(i + 1) + ' truth')

    plt.plot(X, Y_p, label='R' + str(i + 1) + ' predicted')

    plt.plot(points_X, points_Y, 'o')

    plt.legend()

    print("std diff of R" + str(i + 1) + " =", np.sqrt(np.sum(diff * diff) / diff.size))

    sum_ += np.sum(diff * diff) / diff.size

    #print("critical points of R" + str(i + 1) + " =", points_X)

print(sum_)
region = 0

plt.figure(figsize=(20,10))

for i in range(region, region + 1):

    plt.title("Region " + str(i + 1))

    Y = prefix[i]

    Y_p = exp_smoothing(prefix[i], alpha)

    diff = np.abs(Y - Y_p)

    index = np.argsort(-diff)

    points_X = np.arange(diff.size)[diff > threshold]

    points_Y = Y_p[points_X]

    X = np.arange(0, 100)

    plt.plot(X, Y, label='R' + str(i + 1) + ' truth')

    plt.plot(X, Y_p, label='R' + str(i + 1) + ' predicted')

    plt.plot(points_X, points_Y, 'o')

    plt.legend()

    print("critical points of R" + str(i + 1) + " =", points_X)

region += 1
plt.figure(figsize=(20,10))

for i in range(region, region + 1):

    plt.title("Region " + str(i + 1))

    Y = prefix[i]

    Y_p = exp_smoothing(prefix[i], alpha)

    diff = np.abs(Y - Y_p)

    index = np.argsort(-diff)

    points_X = np.arange(diff.size)[diff > threshold]

    points_Y = Y_p[points_X]

    X = np.arange(0, 100)

    plt.plot(X, Y, label='R' + str(i + 1) + ' truth')

    plt.plot(X, Y_p, label='R' + str(i + 1) + ' predicted')

    plt.plot(points_X, points_Y, 'o')

    plt.legend()

    print("critical points of R" + str(i + 1) + " =", points_X)

region += 1
plt.figure(figsize=(20,10))

for i in range(region, region + 1):

    plt.title("Region " + str(i + 1))

    Y = prefix[i]

    Y_p = exp_smoothing(prefix[i], alpha)

    diff = np.abs(Y - Y_p)

    index = np.argsort(-diff)

    points_X = np.arange(diff.size)[diff > threshold]

    points_Y = Y_p[points_X]

    X = np.arange(0, 100)

    plt.plot(X, Y, label='R' + str(i + 1) + ' truth')

    plt.plot(X, Y_p, label='R' + str(i + 1) + ' predicted')

    plt.plot(points_X, points_Y, 'o')

    plt.legend()

    print("critical points of R" + str(i + 1) + " =", points_X)

region += 1
plt.figure(figsize=(20,10))

for i in range(region, region + 1):

    plt.title("Region " + str(i + 1))

    Y = prefix[i]

    Y_p = exp_smoothing(prefix[i], alpha)

    diff = np.abs(Y - Y_p)

    index = np.argsort(-diff)

    points_X = np.arange(diff.size)[diff > threshold]

    points_Y = Y_p[points_X]

    X = np.arange(0, 100)

    plt.plot(X, Y, label='R' + str(i + 1) + ' truth')

    plt.plot(X, Y_p, label='R' + str(i + 1) + ' predicted')

    plt.plot(points_X, points_Y, 'o')

    plt.legend()

    print("critical points of R" + str(i + 1) + " =", points_X)

region += 1
plt.figure(figsize=(20,10))

for i in range(region, region + 1):

    plt.title("Region " + str(i + 1))

    Y = prefix[i]

    Y_p = exp_smoothing(prefix[i], alpha)

    diff = np.abs(Y - Y_p)

    index = np.argsort(-diff)

    points_X = np.arange(diff.size)[diff > threshold]

    points_Y = Y_p[points_X]

    X = np.arange(0, 100)

    plt.plot(X, Y, label='R' + str(i + 1) + ' truth')

    plt.plot(X, Y_p, label='R' + str(i + 1) + ' predicted')

    plt.plot(points_X, points_Y, 'o')

    plt.legend()

    print("critical points of R" + str(i + 1) + " =", points_X)

region += 1
plt.figure(figsize=(20,10))

for i in range(region, region + 1):

    plt.title("Region " + str(i + 1))

    Y = prefix[i]

    Y_p = exp_smoothing(prefix[i], alpha)

    diff = np.abs(Y - Y_p)

    index = np.argsort(-diff)

    points_X = np.arange(diff.size)[diff > threshold]

    points_Y = Y_p[points_X]

    X = np.arange(0, 100)

    plt.plot(X, Y, label='R' + str(i + 1) + ' truth')

    plt.plot(X, Y_p, label='R' + str(i + 1) + ' predicted')

    plt.plot(points_X, points_Y, 'o')

    plt.legend()

    print("critical points of R" + str(i + 1) + " =", points_X)

region += 1
plt.figure(figsize=(20,10))

for i in range(region, region + 1):

    plt.title("Region " + str(i + 1))

    Y = prefix[i]

    Y_p = exp_smoothing(prefix[i], alpha)

    diff = np.abs(Y - Y_p)

    index = np.argsort(-diff)

    points_X = np.arange(diff.size)[diff > threshold]

    points_Y = Y_p[points_X]

    X = np.arange(0, 100)

    plt.plot(X, Y, label='R' + str(i + 1) + ' truth')

    plt.plot(X, Y_p, label='R' + str(i + 1) + ' predicted')

    plt.plot(points_X, points_Y, 'o')

    plt.legend()

    print("critical points of R" + str(i + 1) + " =", points_X)

region += 1
plt.figure(figsize=(20,10))

for i in range(region, region + 1):

    plt.title("Region " + str(i + 1))

    Y = prefix[i]

    Y_p = exp_smoothing(prefix[i], alpha)

    diff = np.abs(Y - Y_p)

    index = np.argsort(-diff)

    points_X = np.arange(diff.size)[diff > threshold]

    points_Y = Y_p[points_X]

    X = np.arange(0, 100)

    plt.plot(X, Y, label='R' + str(i + 1) + ' truth')

    plt.plot(X, Y_p, label='R' + str(i + 1) + ' predicted')

    plt.plot(points_X, points_Y, 'o')

    plt.legend()

    print("critical points of R" + str(i + 1) + " =", points_X)

region += 1