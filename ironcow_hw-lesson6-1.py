import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

# print(os.listdir("../input"))

test = pd.read_csv("../input/mnist_test.csv")

test

list(test.iloc[0][1:])

x_test = []

y_test = []

for i in range(test.shape[0]):

    x_test.append(list(test.iloc[i][1:]))

    y_test.append(list(test.iloc[i][:1]))

    

x_test = np.array(x_test)

y_test = np.array(y_test)
train = pd.read_csv("../input/mnist_train.csv")

# list(test.iloc[0][1:])

x_train = []

y_train = []

for i in range(train.shape[0]):

    x_train.append(list(train.iloc[i][1:]))

    y_train.append(list(train.iloc[i][:1]))
x_train = np.array(x_train)

y_train = np.array(y_train)

x_train.shape
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')
my_1000_digits_X = x_test[:1000]

my_1000_digits_Y = y_test[:1000]
my_answer = [] #[3, 4, 5, 0, 0, 3, 1, ...]

def knn_estimator(x_test, y_test, x_train, y_train, k):

    for ex in x_test:

        euclids = {} # {eucl: [index, y_tr]}

        for j, my in enumerate(x_train):

            euclids[np.dot((ex - my),(ex - my))**0.5] = [j, int(y_train[j])]

        number_count = {} #[number(0-9): count]    

#         print(sorted(euclids)[:k])

        for t in sorted(euclids)[:k]:

            if euclids[t][1] not in number_count:

                number_count[euclids[t][1]] = 0

            number_count[euclids[t][1]] += 1

#         print(len(number_count))

        count = -1

        choice = None

#         print(len(number_count))

        for n in number_count:

            if number_count[n] > count:

                count = number_count[n]

                choice = n

        my_answer.append(choice)

#         print(len(my_answer))

    total_correct = 0

    for y in range(len(x_test)):

        if my_answer[y] == y_test[y]:

            total_correct += 1 

    return total_correct

%%time

my_answer = [] #[3, 4, 5, 0, 0, 3, 1, ...]

print(f"Accuracy with {10} nearest neighbours is {knn_estimator(my_1000_digits_X, my_1000_digits_Y, x_train, y_train, 10)} out of {len(my_1000_digits_X)}")
%%time

my_answer = [] #[3, 4, 5, 0, 0, 3, 1, ...]

print(f"Accuracy with {50} nearest neighbours is {knn_estimator(my_1000_digits_X, my_1000_digits_Y, x_train, y_train, 50)} out of {len(my_1000_digits_X)}")