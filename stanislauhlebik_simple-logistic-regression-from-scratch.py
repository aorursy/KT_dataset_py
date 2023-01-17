import tensorflow as tf

!pip install tensorflow_datasets

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
ds, info = tfds.load('iris', split='train', with_info=True)
print('Total classes %d, %s' % (info.features['label'].num_classes, info.features['label'].names))

arr = []

for i in tfds.as_numpy(ds):

    arr.append(i['features'])

print('Total samples: %d' % len(arr))



count_by_class = {0:0, 1:0, 2:0}

labels = []

for i in tfds.as_numpy(ds):

    count_by_class[i['label']] += 1

    labels.append(i['label'])



for cl, count in count_by_class.items():

    print("Class #%d, samples: %d" % (cl, count))
df = pd.DataFrame(arr)

df.describe()
plt.figure(figsize=(10, 10))

for i in range(0, 4):   

    plt.subplot(2, 2, i + 1)

    plt.title("Feature #%d" % i)

    plt.hist(df[i])
labels = np.array(labels)

feature_num = 4

def plot_by_features(first_feature, second_feature, subplot_num):

    first_class = df[labels == 0]

    second_class = df[labels == 1]

    third_class = df[labels == 2]



    plt.subplot(feature_num, feature_num, subplot_num)

    plt.title('%d %d ' % (first_feature, second_feature))

    plt.plot(first_class[first_feature], first_class[second_feature], 'go')

    plt.plot(second_class[first_feature], second_class[second_feature], 'ro')

    plt.plot(third_class[first_feature], third_class[second_feature], 'bo')



plt.figure(figsize=(15, 15))

num = 0

for i in range(0, feature_num):

    for j in range(0, feature_num):

        num += 1

        plot_by_features(i, j, num)



plt.suptitle('2D plotting')
from mpl_toolkits.mplot3d import Axes3D



def draw_3d(first_feature, second_feature, third_feature, subplotnum):

    first_class = df[labels == 0]

    second_class = df[labels == 1]

    third_class = df[labels == 2]



    ax = fig.add_subplot(2, 1, subplotnum, projection='3d')

    ax.scatter(first_class[first_feature], first_class[second_feature], first_class[third_feature], marker='o')

    ax.scatter(second_class[first_feature], second_class[second_feature], second_class[third_feature], marker='^')

    ax.scatter(third_class[first_feature], third_class[second_feature], third_class[third_feature], marker='v')



fig = plt.figure(figsize=(10, 10))

draw_3d(0, 1, 2, 1)

draw_3d(0, 1, 3, 2)
selected_class = 0



selected = df[labels == selected_class]

notselected = df[labels != selected_class]

plt.plot(selected.iloc[:, [0]], selected.iloc[:, [3]], 'ro')

plt.plot(notselected.iloc[:, [0]], notselected.iloc[:, [3]], 'go')
def sigmoid(a):

    return 1. / (1. + np.exp(-1 * a))



def compute_class_probability(sample, w, b):

    a = 0

    for i in range(len(sample)):

        a += sample[i] * w[i]

    a += b

    return sigmoid(a)



def update_weights(batch, labels, w, b, learning_rate):

    """Stochastic gradient descent

    Based on https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc

    

    Returns: updated parameters `w` and `b`

    """

    loss = 0

    for (sample_num, (_, sample)) in enumerate(batch.iterrows()):

        sample = sample.to_numpy()

        y = labels[sample_num]



        a = compute_class_probability(sample, w, b)



        if y == 0:

            loss -= np.log(1 - a)

        else:

            loss -= np.log(a)



        for i in range(len(sample)):

            w[i] -= learning_rate * (a - y) * sample[i]

        b -= learning_rate * (a - y)



    return ((w, b), loss)
def compute_x2(w1, w2, b, x1):

    return (-1 * w1 * x1 - b) / w2



def plot_line(w1, w2, b):

    first = [0, 8]

    second = [compute_x2(w1, w2, b, 0), compute_x2(w1, w2, b, 8)]

    

    plt.plot(first, second) 



def train_and_plot(ds, labels, epochs=3, learning_rate=0.01):

    w = [0, 0]

    b = 0



    for round_num in range(4):

        for i in range(epochs):

            (w, b), loss = update_weights(ds, labels, w, b, learning_rate)

        print("loss %f" % loss)



        plt.subplot(2, 2, round_num + 1)

        plt.title('%d epochs' % ((round_num + 1) * epochs))

        selected = ds[labels == 1]

        notselected = ds[labels != 1]

        plt.plot(selected.iloc[:, [0]], selected.iloc[:, [1]], 'go')

        plt.plot(notselected.iloc[:, [0]], notselected.iloc[:, [1]], 'ro')

        plot_line(w[0], w[1], b)



    return (w, b)
plt.figure(figsize=(10, 10))



columns = [0, 1]

train_and_plot(df.iloc[:, columns], (labels == 0).astype(int))

plt.suptitle('Features %s' % columns)
plt.figure(figsize=(10, 10))



columns = [0, 2]

train_and_plot(df.iloc[:, [0, 2]], (labels == 0).astype(int))

plt.suptitle('Features %s' % columns)
plt.figure(figsize=(10, 10))



columns = [0, 3]

train_and_plot(df.iloc[:, columns], (labels == 0).astype(int))

plt.suptitle('Features %s' % columns)
def train_and_evaluate(df, labels):

    w = [0, 0, 0, 0]

    b = 0

    iterations = 10

    for round_num in range(4):

        for i in range(iterations):

            (w, b), loss = update_weights(df, labels, w, b, learning_rate=0.01)

        print("loss %f" % loss)



    correct = 0

    for sample_num, (_, sample) in enumerate(df.iterrows()):

        predicted = int(compute_class_probability(sample, w, b) > 0.5)

        actual = labels[sample_num]

        if actual == predicted:

            correct += 1



    print("Accuracy %f" % (float(correct) / len(df) ))

    print((w, b))



train_and_evaluate(df, (labels == 0).astype(int))
second_and_third_df = df[labels != 0]

second_and_third_labels = labels[labels != 0]

second_and_third_labels = (second_and_third_labels == 1).astype(int)



train_and_evaluate(second_and_third_df, second_and_third_labels)