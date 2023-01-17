# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Read and present our input and label

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

x_train = train.drop("label",axis = 1)

y_train = train.label
type(x_train)
y_train[643]
img = img.reshape(28,28,1)
import matplotlib.pyplot as plt

img = x_train.iloc[642].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title("ONE")

plt.axis("off")

plt.show()
zero = x_train.iloc[1]

one = x_train.iloc[0]

two = x_train.iloc[643]

plt.scatter(range(0,784),zero,color = "red")

plt.scatter(range(0,784),one,color = "blue")

plt.scatter(range(0,784),two,color = "green")
tr_encode =  y_train[5000:10000]

te_encode = y_train[10000:10200]

Ytr = tf.one_hot(tr_encode,10)

Yte = tf.one_hot(te_encode,10)
Xtr = x_train.iloc[5000:10000,:]

Xte = x_train.iloc[10000:10200,:]

Xtr = np.array(Xtr)

Xte = np.array(Xte)

xtr = tf.placeholder("float", [None, 784])

xte = tf.placeholder("float", [784])



# Nearest Neighbor calculation using L1 Distance

# Calculate L1 Distance

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

# Prediction: Get min distance index (Nearest neighbor)

pred = tf.arg_min(distance, 0)



accuracy = 0



# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()



# Start training

with tf.Session() as sess:



    # Run the initializer

    sess.run(init)



    # loop over test data

    for i in range(len(Xte)):

        # Get nearest neighbor

        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})

        # Get nearest neighbor class label and compare it to its true label

        print("Test", i, "Prediction:", np.argmax(sess.run(Ytr[nn_index])),

            "True Class:", np.argmax(sess.run(Yte[i])))

        # Calculate accuracy

        if np.argmax(sess.run(Ytr[nn_index])) == np.argmax(sess.run(Yte[i])):

            accuracy += 100/len(Xte)

    print("Done!")

    print("Accuracy:", accuracy)