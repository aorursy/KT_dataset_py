import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



import os

print(os.listdir("../input"))

mnist_train = pd.read_csv("../input/train.csv")

mnist_test = pd.read_csv("../input/test.csv")
mnist_train.head(5)
mnist_test
train_images = mnist_train.values[:, 1:]

train_images.shape
mnist_test.head(5)

mnist_test.shape
test_images = mnist_test.values[:]

test_images.shape
train_label = mnist_train["label"].values

train_label
nn_classifier = MLPClassifier(hidden_layer_sizes=(500,), 

                              activation='logistic',

                              n_iter_no_change = 300,

                              learning_rate_init=0.005, 

                              max_iter=500, 

                              solver='sgd', 

                              learning_rate="constant", verbose=True )
nn_classifier = nn_classifier.fit(train_images, train_label)
all_predictions = nn_classifier.predict(test_images)
all_predictions
index = []

for i in range(28001):

    index.append(i)
my_submission = pd.DataFrame({'ImageId': index[1:], 'Label': all_predictions})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)

my_submission
test_images = test_images.reshape(28000, 28, 28)

test_images.shape
plt.imshow(test_images[27998], cmap="Greys")