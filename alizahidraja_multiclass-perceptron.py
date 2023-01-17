# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install python-mnist
import numpy as np

from mnist.loader import MNIST
"""Activation Functions"""



def signum(x):

    x[x > 0] = 1

    x[x <= 0] = -1



    return x





# --------------------------------------------------------



"""Predicts the labels by choosing the label of the classifier with highest confidence(probability)"""

def predict(all_weights, images, labels, size):

    images = np.hstack((np.ones((size, 1)), images))



    predicted_labels = np.dot(all_weights, images.T)



    # signum activation function

    predicted_labels = signum(predicted_labels)



    predicted_labels = np.argmax(predicted_labels, axis=0).T



    corr = 0

    label_tot = {}

    label_corr = {}

    label_acc = {}

    L = np.unique(labels)

    for i in L:

        label_tot[i] = 0

        label_corr[i] = 0

        label_acc[i] = 0



    for i in range(len(labels)):

        label_tot[labels[i]] += 1

        if predicted_labels[i] == labels[i]:

            corr += 1

            label_corr[labels[i]] += 1



    for i in L:

        label_acc[i] = label_corr[i]/label_tot[i] * 100



    return corr/len(labels)*100, label_acc





# --------------------------------------------------------

def learning(train_images, train_labels, weights):

    epochs_values = []

    error_values = []



    for k in range(epochs):

        missclassified = 0



        for t, l in zip(train_images, train_labels):

            h = np.dot(t, weights)



            h = signum(h)



            if h[0] != l[0]:

                missclassified += 1



            gradient = t * (h - l)



            # reshape gradient

            gradient = gradient.reshape(gradient.shape[0], 1)



            weights = weights - (gradient * alpha)



        error_values.append(missclassified / training_size)

        epochs_values.append(k)



    return weights





"""Find optimal weights for each logistic binary classifier"""





def train(train_images, train_labels):

    # add 1's as x0

    train_images = np.hstack((np.ones((training_size, 1)), train_images))



    # add w0 as 0 initially

    all_weights = np.zeros((labels, train_images.shape[1]))



    train_labels = train_labels.reshape((training_size, 1))



    train_labels_copy = np.copy(train_labels)



    for j in range(labels):



        print("Training Classifier: ", j+1)



        train_labels = np.copy(train_labels_copy)



        # initialize all weights to zero

        weights = np.zeros((train_images.shape[1], 1))



        for k in range(training_size):

            if train_labels[k, 0] == j:

                train_labels[k, 0] = 1

            else:

                train_labels[k, 0] = -1



        weights = learning(train_images, train_labels, weights)



        all_weights[j, :] = weights.T



    return all_weights



# Global Variables

training_size = 60000

testing_size = 10000





alpha = 0.01

iterations = 2000  # epochs for batch mode gradient descent



epochs = 15



labels = 10



# load data

data = MNIST('/kaggle/input/mnist-samples/samples/')



train_images, train_labels = data.load_training()

test_images, test_labels = data.load_testing()



train_images = np.array(train_images[:training_size])

train_labels = np.array(train_labels[:training_size], dtype=np.int32)



test_images = np.array(test_images[:testing_size])

test_labels = np.array(test_labels[:testing_size], dtype=np.int32)



"""Rescaling Data"""

train_images = train_images / 255

test_images = test_images / 255
""" learning a perceptron with perceptron learning rule """

print("Running Experiment using Perceptron Learning Rule for Thresholded Unit")



print("Training")

all_weights = train(train_images, train_labels)

print("Weights Learned!")
train_acc = predict(all_weights, train_images, train_labels, training_size)

print('Training Accuracy:', train_acc[0], '%\n')

train_best = max(train_acc[1], key=train_acc[1].get)

train_worst = min(train_acc[1], key=train_acc[1].get)

print('Best Digit in Train', train_best,', with Accuracy:', train_acc[1][train_best], '%')

print('Worst Digit in Train', train_worst,', with Accuracy:', train_acc[1][train_worst], '%')
test_acc = predict(all_weights, test_images, test_labels, testing_size)

print('Test Accuracy:', test_acc[0], '%\n')

test_best = max(test_acc[1], key=test_acc[1].get)

test_worst = min(test_acc[1], key=test_acc[1].get)

print('Best Digit in Train', test_best,', with Accuracy:', test_acc[1][test_best], '%')

print('Worst Digit in Train', test_worst,', with Accuracy:', test_acc[1][test_worst], '%')
