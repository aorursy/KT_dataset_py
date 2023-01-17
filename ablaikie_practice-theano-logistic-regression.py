# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline





import theano

import theano.tensor as T

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read training data from CSV file 

data = pd.read_csv('../input/train.csv')

data.shape
# First column is labels, last 784 are flattened 28x28 pixel images

images = data.iloc[:,1:].values

labels_flat = data.iloc[:,0].values



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)



m = images.shape[0] # Number of training examples

n = data.shape[1] # Number of variables
state = shared(0)

inc = T.iscalar('inc')

accumulator = function([inc], state, updates=[(state, state+inc)])
# display image

def display(img):

    

    # (784) => (28,28)

    one_image = img.reshape(28, 28)

    plt.imshow(one_image, cmap='gray')
# output image

imageToDisplay = 999

display(images[imageToDisplay])

labels_flat[imageToDisplay]
# There are 10 unique class labels we need to predict

labels_count = np.unique(labels_flat).shape[0]



print('labels_count => {0}'.format(labels_count))
# convert class labels from scalars to one-hot vectors

# 0 => [1 0 0 0 0 0 0 0 0 0]

# 1 => [0 1 0 0 0 0 0 0 0 0]

# ...

# 9 => [0 0 0 0 0 0 0 0 0 1]



def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot



labels = dense_to_one_hot(labels_flat, labels_count)

labels = labels.astype(np.uint8)



print('labels({0[0]},{0[1]})'.format(labels.shape))
print(labels[48])

print(labels_flat[48])
w = np.zeros((784,10))

print(w.shape)

print(images.shape)

print(np.dot(images, w).shape)
np.argmax([0,0,0,1,0])
# Machine learning parameters

lam = 0.01 # Regularization parameter

alpha = 0.1 # Gradient descent step function

training_steps = 10000 # Number of training steps
# Declare Theano symbolic variables

x = T.dmatrix("x")

y = T.dvector("y")
# w is shared so it can keep its values between training iterations

w = theano.shared(np.random.rand(784, 10), name="w")
# Construct Theano expression graph

p_1 = 1 / (1 + T.exp(-T.dot(x,w))) # Probability that target = 1

prediction = p_1.argmax(axis = 1) # The prediction threshold

xent = -y * T.log(p_1) - (1 - y) * T.log(1-p_1) # Cross-entropy loss function

cost = xent.mean(axis = 0)

gw0 = T.grad(cost[0], w) 
# Compile

train = theano.function(

            inputs = [x,y],

            outputs = [prediction, cost],

            updates = [(w, w - alpha * gw0)],

            )

predict = theano.function(inputs=[x], outputs=prediction)
# Train

for i in range(training_steps):

    pred, err = train(images, labels)
predict(images)
state = theano.shared(0)

inc = T.iscalar('inc')

accumulator = theano.function([inc], state, updates=[(state, state+inc)])

accumulator(1)
accumulator(3)
accumulator(2)