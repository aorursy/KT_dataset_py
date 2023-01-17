# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
filename = '../input/mnist-data/MNIST.csv'
dataset = pd.read_csv(filename)

dataset
train_dataset = dataset.sample(frac=0.8,random_state=200)
test_dataset = dataset.drop(train_dataset.index)

#forming mini batches for training
import math
def create_mini_batches(X, Y, mini_batch_size = 128, seed = 0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
import tensorflow as tf 
label_train = train_dataset['label'].values
train_dataset = train_dataset.drop(['label'],axis=1)
train_x = train_dataset.as_matrix()/255
label_test = test_dataset['label'].values
test_dataset = test_dataset.drop(['label'],axis=1)
test_x = test_dataset.as_matrix()/255
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
train_y = enc.fit_transform(label_train.reshape(-1,1),)

import matplotlib.pyplot as plt
from scipy.misc import imread,imresize
plt.imshow(train_x[1].reshape(28,28),cmap = 'gray')
def rbf(x,y,sigma):
    z = -np.sum((x-y)*(x-y))/(2*sigma)
    return np.exp(z)
#cluster_centre = train_x[np.random.choice(33600,784,replace=False)]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=784, random_state=0).fit(train_x)
cluster_centre = kmeans.cluster_centers_
# checking the distance between training data samples with 784 cluster centres
train_rbf = np.zeros([33600,784])
for i in range(33600):
    for j in range(784):
        train_rbf[i,j] = rbf(train_x[i],cluster_centre[j],100)
    print(i)
train_rbf
test_rbf = np.zeros([8400,784])
for i in range(8400):
    for j in range(784):
        test_rbf[i,j] = rbf(test_x[i],cluster_centre[j],100)
    print(i)
test_rbf
image = []
image.append(imread('../input/digiitnumber/digit1.jpg'))
image.append(imread('../input/digiitnumber1/digit2.jpg'))
image.append(imread('../input/digiitnumber1/digit3.jpg'))
for i in range(3):
    image[i] = imresize(image[i],(28,28))
    image[i] = (255 - np.mean(image[i],axis=2))/255

plt.subplot(131)
plt.imshow(image[0],cmap='gray')
plt.subplot(132)
plt.imshow(image[1],cmap='gray')
plt.subplot(133)
plt.imshow(image[2],cmap='gray')
plt.show()
# Initialize placeholders 
X = tf.placeholder(dtype = tf.float32, shape = [None,784])
y = tf.placeholder(dtype = tf.int32, shape = [None,10])


# the output layer or NN
output_layer = tf.contrib.layers.fully_connected(X,10,tf.nn.softmax)
pred_layer = tf.argmax(output_layer,1)

# cost function 
cost_function = tf.reduce_mean(tf.losses.softmax_cross_entropy(y,output_layer))

# optimizer
optimizer_function = tf.train.AdamOptimizer().minimize(cost_function)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_cost = 0
    
    minibatches = create_mini_batches(train_rbf,train_y)
    for epochs in range(150):
        for minibatch in minibatches:
            (mini_x,mini_y) = minibatch
            mini_cost,_ = sess.run([cost_function,optimizer_function],feed_dict={X:mini_x,y:mini_y})
            total_cost = total_cost+mini_cost
        print('Cost/epoch value for epoch  ',epochs,' : ',total_cost/((epochs+1)*len(minibatches)))
    
    predicted_train = pred_layer.eval({X:train_rbf})
    train_acc = np.sum(predicted_train==label_train)/label_train.shape[0]
    print('training set accuracy accuracy : '+str(train_acc))
    
    predicted_test = pred_layer.eval({X:test_rbf})
    test_acc = np.sum(predicted_test==label_test)/label_test.shape[0]
    print('test set accuracy : '+str(test_acc))
    
    
    predicted_values = []
    for i in range(3):
        predicted_values.append(pred_layer.eval({X:image[i].reshape(1,-1)}))
    print(predicted_values)

