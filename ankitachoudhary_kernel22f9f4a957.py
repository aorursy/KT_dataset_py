import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

df = pd.read_csv('../input/digitrecognizer/train.csv')
df.shape
df.head()
train_df = df.sample(frac=0.8,random_state=200)
test_df = df.drop(train_df.index)
train_df.head()
import math
def random_mini_batches(X, Y, mini_batch_size = 128, seed = 0):

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
label_train = train_df['label'].values
train_df = train_df.drop(['label'],axis=1)
train_x = train_df.as_matrix()/255
label_test = test_df['label'].values
test_df = test_df.drop(['label'],axis=1)
test_x = test_df.as_matrix()/255

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
train_y = enc.fit_transform(label_train.reshape(-1,1),)
train_y
import matplotlib.pyplot as plt
from scipy.misc import imread,imresize
img = []
img.append(imread('../input/digitnumbers/digit1.jpg'))
img.append(imread('../input/digitnumbers/digit2.jpg'))
img.append(imread('../input/digitnumbers/digit3.jpg'))
for i in range(3):
    img[i] = imresize(img[i],(28,28))
    img[i] = (255 - np.mean(img[i],axis=2))/255
plt.subplot(131)
plt.imshow(img[0],cmap='gray')
plt.subplot(132)
plt.imshow(img[1],cmap='gray')
plt.subplot(133)
plt.imshow(img[2],cmap='gray')
plt.show()
def rbf(x,y,sigma):
    z = -np.sum((x-y)*(x-y))/(2*sigma)
    return np.exp(z)

#cl_cen = train_x[np.random.choice(33600,784,replace=False)]
#cl_cen.shape
from sklearn.cluster import KMeans
kmean_cluster=KMeans(n_clusters=784, random_state=0).fit(train_x)
cluster_center=kmean_cluster.cluster_centers_
#rbf(train_x[0],cl_cen[0],100)

train_rbf = np.zeros([33600,784])
for i in range(33600):
    for j in range(784):
        train_rbf[i,j] = rbf(train_x[i],cluster_center[j],100)
    print(i)
train_rbf
test_rbf = np.zeros([8400,784])
for i in range(8400):
    for j in range(784):
        test_rbf[i,j] = rbf(test_x[i],cl_cen[j],100)
    print(i)
test_rbf

# Initialize placeholders 
X = tf.placeholder(dtype = tf.float32, shape = [None,784])
y = tf.placeholder(dtype = tf.int32, shape = [None,10])


# the output layer or NN
output_layer = tf.contrib.layers.fully_connected(X,10,tf.nn.softmax)
pred_layer = tf.argmax(output_layer,1)

# cost function 
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(y,output_layer))

# optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    total_cost = 0
    
    minibatches = random_mini_batches(train_rbf,train_y)
    for epochs in range(150):
        for minibatch in minibatches:
            (mini_x,mini_y) = minibatch
            mini_cost,_ = sess.run([cost,optimizer],feed_dict={X:mini_x,y:mini_y})
            total_cost = total_cost+mini_cost
        print('epoch ',epochs,' : ',total_cost/((epochs+1)*len(minibatches)))
    
    pred_train = pred_layer.eval({X:train_rbf})
    train_acc = np.sum(pred_train==label_train)/label_train.shape[0]
    print('train accuracy : '+str(train_acc))
    
    pred_test = pred_layer.eval({X:test_rbf})
    test_acc = np.sum(pred_test==label_test)/label_test.shape[0]
    print('test accuracy : '+str(test_acc))
    
    pred = []
    for i in range(3):
        pred.append(pred_layer.eval({X:img[i].reshape(1,-1)}))
    print(pred)
plt.subplot(131)
plt.imshow(img[0],cmap='gray')
plt.subplot(132)
plt.imshow(img[1],cmap='gray')
plt.subplot(133)
plt.imshow(img[2],cmap='gray')
plt.show()

pred
img[0][img[0]>25] = 255
plt.imshow(img[0].reshape(28,28),cmap='gray')
plt.show()

np.save('train_rbf.npy',train_rbf)
np.save('test_rbf.npy',test_rbf)
