import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
########## Can run the following commands to check the devices compatible with tf 





# from tensorflow.python.client import device_lib

# device_lib.list_local_devices()





########### Loading Dataset :

# mnist_dataset = tf.keras.datasets.mnist

# (x_train,y_train),(x_test, y_test) = mnist_dataset.load_data()



training_data = pd.read_csv("../input/digit-recognizer/train.csv")

testing_data = pd.read_csv("../input/digit-recognizer/test.csv")
flat_labels = training_data.iloc[:,0].values

print(" Image Labels :  ",flat_labels[0:20])

images = training_data.iloc[:,1:].values

images = images.astype(np.float32)
plt.imshow(images[6].reshape(28,28))

plt.show()
# # https://www.kaggle.com/bhushan23/mnist-with-softmax-tensorflow-tutorial

# class Dataset(object):

#     def __init__(self, data):

#         self.rows = len(data.values)

#         self.images = data.iloc[:,1:].values

#         self.images = self.images.astype(np.float32)

#         self.images = np.multiply(self.images, 1.0 / 255.0)

#         self.labels = np.array([np.array([int(i == l) for i in range(10)]) for l in data.iloc[:,0].values]) #one-hot

#         self.index_in_epoch = 0

#         self.epoch = 0

#     def next_batch(self, batch_size):

#         start = self.index_in_epoch

#         self.index_in_epoch += batch_size

#         if self.index_in_epoch > self.rows:

#             self.epoch += 1

#             perm = np.arange(self.rows)

#             np.random.shuffle(perm)

#             self.images = self.images[perm]

#             self.labels = self.labels[perm]

#             #next epoch

#             start = 0

#             self.index_in_epoch = batch_size

#         end = self.index_in_epoch

#         return self.images[start:end] , self.labels[start:end]



one_hot_encoder = OneHotEncoder(sparse = False)

flat_labels = flat_labels.reshape(len(flat_labels), 1)

labels = one_hot_encoder.fit_transform(flat_labels)

labels = labels.astype(np.uint8)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 0)
#input tensor 

x = tf.placeholder(tf.float32, [None, 784])   #None basically means it's dynamically changeable

y_ = tf.placeholder(tf.float32, [None, 10])   #Actual Ys 



#variables 



W = tf.Variable(tf.zeros([784, 10]))     #weights 

b = tf.Variable(tf.zeros([10]))          #biases 

y = tf.nn.softmax(tf.matmul(x,W) + b)       #y_pred (out)


#########Defining error // Loss function



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y), reduction_indices=[1] ))



#############Optimizing algo: 

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# train = Dataset(training_data.iloc[0:37000])

# training_data.shape
# sess = tf.InteractiveSession()       ## initializing session for nn to compute



# tf.global_variables_initializer().run()  #Pretty self-explanatory 



# for _ in range(1000):

#     batch_x, batch_y = train.next_batch(60)

#     sess.run(train_step, feed_dict = {x:batch_x, y_:batch_y})

# train_data = Dataset(training_data)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

batch_size = 200

for i in range(500):

    batch_start = ( i * batch_size) % (x_train.shape[0] - batch_size)

    batch_end = batch_start + batch_size 

    batch_xs = x_train[batch_start:batch_end]

    batch_ys = y_train[batch_start:batch_end]

#     batch_xs, batch_ys = train_data.next_batch(500)



    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1) , tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
test = testing_data.values.astype(np.float32)

test = np.multiply(test, 1.0/255)
pred = sess.run(y, feed_dict={x:test})

pred = [np.argmax(p) for p in pred]

result = pd.DataFrame({

    'ImageId' : range(1,len(pred)+1), 

    'Label' : pred

})



result.to_csv('result.csv', index = False, encoding = 'utf-8')