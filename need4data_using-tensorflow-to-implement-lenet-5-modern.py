# import modules

import pandas as pd

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.cm as cm 

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split





df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')

# inspecting first five rows

df_train.head()
print(df_train.shape)
sns.countplot(df_train.label)
# extract label from train dataset

labels = df_train[['label']]

df_train = df_train.drop('label',axis = 1)
# Normalization

df_train = df_train.astype('float32')/255

df_test = df_test.astype('float32')/255
train = df_train.values.reshape(-1,28,28,1)
# visualize three random pictures



plt.subplot(131)

plt.imshow(train[0][:,:,0],cmap=cm.binary)



plt.subplot(132)

plt.imshow(train[10][:,:,0],cmap=cm.binary)



plt.subplot(133)

plt.imshow(train[100][:,:,0],cmap=cm.binary)
labels['label'] = labels['label'].astype('str')

labels = pd.get_dummies(labels)
# setting random_state as 189 (the instance labels we visualize above :-> )

X_train,X_test,y_train,y_test = train_test_split(

    df_train,labels,

    test_size = 0.10, 

    random_state = 189,

    shuffle = True)
print('Train data size:')

print('X_train:',X_train.shape)

print('y_train:',y_train.shape)



print('Test data size:')

print('X_test:',X_test.shape)

print('y_test:',y_test.shape)

image_size = X_train.shape[1]

label_size = y_train.shape[1]
label_size
image_size
X_train.head()
# Inputs and Outputs

x = tf.placeholder('float', shape=[None, image_size])

y = tf.placeholder('float', shape=[None, label_size])



# weights init



def weight_variable(shape):

    initial = tf.truncated_normal(shape,stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape = shape)

    return tf.Variable(initial)



# convolution definition



def conv2d(x, W, padding='VALID' ):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)



# pooling definition

def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')





# settings

LEARNING_RATE = 1e-4

       

    

DROPOUT = 0.7

BATCH_SIZE =200
# shape = width*height*channels





# input     shape:28*28*1

image = tf.reshape(x, [-1,28 , 28,1])



# layer 1 convolution     shape:28*28*6

W_conv1 = weight_variable([5,5,1,6])

b_conv1 = bias_variable([6])



h_conv1 = tf.nn.relu(conv2d(image, W_conv1,padding = 'SAME') + b_conv1)



# layer 2 average pooling     shape:14*14*6

h_pool1 = max_pool_2x2(h_conv1)



# layer 3 convolution     shape:10*10*16

W_conv2 = weight_variable([5,5,6,16])

b_conv2 = bias_variable([16])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)



# layer 4 average pooling     shape:5*5*16

h_pool2 = max_pool_2x2(h_conv2)



# layer 5 fully connect with 120 neurons     120 dim array

W_fc1 = weight_variable([5*5*16, 120])

b_fc1 = bias_variable([120])



h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*16])



h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)



## add dropout for regularization

keep_prob = tf.placeholder('float')

h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)



# layer 6 fully connect with 84 neurons     84 dim array

W_fc2 = weight_variable([120, 84])

b_fc2 = bias_variable([84])



h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)



## add dropout for regularization



h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)





# layer 7 softmax output     10 dim array

W_readout = weight_variable([84,label_size])

b_readout = bias_variable([label_size])



y_prob = tf.nn.softmax(tf.matmul(h_fc2_drop, W_readout) + b_readout)



# Cost function: Cross Entropy Loss

cross_entropy = -tf.reduce_sum(y*tf.log(y_prob))



# optimisation function

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)





# evaluation

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_prob,1))



accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

predict = tf.argmax(y_prob,1)
epochs_completed = 0

index_in_epoch = 0

num_examples = X_train.shape[0]



train = X_train.values

label = y_train.values



epoches = 20

# serve data by batches

def next_batch(batch_size):

    

    global train

    global label

    global index_in_epoch

    global epochs_completed

    



    

    start = index_in_epoch

    index_in_epoch += batch_size

    

    # when all trainig data have been already used, it is reorder randomly    

    if index_in_epoch > num_examples:

        # finished epoch

        epochs_completed += 1

        # shuffle the data

        perm = np.arange(num_examples)

        np.random.shuffle(perm)

        train = train[perm]

        label = label[perm]

        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples

    end = index_in_epoch

    return train[start:end], label[start:end]
# extract values from DataFrame as numpy array

validation_features = X_test.values

validation_labels = y_test.values

test_features = df_test.values



# setting epoch

EPOCHES = 50



# initialize all variables 

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()

sess.run(init)



# start trainning

# per epoch train

for epoch in range(EPOCHES):

    # per mini-batch train

    for mini_batch in range(num_examples // BATCH_SIZE):

        # get mini-batch

        batch_xs, batch_ys = next_batch(BATCH_SIZE)

        

        # forward propagation and backward propagation, optimize with Adam gradient descent

        

        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob:DROPOUT})

    

    # caculate train dataset accuracy and validation dataset accuracy

    train_accuracy = sess.run(accuracy, feed_dict={x: train, y: label, keep_prob:1.0})

    validation_accuracy = sess.run(accuracy, feed_dict={x: validation_features, y: validation_labels, keep_prob:1.0})

    

    print("epoch "+ str(epoch)+ ': '+'train_accuracy: '+ str(train_accuracy),

             'validation_accuracy: '+ str(validation_accuracy))



predicted_lables  = sess.run(predict,feed_dict={x:test_features,keep_prob: 1.0})
predicted_lables
id = np.array(list(df_test.index)) +1
submission = pd.DataFrame({'ImageId':list(id),'Label':list(predicted_lables)})
submission.to_csv('submission.csv',index = False)
sess.close()