import tensorflow as tf 

import pickle

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
learning_rate = 0.001

n_epochs = 100

batch_size = 50
data_dir="../input"



def unpickle(file):

    '''Load byte data from file'''

    with open(file, 'rb') as f:

        data = pickle.load(f,encoding='latin1')

    return data





def load_cifar10_data():

    '''

    Return train_data, train_labels, test_data, test_labels

    The shape of data returned would be as it is in the data-set N X 3072



    We don't particularly need the metadata - the mapping of label numbers to real labels

    '''

    train_data = None

    train_labels = []



    for i in range(1, 6):

        data_dic = unpickle(data_dir + "/data_batch_"+str(i))

        if i == 1:

            train_data = data_dic['data']

        else:

            train_data = np.append(train_data, data_dic['data'])

        train_labels += data_dic['labels']



    test_data_dic = unpickle(data_dir + "/test_batch")

    test_data = test_data_dic['data']

    test_labels = test_data_dic['labels']

    names=unpickle(data_dir+'/batches.meta')

    

    return train_data, np.array(train_labels), test_data, np.array(test_labels), names['label_names']

train_data,train_labels,test_data,test_labels,names=load_cifar10_data()

train_data.shape
IMG_IDX=13

train_data_vis=train_data.reshape((-1,3072))

train_data_vis=train_data_vis[IMG_IDX].reshape(3,32,32).transpose([1, 2, 0])

plt.imshow(train_data_vis,interpolation='sinc')

print (names[train_labels[IMG_IDX]])



features=tf.placeholder(tf.float32,shape=(None,3*1024)) # will store the train-set images in each of its rows

labels=tf.placeholder(tf.float32,shape=(None))



input_layer=tf.reshape(features,[-1,32,32,3]) # -1 is to derive the shape along that axis automatically



# Convolutional Layer #1

kernel_1 = tf.Variable(tf.random_normal([5,5,3,64]))

biases_1 = tf.Variable(tf.random_normal([64]))

conv1_ = tf.nn.conv2d(

    input_layer,

    kernel_1,

    [1,1,1,1],

    padding="VALID")



conv1=tf.nn.relu(tf.nn.bias_add(conv1_,biases_1))



# Pooling Layer #1

pool1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,1,1,1],"VALID")
pool1.shape
# Convolutional Layer #2 and Pooling Layer #2

kernel_2 = tf.Variable(tf.random_normal([5,5,64,128]))

biases_2=tf.Variable(tf.random_normal([128]))

conv2_ = tf.nn.conv2d(

    pool1,

    kernel_2,

    [1,1,1,1],

    padding="VALID")

conv2=tf.nn.relu(tf.nn.bias_add(conv2_,biases_2))

pool2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1],"VALID")

pool2_flat=tf.reshape(pool2,[-1,11*11*128])
pool2.shape
'''

# Convolutional Layer #3 and Pooling Layer #3

kernel_3 = tf.Variable(tf.random_normal([3,3,128,32]))

biases_3 = tf.Variable(tf.random_normal([32]))

conv3_ = tf.nn.conv2d(

    pool2,

    kernel_3,

    [1,1,1,1],

    padding="VALID")

conv3=tf.nn.relu(tf.nn.bias_add(conv3_,biases_3))

pool3 = tf.nn.max_pool(conv3, [1,2,2,1], [1,2,2,1],"VALID")

'''
pool3.shape
#pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 32])



dense1 = tf.layers.dense(inputs=pool2_flat, units=3072)

dense2 = tf.layers.dense(inputs=dense1, units=1012)

dense3 = tf.layers.dense(inputs=dense2, units=200)



# we have only 10 classes to classify

logits = tf.layers.dense(inputs=dense3, units=10)



# finding the class wiht max probability

predictions = tf.argmax(input=logits, axis=1)

onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)





accu = tf.metrics.accuracy(labels=labels, predictions=predictions)

init_g = tf.global_variables_initializer()

init_l = tf.local_variables_initializer()
sess=tf.Session()

sess.run(init_g)

sess.run(init_l)



#############################  The real training stage begins here ##############################################

train_data = train_data.reshape(-1,3072)

rand_index = np.arange(len(train_data))

for j in range(n_epochs):

    np.random.shuffle(rand_index)

    train_data=train_data[rand_index]

    train_labels=train_labels[rand_index]

    train_data_split=np.split(train_data,batch_size)

    train_labels_split=np.split(train_labels,batch_size)

    n_batches = len(train_data_split)

    for i in range(n_batches):

        batch_x = train_data_split[i]

        batch_y = train_labels_split[i]

        sess.run(train_step, feed_dict={features: batch_x, labels: batch_y})

        # finding E_out at the end of each epoch

    acc_val=sess.run(accu,feed_dict={features: test_data.reshape((-1,3072)), labels: test_labels})

    print ("epoch:",j,":\t\t Test Accuracy:% 0.4f" %(acc_val[1]))