import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize
dataset = pd.read_csv('../input/train.csv', dtype = np.float32)

kaggletest = pd.read_csv('../input/test.csv', dtype = np.float32)
labels = dataset.label

dataset = dataset.drop('label',1)
def one_hot(labels):

    oneHot = []

    for i in range(len(labels)):

        one = [0,0,0,0,0,0,0,0,0]

        one.insert(labels[i],1)

        oneHot.append(one)

    return oneHot
labels = np.array(one_hot(labels.astype(int)))
dataset = normalize(np.array(dataset))

kaggletest = normalize(np.array(kaggletest))
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels, test_size = 0.1)
train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(train_dataset, train_labels, test_size = 0.13)
print('Training set', train_dataset.shape, train_labels.shape)

print('Validation set', valid_dataset.shape, valid_labels.shape)

print('Test set', test_dataset.shape, test_labels.shape)

train_dataset = np.reshape(train_dataset, [-1, 28, 28, 1])

print('Training set', train_dataset.shape, train_labels.shape)
def accuracy(predictions, labels):

    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])
batch_size =100

K = 6

L = 12

M = 24

N = 200

graph = tf.Graph()

with graph.as_default():

    

    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,28,28,1))

    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,10))

    tf_valid_dataset = tf.constant(valid_dataset)

    tf_valid_dataset = tf.reshape(tf_valid_dataset, [-1, 28, 28, 1])

    tf_test_dataset = tf.constant(test_dataset)

    tf_test_dataset = tf.reshape(tf_test_dataset, [-1, 28, 28, 1])

    

    tf_kaggletest = tf.constant(kaggletest)

    tf_kaggletest = tf.reshape(tf_kaggletest, [-1, 28, 28, 1])

    

    weight1 = tf.Variable(tf.truncated_normal([6,6,1,K] ,stddev=0.1))

    bias1 = tf.Variable(tf.ones([K])/10)

    weight2 = tf.Variable(tf.truncated_normal([5,5,K,L] ,stddev=0.1))

    bias2 = tf.Variable(tf.ones([L])/10)

    weight3 = tf.Variable(tf.truncated_normal([4,4,L,M] ,stddev=0.1))

    bias3 = tf.Variable(tf.ones([M])/10)

    weight4 = tf.Variable(tf.truncated_normal([7*7*M,N] ,stddev=0.1))

    bias4 = tf.Variable(tf.ones([N])/10)

    weight5 = tf.Variable(tf.truncated_normal([N,10] ,stddev=0.1))

    bias5 = tf.Variable(tf.ones([10])/10)

    def nn(data, drop):

        logits1 = tf.nn.conv2d(data, weight1, strides = [1,1,1,1], padding = 'SAME')+bias1

        relu1 = tf.nn.relu(logits1)

        logits2 = tf.nn.conv2d(relu1,weight2, strides = [1,2,2,1], padding = 'SAME')+bias2

        relu2 = tf.nn.relu(logits2)

        logits3 = tf.nn.conv2d(relu2,weight3, strides = [1,2,2,1], padding = 'SAME')+bias3

        relu3 = tf.nn.relu(logits3)

        logits4 = tf.reshape(relu3, shape =[-1,7*7*M])

        logits5 = tf.matmul(logits4,weight4)+bias4

        relu4 = tf.nn.relu(logits5)

        logits6 = tf.matmul(relu4,weight5)+bias5

        logits7 =tf.nn.dropout(logits6, drop)

        logits8 = tf.nn.softmax(logits7)

        return logits8

    

    loss = -tf.reduce_mean(tf_train_labels*tf.log(nn(tf_train_dataset, 0.75)))

    learning_rate=0.3

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    

    

    train_prediction = nn(tf_train_dataset, 1)

    valid_prediction = nn(tf_valid_dataset, 1)

    test_prediction = nn(tf_test_dataset, 1)

    

    kaggletest_prediction = nn(tf_kaggletest, 1)
with tf.Session(graph = graph) as session:

    

    tf.global_variables_initializer().run()

    

    

    for step in range(10001):

        upper = np.random.randint(batch_size,32886)

    

        data_batch = train_dataset[upper-batch_size:upper,:]

        labels_batch = train_labels[upper-batch_size:upper]

        learning_rate = 0.001+(0.3-0.001)*np.exp(-step/10000)

        feed_dict = {tf_train_dataset: data_batch, tf_train_labels:labels_batch}

        _, ent , predict = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

        if(step % 500 == 0):

            print("Loss at " + str(step) + " is " + str(ent))

            print('Train Accuracy at ' + str(step) + ' is ' + str(accuracy(predict, labels_batch)))

            print('Valid Accuracy at '+ str(step) + ' is ' + str(accuracy(valid_prediction.eval(), valid_labels)))

    kaggleprediction = np.argmax(kaggletest_prediction.eval(), 1)

    print('Test Accuracy at ' + str(step) + ' is ' + str(accuracy(test_prediction.eval(), test_labels)))
kaggleprediction1 = pd.DataFrame(kaggleprediction)

kaggleprediction1.to_csv('results.csv',header =['Label'], index_label = 'ImageId')

df = pd.read_csv('results.csv', index_col = ['ImageId'])

df.index += 1

df.index.name = 'ImageId'

df.to_csv('results1.csv')