import numpy as np

import pandas as pd

import tensorflow as tf
print('Loading data...')

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape)

print(test.shape)
train.head()
test.head()
train_X = train.iloc[:, 1:].values.astype('float32')

train_y = train.iloc[:, 0].values.astype('float32') 
print(train_X.shape)

print(train_y.shape)
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

lb.fit(range(0,10))

print(lb.classes_)

y_onehot = lb.transform(train_y)

print(y_onehot[0], train_y[0])



X_train, X_val, y_train, y_val = map(lambda x : np.array(x).astype(np.float32), train_test_split(train_X, y_onehot, test_size=0.2))
X_train.shape
n_input = 784  #MNIST data input (img shape: 28*28)

n_classes = 10 #MNIST total classes (0-9 digits)

hidden_layer_size = 50
#Input data and labels in batch

X_input = tf.placeholder(tf.float32,[None, n_input])

y_teacher = tf.placeholder(tf.float32,[None, n_classes])



#drop out to avoid overlearning

keep_prob_input = tf.placeholder(tf.float32)



##First layer##

X_input_layer = tf.nn.dropout(X_input, keep_prob=keep_prob_input)



def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.01)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.0, shape=shape)

    return tf.Variable(initial)



W_fc1 = weight_variable([n_input, hidden_layer_size])

b_fc1 = bias_variable([hidden_layer_size])



###Use relu function as activation function

h_fc1 = tf.nn.relu(tf.matmul(X_input_layer, W_fc1)+ b_fc1)



keep_prob = tf.placeholder(tf.float32)



###output after drop out

h_fc1_dout = tf.nn.dropout(h_fc1, keep_prob)

##Second layer##

W_fc2 = weight_variable([hidden_layer_size, hidden_layer_size])

b_fc2 = bias_variable([hidden_layer_size])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dout, W_fc2)+ b_fc2)

h_fc2_dout = tf.nn.dropout(h_fc2, keep_prob)
##Third layer##

W_fc3 = weight_variable([hidden_layer_size, hidden_layer_size])

b_fc3 = bias_variable([hidden_layer_size])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_dout, W_fc3)+ b_fc3)

h_fc3_dout = tf.nn.dropout(h_fc3, keep_prob)
##Fourth layer##

W_fc4 = weight_variable([hidden_layer_size, n_classes])

b_fc4 = bias_variable([n_classes])

y_out = tf.nn.softmax(tf.matmul(h_fc3_dout, W_fc4)+ b_fc4)
#Cost function

#learning_rate = tf.placeholder(tf.float32)



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_teacher*tf.log(y_out), reduction_indices=[1]))



#Optimization function

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)



#Iteration



correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_teacher, 1))



#Accuracy

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Hyper parameters

batch_size = 100

epoch_size = 20

best_accuracy = 0.0
init = tf.global_variables_initializer()



import random

saver = tf.train.Saver()



sess = tf.Session()

sess.run(init)



saver.save(sess, 'mnist_fc_best')





def random_sample(X, y, size=100):

    idx = range(0, len(y))

    random_idx = random.sample(idx, size)

    return X[random_idx, :], y[random_idx, :]
for epoch in range(1,epoch_size+1):

    

    for i in range(int(len(y_train)/batch_size)):

        X_batch, y_batch = random_sample(X_train, y_train, batch_size)

        

        if i == 0:

            print("=====================")

            train_accuracy = sess.run(accuracy, feed_dict = {

                    X_input:X_batch, y_teacher:y_batch, keep_prob_input:1.0, keep_prob:1.0

            })

            print("{} : training accuracy {}%".format(epoch, train_accuracy*100))

            

            val_accuracy = sess.run(accuracy, feed_dict={

                X_input: X_val, y_teacher: y_val, keep_prob_input: 1.0, keep_prob: 1.0})

            

            print("{} : validation accuracy {}%".format(epoch, val_accuracy*100))

            

    

            if val_accuracy >= best_accuracy:

                saver.save(sess, 'mnist_fc_best')

                best_accuracy = val_accuracy

                print("Validation accuracy improved: {}%. Saving the network.".format(val_accuracy*100))

            else:

                saver.restore(sess, 'mnist_fc_best')

                print("restore!!!! now : {}, before : {}".format(val_accuracy*100, best_accuracy*100))

                

        sess.run(optimizer, feed_dict={

                X_input: X_batch, y_teacher: y_batch, keep_prob_input: 0.9, keep_prob: 1.0})

        

saver.restore(sess,'mnist_fc_best')
test_X = test.values.astype('float32')
test_X.shape
predictions = sess.run([tf.argmax(y_out, 1)], feed_dict={X_input: test_X, keep_prob_input: 0.9, keep_prob: 1.0})
print(predictions)
results = pd.DataFrame({'ImageId': pd.Series(range(1, len(predictions[0]) + 1)), 'Label': pd.Series(predictions[0])})

results.to_csv('tensorflor_result.csv', index=False)