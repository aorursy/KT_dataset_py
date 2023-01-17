import numpy as np 

import pandas as pd 



import tensorflow as tf



from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



# importation de train et test

data_tr = pd.read_csv("../input/fashion-mnist_train.csv")

data_ts = pd.read_csv("../input/fashion-mnist_test.csv")
x_train, x_valid, y_train, y_valid = train_test_split(data_tr.drop("label",axis=1),data_tr.label, test_size = 0.20, random_state = 200)



x_train = np.array((x_train*1.0)/255.0,dtype=np.float32)

x_valid = np.array((x_valid*1.0)/255.0,dtype=np.float32)
y_train = OneHotEncoder(sparse = False).fit_transform(y_train.reshape(-1,1)).astype(np.uint8)

y_valid = OneHotEncoder(sparse = False).fit_transform(y_valid.reshape(-1,1)).astype(np.uint8)
data_ts_label = data_ts.label.values

y_test = OneHotEncoder(sparse = False).fit_transform(data_ts_label.reshape(-1,1)).astype(np.uint8)

x_test = np.array((data_ts.drop("label",axis=1)*1.0)/255.0,dtype=np.float32)
epochs_completed = 0

index_in_epoch = 0

num_examples = x_train.shape[0]



# serve data by batches

def next_batch(batch_size):

    

    global x_train

    global y_train

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

        x_train = x_train[perm]

        y_train = y_train[perm]

        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples

    end = index_in_epoch

    return x_train[start:end], y_train[start:end]
#x_valid = tf.reshape(x_valid,[-1,28,28,1])



X = tf.placeholder(tf.float32,[None,784])

y_true = tf.placeholder(tf.float32,[None,10])



L = 12 

M = 24

N = 48

O = 200



W1 = tf.Variable(tf.truncated_normal([5,5,1,L], stddev=0.1))

b1 = tf.Variable(tf.ones([L])/10)

W2 = tf.Variable(tf.truncated_normal([5,5,L,M], stddev=0.1))

b2 = tf.Variable(tf.ones([M])/10)

W3 = tf.Variable(tf.truncated_normal([4,4,M,N], stddev=0.1))

b3 = tf.Variable(tf.ones([N])/10)



W4 = tf.Variable(tf.truncated_normal([7 * 7 * N, O], stddev=0.1))

b4 = tf.Variable(tf.ones([O])/10)

W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))

b5 = tf.Variable(tf.ones([10])/10)



# The model

XX = tf.reshape(X,shape = [-1,28,28,1])

stride = 1  

Y1 = tf.nn.relu(tf.nn.conv2d(XX, W1, strides=[1, stride, stride, 1], padding='SAME') + b1)

stride = 2 

Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + b2)

stride = 2  

Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + b3)



YY = tf.reshape(Y3, shape=[-1, 7 * 7 * N])



Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)



keep_prob = tf.placeholder('float')

Y4_drop_out = tf.nn.dropout(Y4, keep_prob)



Ylogits = tf.matmul(Y4_drop_out, W5) + b5

y_pred = tf.nn.softmax(Ylogits)



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= Ylogits, labels = y_true)

cost = tf.reduce_mean(cross_entropy)



global_step = tf.Variable(0, trainable=False)

starter_learning_rate = 0.001

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100, 0.90, staircase=True)



#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)



correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

sess.run(init)



print("Training step")

for i in range(15000):

    batch_x_tr, batch_y_tr = next_batch(100)

    _,a = sess.run([train_step,accuracy],feed_dict = {X: batch_x_tr, y_true: batch_y_tr,keep_prob: 0.65})

    if i % 100 == 0:

        print("Step:",i, "Accuracy:",a)



a_valid = sess.run(accuracy, feed_dict = {X: x_valid, y_true: y_valid,keep_prob: 1.0})

print("Validation accuracy", a_valid)
a_ts = sess.run(accuracy, feed_dict = {X: x_test, y_true: y_test,keep_prob: 1.0})

print("Test accuracy", a_ts)


