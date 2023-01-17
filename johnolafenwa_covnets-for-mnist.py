import tensorflow as tf

import tensorflow.contrib.slim as slim

import numpy as np



import pickle, gzip





pck = gzip.open("../input/mnist.pkl.gz","rb")



data = pickle.load(pck,encoding="latin1")



(x_train, y_train), (x_test, y_test), (x_valid, y_valid) = data



#OUTPUT SHAPES



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(x_valid.shape)

print(y_valid.shape)



def one_hot_encode(x):

    labels = []

    for label in x:

        one_hot = np.array([int(i == label) for i in range(10)])

        labels.append(one_hot)

    return np.array(labels)



y_train = one_hot_encode(y_train)

y_test = one_hot_encode(y_test)

y_valid = one_hot_encode(y_valid)



print(y_train)

n_classes = 10

batch_size = 1000







def network(inputs,num_classes, is_training=True):



    net = slim.conv2d(inputs,num_outputs=32,kernel_size=[2,2],stride=[2,2],padding="SAME",activation_fn=tf.nn.relu)



    net = slim.max_pool2d(net,kernel_size=[2,2])

    net = slim.conv2d(net,num_outputs=32,kernel_size=[2,2],stride=[2,2],padding="SAME",activation_fn=tf.nn.relu)





    net = slim.batch_norm(net,is_training=is_training)



    net = slim.fully_connected(net,num_outputs=64,activation_fn=tf.nn.relu)

    net = slim.fully_connected(net,num_outputs=num_classes,activation_fn=None)



    return net





def run():



    log_dir = "../model"

    with tf.Graph().as_default():



        tf.logging.set_verbosity(tf.logging.INFO)



        predictions = network(x_train,n_classes,True)



        loss = slim.losses.softmax_cross_entropy(logits=predictions,onehot_labels =y_train)

        total_loss = slim.losses.get_total_loss()



        optimizer = slim.train.AdamOptimizer(learning_rate=0.001)



        train_op = optimizer.minimize(total_loss)



        slim.learning.train(logdir=log_dir,train_op=train_op, number_of_steps=1000,save_interval_secs=20, save_summaries_secs=10)





        test_predictions = network(x_test,n_classes,False)



        test_loss = slim.metrics.streaming_accuracy(test_predictions,y_test)



        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            sess.run(tf.local_variables_initializer())



            for batch_id in range(batch_size):

                sess.run(test_loss)



            acc = sess.run(test_loss)



            for key, value in acc:

                print(key, " = ",value)





run()












