import pandas as pd

import numpy as np

import random

from sklearn.preprocessing import StandardScaler



def parse_mydata():

    features = []

    with open('../input/data_5000.csv') as f:

        lines = f.readlines()

        for line in lines:

            splitted = line.split(';')

            featureset = list(map(int,splitted[:9414]))

            label = splitted[-1].rstrip()

            hotlable = []

            if label == '1':

                hotlable =[1,0]

            else:

                hotlable = [0,1]

            features.append([featureset, hotlable])

    return features



def bring_processed():

    features = parse_mydata()

    random.seed(1)

    random.shuffle(features)

    features = np.array(features)



    testing_size = int(0.2 *len(features))

    scaler = StandardScaler()

    

    train_x = list(features[:,0][:-testing_size])

    scaler.fit(train_x)

    train_x = scaler.transform(train_x)

    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])

    scaler.fit(test_x)

    test_x = scaler.transform(test_x)

    test_y = list(features[:,1][-testing_size:])



    return train_x, train_y, test_x, test_y

import tensorflow as tf

import numpy as np

#from processdata import create_feature_sets_and_labels

#from my_data_parse import bring_processed

# train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

train_x,train_y,test_x,test_y = bring_processed()





n_nodes_hl1 = 2000

n_nodes_hl2 = 500

n_nodes_hl3 = 100



n_classes = 2

batch_size = 100

n_epochs = 20



x = tf.placeholder('float', [None, len(train_x[0])])

y = tf.placeholder(tf.int32, [None, len(train_y[0])])



def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),

                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}



#     hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),

#                       'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}



#     hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),

#                       'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}



    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),

                    'biases':tf.Variable(tf.random_normal([n_classes])),}



    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])

    l1 = tf.nn.relu(l1)



#     l2 = tf.nn.bias_add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])

#     l2 = tf.nn.relu(l2)



#     l3 = tf.nn.bias_add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])

#     l3 = tf.nn.relu(l3)



    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']



    return output



def train_neural_network(x):

    #prediction = neural_network_model(x)

    logits = neural_network_model(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y) )

    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

#     prediction = tf.cast(tf.argmax(logits, 1), tf.int32)

#     accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))

#     global_step = tf.Variable(0, trainable=False)

#     starter_learning_rate = 0.01

#     learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,

#                                        100000, 0.96, staircase=True)

    global_step = tf.Variable(0, trainable=True)

    starter_learning_rate = 0.01

    k = 0.5

    learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)





    

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):

            epoch_loss = 0

            i=0

            while i < len(train_x):

                start = i

                end = i+batch_size

                batch_x = np.array(train_x[start:end])

                batch_y = np.array(train_y[start:end])



                _, c ,acc= sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x,y: batch_y})

                epoch_loss += c

                i+=batch_size





            print('Epoch', epoch + 1, 'completed out of',n_epochs,'loss:',epoch_loss, "accuracy={:.4f}".format(acc))

            #prediction = tf.nn.softmax(logits)

            corr = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

    



            acc2 = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',acc2.eval({x:test_x, y:test_y}))



train_neural_network(x)



# final accuracy

# final accuracy

predicted, acc = sess.run([correct, accuracy], feed_dict = {input_x: test_x, input_y: test_y})



print("Final accuracy on test set:", acc)
