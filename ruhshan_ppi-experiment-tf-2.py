import tensorflow as tf

import numpy as np
#helper functions

def make_feature_row(line):

    splitted = line.split(';')

    #featureset = list(map(int,splitted[:9414]))

    featureset = [int(float(x)) for x in splitted[:len(splitted)-1]]

    label = int(float(splitted[-1].rstrip()))

    #print(label)

    hotlable = []

    if label == 1:

        #print('o', end=',')

        hotlable =[1,0]

    else:

        #print('t', end=',')

        hotlable = [0,1]

    #print(label, hotlable)

    return [featureset, hotlable]



def get_test_data():

    features = []

    with open('../input/ppi_data_15000/test_data.csv') as f:

        features = [make_feature_row(l) for l in f]

    features = np.array(features)

    test_x = list(features[:,0])

    test_y = list(features[:,1])

    return test_x, test_y



def get_train_data_batch(n):

    features = []

    with open('../input/ppi_data_15000/train_data.csv') as f:

        ct=0

        for l in f:

            features.append(make_feature_row(l))

            ct+=1

            if ct%n==0:

                features = np.array(features)

                train_x = list(features[:,0])

                train_y = list(features[:,1])

                if ct%2000==0:

                    print("giving data", ct)

                yield train_x, train_y

                features = []

n_nodes_hl1 = 1000

n_nodes_hl2 = 500

n_nodes_hl3 = 100



n_classes = 2

batch_size = 100

hm_epochs = 2



train_data_batch_gen = get_train_data_batch(batch_size)

train_x,train_y = train_data_batch_gen.__next__()

print("intitial training load done")

test_x,test_y  = get_test_data()

print("loading done")



x = tf.placeholder('float', [None, len(train_x[0])])

y = tf.placeholder('float', [None, len(train_y[0])])



def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),

                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}



    # hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),

    # 				  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    #

    # hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),

    # 				  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}



    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),

                        'biases':tf.Variable(tf.random_normal([n_classes])),}





    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])

    l1 = tf.nn.relu(l1)



    # l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])

    # l2 = tf.nn.relu(l2)

    #

    # l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])

    # l3 = tf.nn.relu(l3)



    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']



    return output



def train_neural_network(x):

    prediction = neural_network_model(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)



    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):

            train_data_batch_gen = get_train_data_batch(batch_size)

            epoch_loss = 0

            i=0

            while i < 24000:

                start = i

                end = i+batch_size

                train_x, train_y = train_data_batch_gen.__next__()





                batch_x = np.array(train_x)

                batch_y = np.array(train_y)



                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,

                                                              y: batch_y})

                epoch_loss += c

                i+=batch_size







            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)



            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))



            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))



        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))



train_neural_network(x)