import pandas as pd

import tensorflow as tf

from bayes_opt import BayesianOptimization

from sklearn.cross_validation import train_test_split

import numpy as np
df = pd.read_json('../input/train.json')

df.inc_angle = df.inc_angle.replace('na', 0)

df.inc_angle = df.inc_angle.astype(float).fillna(0.0)
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])

X_train = np.concatenate([x_band1[:, :, :, np.newaxis]

                          , x_band2[:, :, :, np.newaxis]

                         , ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)

X_angle_train = np.array(df.inc_angle)

y_train = np.array(df["is_iceberg"])
# just take 100 dataset to do optimization

# we assume this 100 able to generalize the whole dataset

# if not enough, increase the number

X_train = X_train[:100]

y_train = y_train[:100]

X_angle_train = X_angle_train[:100].reshape((-1, 1))

y_onehot = np.zeros((y_train.shape[0], np.unique(y_train).shape[0]))

for i in range(y_train.shape[0]):

    y_onehot[i, y_train[i]] = 1.0

    

x_train, x_test, y_train, y_test, x_train_angle, x_test_angle = train_test_split(X_train, y_onehot, X_angle_train, test_size = 0.20)
def neural_network(fully_conn_size, # wide size of fully connected layer

                   len_layer_conv, # each conv layer included max pooling

                   kernel_size, # kernel size for conv process

                   learning_rate, # learning rate

                   pooling_size, # kernel and stride size for pooling

                   multiply, # constant to multiply for conv output

                   dropout_rate, # dropout

                   beta, # l2 norm discount

                   activation,

                   batch_normalization, 

                   batch_size = 20):

    

    tf.reset_default_graph()

    if activation == 0:

        activation = tf.nn.sigmoid

    elif activation == 1:

        activation = tf.nn.tanh

    else:

        activation = tf.nn.relu

    

    def conv_layer(x, conv, out_shape):

        w = tf.Variable(tf.truncated_normal([conv, conv, int(x.shape[3]), out_shape]))

        b = tf.Variable(tf.truncated_normal([out_shape], stddev = 0.01))

        return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding = 'SAME') + b

    

    def fully_connected(x, out_shape):

        w = tf.Variable(tf.truncated_normal([int(x.shape[1]), out_shape]))

        b = tf.Variable(tf.truncated_normal([out_shape], stddev = 0.01))

        return tf.matmul(x, w) + b

    

    def pooling(x, k = 2, stride = 2):

        return tf.nn.max_pool(x, ksize = [1, k, k, 1], 

                              strides = [1, stride, stride, 1], 

                              padding = 'SAME')

    

    X_img = tf.placeholder(tf.float32, (None, 75, 75, 3))

    X_angle = tf.placeholder(tf.float32, (None, 1))

    Y = tf.placeholder(tf.float32, (None, y_onehot.shape[1]))

    # for batch normalization

    train = tf.placeholder(tf.bool)

    

    for i in range(len_layer_conv):

        if i == 0:

            conv = activation(conv_layer(X_img, kernel_size, 

                                         int(np.around(int(X_img.shape[3]) * multiply))))

        else:

            conv = activation(conv_layer(conv, kernel_size, 

                                         int(np.around(int(conv.shape[3]) * multiply))))

        conv = pooling(conv, k = pooling_size, stride = pooling_size)

        if batch_normalization:

            conv = tf.layers.batch_normalization(conv, training = train)

        conv = tf.nn.dropout(conv, dropout_rate)

    print(conv.shape)

    output_shape = int(conv.shape[1]) * int(conv.shape[2]) * int(conv.shape[3])

    conv = tf.reshape(conv, [-1, output_shape])

    conv = tf.concat([conv, X_angle], axis = 1)

    

    # our fully connected got 1 layer

    # you can increase it if you want

    for i in range(1):

        if i == 0:

            fc = activation(fully_connected(conv, fully_conn_size))

        else:

            fc = activation(fully_connected(fc, fully_conn_size))

        fc = tf.nn.dropout(fc, dropout_rate)

        if batch_normalization:

            fc = tf.layers.batch_normalization(fc, training = train)

            

    logits = fully_connected(fc, y_onehot.shape[1])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, 

                                                                  logits = logits))

    cost += sum(beta * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())

    COST, TEST_COST, ACC, TEST_ACC = [], [], [], []

    

    for i in range(20):

        train_acc, train_loss = 0, 0

        for n in range(0, (X_train.shape[0] // batch_size) * batch_size, batch_size):

            _, loss = sess.run([optimizer, cost], 

                               feed_dict = {X_img: x_train[n: n + batch_size, :, :, :],

                                            X_angle: x_train_angle[n: n + batch_size],

                                            Y: y_train[n: n + batch_size, :], 

                                            train: True})

            train_acc += sess.run(accuracy, 

                                  feed_dict = {X_img: x_train[n: n + batch_size, :, :, :],

                                               X_angle: x_train_angle[n: n + batch_size],

                                               Y: y_train[n: n + batch_size, :], 

                                               train: False})

            train_loss += loss

        results = sess.run([cost, accuracy], 

                           feed_dict = {X_img: x_test,

                                        X_angle: x_test_angle,

                                        Y: y_test, 

                                        train: False})

        TEST_COST.append(results[0])

        TEST_ACC.append(results[1])

        train_loss /= (x_train.shape[0] // batch_size)

        train_acc /= (x_train.shape[0] // batch_size)

        ACC.append(train_acc)

        COST.append(train_loss)

    COST = np.array(COST).mean()

    TEST_COST = np.array(TEST_COST).mean()

    ACC = np.array(ACC).mean()

    TEST_ACC = np.array(TEST_ACC).mean()

    return COST, TEST_COST, ACC, TEST_ACC
def generate_nn(fully_conn_size, len_layer_conv, kernel_size, learning_rate, pooling_size, multiply,

                dropout_rate, beta, activation, batch_normalization):

    global accbest

    param = {

        'fully_conn_size' : int(np.around(fully_conn_size)),

        'len_layer_conv' : int(np.around(len_layer_conv)),

        'kernel_size': int(np.around(kernel_size)),

        'learning_rate' : max(min(learning_rate, 1), 0.0001),

        'pooling_size': int(np.around(pooling_size)),

        'multiply': multiply,

        'dropout_rate' : max(min(dropout_rate, 0.99), 0),

        'beta' : max(min(beta, 0.5), 0.000001),

        'activation': int(np.around(activation)),

        'batch_normalization' : int(np.around(batch_normalization))

    }

    learning_cost, valid_cost, learning_acc, valid_acc = neural_network(**param)

    print("stop after 20 iteration with train cost %f, valid cost %f, train acc %f, valid acc %f" % (learning_cost, valid_cost, learning_acc, valid_acc))

    # a very simple benchmark, just use correct accuracy

    # if you want to change to f1 score or anything else, can

    if (valid_acc > accbest):

        costbest = valid_acc

    return valid_acc
accbest = 0.0

NN_BAYESIAN = BayesianOptimization(generate_nn, 

                              {'fully_conn_size': (16, 128),

                               'len_layer_conv': (3, 5),

                               'kernel_size': (2, 7),

                               'learning_rate': (0.0001, 1),

                               'pooling_size': (2, 4),

                               'multiply': (1, 3),

                               'dropout_rate': (0.1, 0.99),

                               'beta': (0.000001, 0.49),

                               'activation': (0, 2),

                               'batch_normalization': (0, 1)

                              })

NN_BAYESIAN.maximize(init_points = 10, n_iter = 10, acq = 'ei', xi = 0.0)
print('Maximum NN accuracy value: %f' % NN_BAYESIAN.res['max']['max_val'])

print('Best NN parameters: ', NN_BAYESIAN.res['max']['max_params'])