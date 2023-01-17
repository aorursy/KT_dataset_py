

def xrange(x):



    return iter(range(x))



def xaver_init(n_inputs, n_outputs, uniform = True):

    if uniform:

        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))

        return tf.random_uniform_initializer(-init_range, init_range)



    else:

        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs) /2)

        return tf.truncated_normal_initializer(stddev=stddev)

    



import numpy as np

import tensorflow as tf

import pandas as pd 



learning_rate = 0.01



#reset

tf.reset_default_graph()



#데이터 셋 준비

train_ori = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#y값에 따라 3개 클래스로 분리

y_train = pd.get_dummies(train_ori[["type"]], prefix="")

#x-color를 0,1 인코드

color = pd.get_dummies(train_ori[["color"]], prefix="")



train_ori.drop('type',inplace=True, axis=1)

train_ori.drop('id',inplace=True, axis=1)

train_ori.drop('color',inplace=True, axis=1)



x_train = pd.concat([train_ori, color], axis=1)



#linear regression multi variable

x_data = np.array(x_train.values,dtype=np.float32)

y_data = np.array(y_train.values,dtype=np.float32)



X = tf.placeholder('float', [None, 10])

Y = tf.placeholder('float', [None, 3])



W1 = tf.get_variable("W1", shape=[10, 256], initializer=xaver_init(10, 256))

W2 = tf.get_variable("W2", shape=[256, 256], initializer=xaver_init(256, 256))

W3 = tf.get_variable("W3", shape=[256, 3], initializer=xaver_init(256, 3))



B1 = tf.Variable(tf.random_normal([256]))

B2 = tf.Variable(tf.random_normal([256]))

B3 = tf.Variable(tf.random_normal([3]))



L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))

L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))



hypo = tf.add(tf.matmul(L2, W3), B3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypo, labels=Y)) # 구현되어잇는 softmax

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost) # 그라디언트 보다 좀더 좋음









init = tf.initialize_all_variables()



with tf.Session() as sess:

    sess.run(init)

    for step in xrange(4001):

        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})

        if step % 100 == 0:

            print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    

    correct_prediction = tf.equal(tf.argmax(hypo, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))

    



    a = sess.run(hypo, feed_dict={X: [[0.3545121845821541,0.35083902671065004,0.4657608918291205,0.78114166586219,0,0,0,1,0,0]]})

    print ("a :", a, sess.run(tf.arg_max(a, 1)))

    

    #테스트셋

    test_ori = pd.read_csv("../input/test.csv")

    test_color = pd.get_dummies(test_ori[["color"]], prefix="")

    id_list = test_ori['id']

    test_ori.drop('id',inplace=True, axis=1)

    test_ori.drop('color',inplace=True, axis=1)

    

    x_test = pd.concat([test_ori, color], axis=1)

    test_data = np.array(x_test.values,dtype=np.float32)

    

    a = sess.run(hypo, feed_dict={X: x_test})

    predict_ori = sess.run(tf.arg_max(a, 1))

    



    def numToName(num):

        if num == 0:

            return 'Ghost'

        elif num == 1:

            return 'Ghoul'

        else:

            return 'Goblin'



    predic = list(map(numToName, predict_ori));

    

    type_field = pd.DataFrame(predic, columns = ['type'])

    

    result = pd.concat([id_list, type_field], axis=1)

    print (result)

    result.to_csv('result.csv', index=False)