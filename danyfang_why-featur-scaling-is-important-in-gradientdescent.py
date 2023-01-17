import tensorflow as tf

import numpy as np

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt



boston=load_boston()

x_train, y_train = boston.data, boston.target.reshape(-1,1)
weights = np.array([-1.08011358e-01, 4.64204584e-02,  2.05586264e-02,  2.68673382e+00,

                    -1.77666112e+01,  3.80986521e+00,  6.92224640e-04, -1.47556685e+00,

                    3.06049479e-01, -1.23345939e-02, -9.52747232e-01,  9.31168327e-03,

                    -5.24758378e-01])

bias = np.array([36.459488385090246])
from sklearn import metrics

pred = np.matmul(x_train, weights) + bias

print(np.mean(np.square(y_train-pred)))

print(metrics.mean_squared_error(y_train, pred))
print(pred.shape)

print(y_train.shape)

print((y_train- pred).shape)

print((pred-y_train).shape)

print(np.ndim(pred))

print(np.ndim(y_train))
print(np.mean(np.square(y_train-pred.reshape(-1,1))))

print(metrics.mean_squared_error(y_train, pred))
tf.reset_default_graph()



W=tf.Variable(tf.zeros((13,1)), dtype=tf.float32)

b=tf.Variable(0.0)

X=tf.placeholder(tf.float32, shape=(None,x_train.shape[-1]), name='input')

Y=tf.placeholder(tf.float32, shape=(None,1), name='ouput')



Y_= tf.matmul(X, W) + b



loss=tf.reduce_mean(tf.square(Y_-Y))

optimizer = tf.train.GradientDescentOptimizer(0.000001)

train=optimizer.minimize(loss)

init=tf.global_variables_initializer()



with tf.Session() as sess:

    epochs=1000

    sess.run(init)

    points=[ [],[] ]

    for i in range(epochs):

        if(i%100==0):

            print(i,sess.run(loss,feed_dict={X: x_train,Y:y_train}))

        sess.run(train,feed_dict={X: x_train,Y:y_train})

        if(i%2==0):

            points[0].append(1+i)

            points[1].append(sess.run(loss,feed_dict={X: x_train,Y:y_train})) 

    plt.plot(points[0],points[1],'r--')

    plt.axis([0,epochs,0,600])#

    plt.show()
from sklearn.preprocessing import StandardScaler 



scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
optimizer = tf.train.GradientDescentOptimizer(0.1)

train=optimizer.minimize(loss)



with tf.Session() as sess:

    epochs=300

    sess.run(init)

    points=[ [],[] ]

    for i in range(epochs):

        if(i%100==0):

            print(i,sess.run(loss,feed_dict={X: x_train,Y:y_train}))

        sess.run(train,feed_dict={X: x_train,Y:y_train})

        if(i%2==0):

            points[0].append(1+i)

            points[1].append(sess.run(loss,feed_dict={X: x_train,Y:y_train})) 

    plt.plot(points[0],points[1],'r--')

    plt.axis([0,epochs,0,600])#

    plt.show()