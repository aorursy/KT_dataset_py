import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')

%matplotlib inline



print("Tensorflow version ", tf.VERSION)

print("Numpy version ", np.__version__)
x = np.linspace(start = -5, stop = 5, num = 200)
y_step  = tf.constant(np.heaviside(x, 0)) # step function



y_sigmoid = tf.sigmoid(x)



y_bipolar_sig = (1 - tf.exp(-x)) / (1 + tf.exp(-x))



y_softmax = tf.nn.softmax(x)



y_tanh = tf.nn.tanh(x)



y_atan = tf.atan(x) # arctan



y_leCun_tanh = (1.7159 * tf.nn.tanh((2/3) * x)) # LeCun's Tanh



y_relu = tf.nn.relu(x)



y_leaky_relu = tf.nn.leaky_relu(x)



y_softplus = tf.nn.softplus(x)
sess = tf.Session()

y_step, y_sigmoid, y_bipolar_sig, y_softmax, y_tanh, y_atan, y_leCun_tanh, y_relu, y_leaky_relu, y_softplus = sess.run([y_step, y_sigmoid, y_bipolar_sig, y_softmax, y_tanh, y_atan, y_leCun_tanh, y_relu, y_leaky_relu, y_softplus])
plt.figure(1, figsize=(14, 12))



plt.subplot(221)

plt.plot(x, y_step, c='k', linestyle='--', label='step')

plt.plot(x, y_sigmoid, c='blue', label='sigmoid')

plt.plot(x, y_bipolar_sig, c='red', label= 'bipolarSigmoid')

plt.ylim((-1.2, 1.2))

plt.legend(loc='best')



plt.subplot(222)

plt.plot(x, y_tanh, c= 'red', label= 'tanh')

plt.plot(x, y_atan, c= 'grey', label= 'arcTan')

plt.plot(x, y_leCun_tanh, c= 'black', label= 'leCunTanh', linestyle= '--')

plt.ylim((-2.0, 2.0))

plt.legend(loc= 'best')



plt.subplot(223)

plt.plot(x, y_relu, c='green', label='relu')

plt.plot(x, y_leaky_relu, c='grey', label='leaky_relu')

plt.plot(x, y_softplus, c='black', label='softplus', linestyle= '--')

plt.ylim((-1.5, 5))

plt.legend(loc='best')



plt.subplot(224)

plt.plot(x, y_softmax, c='black', label='softmax')

plt.legend(loc='best')



plt.show()