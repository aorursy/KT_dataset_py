# coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32)
    const = tf.constant([1., 3., 6., 10.], dtype=tf.float32)
    f = -1. * tf.reduce_sum(tf.log(1. + tf.exp(-0.5 * tf.square(const - x))))
    grad = tf.gradients(f, x)

    x_val = np.arange(-5., 15., 0.1)
    f_val = []
    grad_val = []
    with tf.Session() as sess:
        for v in x_val:
            f_val.append(sess.run(f, feed_dict={x: v}))
            grad_val.append(sess.run(grad, feed_dict={x: v})[0])
    fig = plt.figure(0)
    ax = fig.add_subplot(211)
    ax.plot(x_val, f_val)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_xlim([-5, 15])

    learning_rate = 0.1
    x0 = 0.
    steps = 0
    x_list = []
    f_list = []
    with tf.Session() as sess:
        x_current = x0
        f_current = sess.run(f, feed_dict={x: x_current})
        while True:
            grad_current = sess.run(grad, feed_dict={x: x_current})[0]
            x_next = x_current - learning_rate * grad_current
            f_next = sess.run(f, feed_dict={x: x_next})
            steps += 1
            if f_next < f_current:
                f_current = f_next
                x_current = x_next
                x_list.append(x_next)
                f_list.append(f_next)
            else:
                print('best x after {n} steps: {x:.4f}'.format(x=x_current, n=steps))
                break

    ax.hold(True)
    ax.scatter(x_list[::10], f_list[::10], c='r')
    ax.hold(False)
    ax = fig.add_subplot(212)
    ax.plot(x_val, grad_val)
    ax.set_xlabel('x')
    ax.set_ylabel('gradient')
    ax.set_xlim([-5, 15])
    fig.show()

   # input('type to exit ...')
