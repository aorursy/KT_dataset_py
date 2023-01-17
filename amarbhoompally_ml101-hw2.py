# coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32)
    const = tf.constant([-3.,-1., 0., 1., 6.], dtype=tf.float32)
    # the subtraction of a vector by a scalar yeilds a vector by subtracting the scalar from each element of the vector
    f = -1. * tf.reduce_sum(tf.log(1. + tf.exp(-1. * tf.square(const - x))))
    grad = tf.gradients(f, x)

    x_val = np.arange(-20., 20., 0.1)
    f_val = []
    grad_val = []
    with tf.Session() as sess:
        for v in x_val:
            f_val.append(sess.run(f, feed_dict={x: v}))
            grad_val.append(sess.run(grad, feed_dict={x: v})[0])
    
    fig = plt.figure(0)
    # create a grid of sub-figures of 2 rows and 1 column. And first plot on the first sub-figure.
    ax = fig.add_subplot(211)
    ax.plot(x_val, f_val)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_xlim([-20, 20])
    
    learning_rates = [2,1.95,1.9,1.85,1.8,1.75,1.7,1.65,1.6,1.55,1.5,1.45,1.4,1.35,1.3,1.2,1.1,1]
    min_fx = None
    for learning_rate in learning_rates:
        #learning_rate = 0.1
        x0 = 1.
        steps = 0
        
        with tf.Session() as sess:
            x_current = x0
            f_current = sess.run(f, feed_dict={x: x_current})
            if min_fx is None:
                min_fx = f_current
            while True:
                # the value returned by sess.run(grad) is a vector with a single element
                grad_current = sess.run(grad, feed_dict={x: x_current})[0]
                x_next = x_current - learning_rate * grad_current
                f_next = sess.run(f, feed_dict={x: x_next})
                steps += 1
                
                if f_next < f_current:
                    f_current = f_next
                    x_current = x_next
                else:
                    #print ('for learning rate {l:.6f} best x after {n} steps: {x:.4f} and current f value is . {f:.20f}'.format(x=x_current, n=steps, f=f_current, l = learning_rate))
                    break
            if min_fx > f_current:
                min_fx = f_current
                result_x = x_current
                result_lr = learning_rate 

    print ('for learning rate {l:.6f} best x: {x:.4f} and  f(x) value is . {f:.20f}'.format(x=result_x, f=min_fx, l = result_lr))
    
    learning_rate = result_lr
    x0 = 1.
    steps = 0
    x_list = []
    f_list = []
    with tf.Session() as sess:
        x_current = x0
        f_current = sess.run(f, feed_dict={x: x_current})
        while True:
            # the value returned by sess.run(grad) is a vector with a single element
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
                #print ('for learning rate {l:.6f} best x after {n} steps: {x:.4f} and current f value is . {f:.20f}'.format(x=x_current, n=steps, f=f_current, l = learning_rate))
                break
    
    # hold the sub-figure to overlap the scattered points on to the plot of function f

    ax.scatter(x_list[::10], f_list[::10], c='r')
    ax = fig.add_subplot(212)
    ax.plot(x_val, grad_val)
    ax.set_xlabel('x')
    ax.set_ylabel('gradient')
    ax.set_xlim([-20, 20])

# coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32)
    const = tf.constant([-3.,-1., 0., 1., 6.], dtype=tf.float32)
    # the subtraction of a vector by a scalar yeilds a vector by subtracting the scalar from each element of the vector
    f = -1. * tf.reduce_sum(tf.log(1. + tf.exp(-1. * tf.square(const - x))))
    grad = tf.gradients(f, x)

    x_val = np.arange(-20., 20., 0.1)
    f_val = []
    grad_val = []
    with tf.Session() as sess:
        for v in x_val:
            f_val.append(sess.run(f, feed_dict={x: v}))
            grad_val.append(sess.run(grad, feed_dict={x: v})[0])
    
    fig = plt.figure(0)
    # create a grid of sub-figures of 2 rows and 1 column. And first plot on the first sub-figure.
    ax = fig.add_subplot(211)
    ax.plot(x_val, f_val)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_xlim([-20, 20])
    
    learning_rate = 0.1
    x0 = -10.
    steps = 0
    x_list = []
    f_list = []
    with tf.Session() as sess:
        x_current = x0
        f_current = sess.run(f, feed_dict={x: x_current})
        while True:
            # the value returned by sess.run(grad) is a vector with a single element
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
                print ('for learning rate {l:.6f} best x after {n} steps: {x:.4f} and current f value is . {f:.20f}'.format(x=x_current, n=steps, f=f_current, l = learning_rate))
                break
    
    # hold the sub-figure to overlap the scattered points on to the plot of function f

    ax.scatter(x_list[::10], f_list[::10], c='r')
    ax = fig.add_subplot(212)
    ax.plot(x_val, grad_val)
    ax.set_xlabel('x')
    ax.set_ylabel('gradient')
    ax.set_xlim([-20, 20])
