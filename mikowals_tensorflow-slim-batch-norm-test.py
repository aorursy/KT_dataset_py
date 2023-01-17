# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow as tf

import tensorflow.contrib.slim as slim



data = tf.random_normal(shape=(64, 16, 16, 3), seed=10)



conv_bn = slim.conv2d(data, 16, (3,3), activation_fn=None,

                      normalizer_fn=slim.batch_norm, normalizer_params={'is_training': True}, 

                      scope='conv')



conv_no_bn = slim.conv2d(data, 16, (3,3), activation_fn=None, biases_initializer=None,

                            reuse=True, scope='conv')



alt_conv_bn = slim.batch_norm(conv_no_bn, is_training=True)



is_equal = tf.reduce_all(tf.equal(conv_bn, alt_conv_bn))

moving_variances = tf.contrib.framework.get_variables_by_suffix('moving_variance')

variance_is_equal = tf.reduce_all(tf.equal(*moving_variances))



moving_means = tf.contrib.framework.get_variables_by_suffix('moving_mean')

mean_is_equal = tf.reduce_all(tf.equal(*moving_means))





with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    print(sess.run(is_equal)) #True

    print(sess.run(is_equal)) #True

    print([v.name for v in moving_variances])

    print([v.name for v in moving_means])

    print(sess.run([variance_is_equal, mean_is_equal]))