import tensorflow as tf

import numpy as np

np.random.seed(0)

print(tf.VERSION)
matrix_1 = tf.placeholder(dtype=tf.float32, name="matrix_1")

matrix_2 = tf.placeholder(dtype=tf.float32, name="matrix_2")
# defining the session

sess = tf.InteractiveSession()
mat_1 = np.random.randint(low = 1, high = 10, size = (4, 4))

data = {matrix_1: mat_1}

print("Shape of \n{} is \n{}".format(mat_1, tf.shape(matrix_1).eval(feed_dict= data)))
print("Size of \n{} is \n{}".format(mat_1, tf.size(matrix_1).eval(feed_dict= data)))
print("Rank of \n{} is \n{}".format(mat_1, tf.rank(matrix_1).eval(feed_dict= data)))
print("Reshape \n{} of shape \n{} into \n{} of shape \n{}".format(mat_1, 

                                                                  tf.shape(matrix_1).eval(feed_dict= data), tf.reshape(matrix_1, 

                                                                                                                       shape = [2,8]).eval(feed_dict= data), tf.shape(tf.reshape(matrix_1, 

                                                                                                                       shape = [2,8])).eval(feed_dict = data)))
mat_2 = np.random.randint(low = 1, high = 10, size = (1,4))

print("Squeeze \n{} of shape \n{} into \n{} of shape \n{}".format(mat_2, 

                                                                  tf.shape(matrix_2).eval(feed_dict={matrix_2: mat_2}), tf.squeeze(matrix_2).eval(feed_dict= {matrix_2: mat_2}), tf.shape(tf.squeeze(matrix_2)).eval(feed_dict = {matrix_2: mat_2}))) 
print("Expand Dimension of matrix \n{} of shape \n{} into matrix \n{} of shape \n{}".format(mat_2,

                                                                                            tf.shape(matrix_2).eval(

                                                                                                feed_dict= {matrix_2: mat_2}), tf.expand_dims(matrix_2, 0).eval(

                                                                                                feed_dict={matrix_2: mat_2}), tf.shape(tf.expand_dims(matrix_2, 0)).eval(feed_dict={matrix_2: mat_2})))
mat_3 = ([[[1, 1, 1], [2, 2, 2]],

                     [[3, 3, 3], [4, 4, 4]],

                    [[5, 5, 5], [6, 6, 6]]])



print("Slice matrix \n{} of shape \n{} into matrix \n{} of shape \n{}".format(mat_3, 

                                                                              tf.shape(matrix_2).eval(feed_dict={matrix_2: mat_3}), tf.slice(

                                                                                  matrix_2, [1,0, 0], [1, 1, 3]).eval(feed_dict={matrix_2: mat_3}), tf.shape(

                                                                                  tf.slice(matrix_2, [1,0, 0], [1, 1, 3])).eval(feed_dict={matrix_2: mat_3})))
split0, split1 = tf.split(mat_1, num_or_size_splits = [2, 2], axis = 1)

# print(tf.shape(split0).eval(feed_dict={matrix_1: mat_1}))

# print("Split0 is of shape {}".format(tf.shape(split0).eval()))

# split0.eval()

print("A matrix \n{} of shape \n{} is split into 2 matrices \n{} of shape \n{} and \n{} of shape \n{} ".format(

    mat_1, tf.shape(mat_1).eval(), split0.eval(), tf.shape(split0).eval(), split1.eval(), tf.shape(split1).eval()))
mat_4 = np.random.randint(low= 1, high= 10, size = 4)

print("Replicating \n{} into \n{}".format(mat_4, tf.tile(mat_4, multiples=[2]).eval()))
mat_5 = np.random.randint(low=1, high=10, size = (2,4))

mat_6 = np.random.randint(low=1, high=10, size= (2,4))

concat = tf.concat(values = [mat_5, mat_6], axis = 0)

print("Concat matrix \n{}  and \n{} into \n{}".format(mat_5, mat_6, concat.eval()))
concat_1 = tf.concat(values = [mat_5, mat_6], axis = 1)

print("Concat matrix \n{} and \n{} into \n{}".format(mat_5, mat_6, concat_1.eval()))
print("Reversing the matrix \n{} into \n{}".format(mat_5, tf.reverse(mat_5, axis=[0]).eval()))
print("Reversing the matrix \n{} into \n{}".format(mat_5, tf.reverse(mat_5, axis=[1]).eval()))
print("A matrix \n{} is gather by indices \n{}".format(mat_4, tf.gather(mat_4, indices=[1,0,3]).eval()))