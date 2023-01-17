import tensorflow as tf

import numpy as np
a = [1,2,3,4]
a[2]
#As another example, suppose we have this 2d-array:

 

dd = [

[1,2,3],

[4,5,6],

[7,8,9]

] 
dd[0][2]
dd = [

[1,2,3],

[4,5,6],

[7,8,9]

]
dd[0]
dd[1]
dd[2]
dd[0][0]
dd[1][0]
dd[2][0]
dd[0][1]
dd[1][1]
dd[2][1]
dd[0][2]
dd[1][2]
dd[2][2]
# Let's say we have a 2D array

dd = [

[1,2,3],

[4,5,6],

[7,8,9]

]

#To work with this tensor's shape, we’ll create a tensor object like so:



t = tf.constant(dd)      # constant() is a function that helps you create a constant tensor

t
type(t)   # To get the type of object t
# Now, we have a Tensor object, and so we can ask to see the tensor's shape:



t.shape
t = tf.constant([

    [1,1,1,1],

    [2,2,2,2],

    [3,3,3,3]

], dtype=tf.float32)    # can also mention the type of the element we want in the tensor
reshaped_tensor = tf.reshape(t, [1,12])     # reshape is a function that helps to reshaping any ndarray or ndtensor

print(reshaped_tensor)
reshaped_tensor = tf.reshape(t, [2,6])

print(reshaped_tensor)
reshaped_tensor = tf.reshape(t, [3,4])

print(reshaped_tensor)
reshaped_tensor = tf.reshape(t, [2,2,3])

print(reshaped_tensor)
print(tf.reshape(t, [1,12]))
print(tf.reshape(t, [1,12]).shape)
print(tf.squeeze(tf.reshape(t, [1,12])))
print(tf.squeeze(tf.reshape(t, [1,12])).shape)
def flatten(t):

    t = tf.reshape(t, [1, -1])

    t = tf.squeeze(t)

    return t
t = tf.ones([4, 3])

t
flatten(t)
t1 = tf.constant([

    [1,2],

    [3,4]

])



t2 = tf.constant([

    [5,6],

    [7,8]

])
tf.concat((t1, t2), axis = 0)  # concat() helps you to concatenate two tensors according to the given axis
tf.concat((t1, t2), axis = 1)
tf.concat((t1, t2), axis = 0).shape
tf.concat((t1, t2), axis = 1).shape