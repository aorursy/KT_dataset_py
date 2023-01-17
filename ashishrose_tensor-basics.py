two_D_tensor = [

[1,2,3],

[4,5,6],

[7,8,9]

]



print("Elements along the first axes will be array")

print("Elements along first axes are:",two_D_tensor[0],'and',two_D_tensor[1],'and',two_D_tensor[2])

print("Elements along the second axes will be a value")

print("Elements along second axes are:",two_D_tensor[0][0],'and',two_D_tensor[1][2],'and',two_D_tensor[2][2],'and similarly 6 more values')
two_D_tensor = [

[1,2,3],

[4,5,6],

[7,8,9]

]



#To work with tensor shape we need to need to create to a tensor object So we are using Tensorflow for this:

import tensorflow as tf

t= tf.constant(two_D_tensor)

print("The value of the tensor is: ",t)

print("The type of the tensor is: ",type(t))

print("The shape of the tensor is ",t.shape)
#here we define a constant tensor with the datatype as float32

t = tf.constant([

    [1,1,1,1],

    [2,2,2,2],

    [3,3,3,3]

], dtype=tf.float32) 



print("The shape of the tensor t is: ",tf.constant(t).shape)

reshaped_tensor = tf.reshape(t, [1,12])

print(reshaped_tensor)



reshaped_tensor = tf.reshape(t, [2,6])

print(reshaped_tensor)



reshaped_tensor = tf.reshape(t, [3,4])

print(reshaped_tensor)



print("\nIn the above 3 examples the Rank of the tensor remain unchanged i.e 2")
reshaped_tensor = tf.reshape(t, [2,2,3])

print(reshaped_tensor)

print("The shape of the tensor t is: ",tf.constant(reshaped_tensor).shape,"so the RANK of the new reshaped tensor is:3")
#consider the following tensor

t = tf.constant([

    [1,1,1,1],

    [2,2,2,2],

    [3,3,3,3]

], dtype=tf.float32)







print(tf.reshape(t, [1,12]))

print(tf.reshape(t, [1,12]).shape)



print(tf.squeeze(tf.reshape(t, [1,12])))

print(tf.squeeze(tf.reshape(t, [1,12])).shape)
#creating a flatten function

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
#this is as good as take rows from t2 and add them to t1 row

tf.concat((t1, t2), axis = 0) 
#this is as good as take column from t2 and add them to t1 column

tf.concat((t1, t2), axis = 1)