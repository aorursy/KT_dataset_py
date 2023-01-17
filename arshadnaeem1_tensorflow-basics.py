#Examples

import tensorflow as tf



#Scalar

a = tf.constant(3)



#Vector

b = tf.constant([3,4,5])



#Matrix

c = tf.constant([[1,2,3],[4,5,6]])



#3dtensor

d = tf.constant([[[1,2,3],[4,5,6]]])



print(a)

print(b)

print(c)

print(d)



#Scalar

a = tf.constant(3,name='a')



#Vector

b = tf.constant([3,4,5],name='b')



#Matrix

c = tf.constant([[1,2,3],[4,5,6]],name='c')



#3d Tensor

d = tf.constant([[[1,2,3],[4,5,6]]],name='d')



print(a)

print(b)

print(c)

print(d)
import tensorflow as tf



a = tf.constant(3.0,name='a')

b = tf.constant(5.0,name='b')



c = tf.multiply(a,b,name='multiply')

#c = a*b



print(c)
import numpy as np



A = 3.0

B = 5.0



C = np.multiply(A,B)

#C=a*b



print (C)
#Graph and Session



#Building Graph

x = tf.constant(2,name='x')

y = tf.constant(3,name='y')

z = tf.add(tf.add(tf.multiply(tf.pow(x,2),y),y),2)

#z = (x**2)*y+y+2

print (z)



#Executing Graph

with tf.Session() as sess:

  print (sess.run(z))
#Option 1



sess = tf.Session()

print (sess.run(z))

sess.close()



#Option2



with tf.Session() as sess:

  print (sess.run(z))



#Option3



with tf.Session() as sess:

  print (z.eval()) 
#Another Example

import tensorflow as tf



#Input Data

x = tf.constant([3,4], name="x", dtype=tf.float32)

x = tf.reshape(x,[1,2])



#Weight Matrix

w = tf.constant([0.5,0.5], name="w", dtype=tf.float32)

w = tf.reshape(w,[2,1])



#Bias

b = tf.constant(0.3, name="b", dtype=tf.float32)



#Forward Propagation

y = tf.add(tf.matmul(x, w), b)



with tf.Session() as session:

    print (session.run(y))
import tensorflow as tf



#Example 1

value=[1,2,3]

a=tf.constant(value, dtype=tf.int32, shape=[1,3], name='Const', verify_shape=False)

print(a)



with tf.Session() as sess:

  print(sess.run(a))



#Example2

value=[[1,2,3]]

b=tf.constant(value, dtype=tf.int32, shape=[1,3], name='Const', verify_shape=True)

print(b)



with tf.Session() as sess:

  print(sess.run(b))



#Example3

with tf.Session() as sess:

  print(sess.run([a,b]))



with tf.Session() as sess:

  a,b=sess.run([a,b])  

print (a,b)
#Examples



s = tf.Variable(2, name="scalar") 

m = tf.Variable([[1, 2], [3, 4]], name="matrix") 

W = tf.Variable(tf.zeros([784,10]),name="weights")



print(s)

print(m)

print(W)
#tf.get_variable(name='NA',

 #               shape=None,

 #               dtype=None,

 #               initializer=None,

 #               regularizer=None,

 #               trainable=True,

 #               collections=None,

 #               caching_device=None,

 #               partitioner=None,

 #               validate_shape=True,

 #               use_resource=None,

 #               custom_getter=None,

 #               constraint=None)
s = tf.get_variable("scalar", initializer=tf.constant(2)) 

m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))

W = tf.get_variable("weight_matrix", shape=(784, 10), initializer=tf.zeros_initializer())



print(s)

print(m)

print(W)
#Attempting to use unitialized variable

import tensorflow as tf

a = tf.get_variable(name="variable1", initializer=tf.constant(2))

b = tf.get_variable(name="variable2", initializer=tf.constant(3))

c = tf.add(a, b, name="add2")



# launch the graph in a session (#It will result in an error.Uncomment the sess.run to see)

#with tf.Session() as sess:

    # now let's evaluate their value

    #print(sess.run(a))

    #print(sess.run(b))

    #print(sess.run(c))
#Initaling all variables with an operation

import tensorflow as tf

a = tf.get_variable(name="A", initializer=tf.constant(2))

b = tf.get_variable(name="B", initializer=tf.constant(3))

c = tf.add(a, b, name="Add")

# add an Op to initialize global variables

init_op = tf.global_variables_initializer()



# launch the graph in a session

with tf.Session() as sess:

    # run the variable initializer operation

    sess.run(init_op)

    # now let's evaluate their value

    print(sess.run(a))

    print(sess.run(b))

    print(sess.run(c))
import tensorflow as tf

# create graph

weights = tf.get_variable(name="W", shape=[2,3], initializer=tf.truncated_normal_initializer(stddev=0.01))

biases = tf.get_variable(name="b", shape=[3], initializer=tf.zeros_initializer())



# add an Op to initialize global variables

init_op = tf.global_variables_initializer()



# launch the graph in a session

with tf.Session() as sess:

    # run the variable initializer

    sess.run(init_op)

    # now we can run our operations

    W, b = sess.run([weights, biases])

    print('weights = {}'.format(W))

    print('biases = {}'.format(b))
import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[5])

b = tf.placeholder(dtype=tf.float32, shape=None, name=None)

X = tf.placeholder(tf.float32, shape=[None, 784], name='input')

Y = tf.placeholder(tf.float32, shape=[None, 10], name='label')
import tensorflow as tf

a = tf.constant([5, 5, 5], tf.float32, name='A')

b = tf.placeholder(tf.float32, shape=[3], name='B')

c = tf.add(a, b, name="Add")



with tf.Session() as sess:

    # create a dictionary:

    d = {b: [1, 2, 3]}

    # feed it to the placeholder

    print(sess.run(c, feed_dict={b:[1,2,3]}))

    

with tf.Session() as sess:

    # create a dictionary:

    d = {b: [1, 2, 3]}

    # feed it to the placeholder

    print(c.eval(feed_dict={b:[1,2,3]}))
#Gradient Descent example: Minimize x**2-10*x+25



import tensorflow as tf



#Define x as variable and initialize with zero

y = tf.get_variable(name='s4',initializer=tf.constant(0.5))



#Define Function

z = y**2-10*y + 25



#Create Optimizer operation



gd = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(z)



#Execute the graph

sess = tf.Session()

sess.run(tf.global_variables_initializer())



for i in range(10):

  sess.run(gd)

  print ("Epoch" + str(i+1) + ": " + str(sess.run(y)) + "," + str(sess.run(z)))
x = tf.constant(2.0,name='x')

y = tf.constant(3.0,name='x')



a = tf.add(x,y)

b = tf.subtract(x,y)

c = tf.multiply(x,y)

d = tf.div(x,y)

e = tf.mod(x,y)

f = tf.abs(x)

g = tf.negative(x)

h = tf.sign(x)

i = tf.square(x)

j = tf.round(x)

k = tf.sqrt(x)

l = tf.pow(x,y)

m = tf.exp(x)

n = tf.log(x)

o = tf.maximum(x,y)

p = tf.minimum(x,y)

q = tf.cos(x)

r = tf.sin(x)
import tensorflow as tf

import numpy as np



def convert(v, t=tf.float32):

    return tf.convert_to_tensor(v, dtype=t)



m1 = tf.random.uniform(shape=[4,4],minval=0,maxval=1,dtype=tf.float32)

m2 = tf.random.uniform(shape=[4,4],minval=0,maxval=1,dtype=tf.float32)

m3 = tf.random.uniform(shape=[4,4],minval=0,maxval=1,dtype=tf.float32)

m4 = tf.random.uniform(shape=[4,4],minval=0,maxval=1,dtype=tf.float32)

m5 = tf.random.uniform(shape=[4,4],minval=0,maxval=1,dtype=tf.float32)



m_tranpose = tf.transpose(m1)

m_mul = tf.matmul(m1, m2)

m_det = tf.matrix_determinant(m3)

m_inv = tf.matrix_inverse(m4)

m_solve = tf.matrix_solve(m5, [[1], [1], [1], [1]])



with tf.Session() as session:

    print (session.run(m_tranpose))

    print (session.run(m_mul))

    print (session.run(m_inv))

    print (session.run(m_det))

    print (session.run(m_solve))
import tensorflow as tf

import numpy as np



def convert(v, t=tf.float32):

    return tf.convert_to_tensor(v, dtype=t)



x = convert(

    np.array(

        [

            (1, 2, 3),

            (4, 5, 6),

            (7, 8, 9)

        ]), tf.int32)



bool_tensor = convert([(True, False, True), (False, False, True), (True, False, False)], tf.bool)



red_sum_0 = tf.reduce_sum(x)

red_sum = tf.reduce_sum(x, axis=1)



red_prod_0 = tf.reduce_prod(x)

red_prod = tf.reduce_prod(x, axis=1)



red_min_0 = tf.reduce_min(x)

red_min = tf.reduce_min(x, axis=1)



red_max_0 = tf.reduce_max(x)

red_max = tf.reduce_max(x, axis=1)



red_mean_0 = tf.reduce_mean(x)

red_mean = tf.reduce_mean(x, axis=1)



red_bool_all_0 = tf.reduce_all(bool_tensor)

red_bool_all = tf.reduce_all(bool_tensor, axis=1)



red_bool_any_0 = tf.reduce_any(bool_tensor)

red_bool_any = tf.reduce_any(bool_tensor, axis=1)





with tf.Session() as session:

    print ("Reduce sum without passed axis parameter: ", session.run(red_sum_0))

    print ("Reduce sum with passed axis=1: ", session.run(red_sum))



    print ("Reduce product without passed axis parameter: ", session.run(red_prod_0))

    print ("Reduce product with passed axis=1: ", session.run(red_prod))



    print ("Reduce min without passed axis parameter: ", session.run(red_min_0))

    print ("Reduce min with passed axis=1: ", session.run(red_min))



    print ("Reduce max without passed axis parameter: ", session.run(red_max_0))

    print ("Reduce max with passed axis=1: ", session.run(red_max))



    print ("Reduce mean without passed axis parameter: ", session.run(red_mean_0))

    print ("Reduce mean with passed axis=1: ", session.run(red_mean))



    print ("Reduce bool all without passed axis parameter: ", session.run(red_bool_all_0))

    print ("Reduce bool all with passed axis=1: ", session.run(red_bool_all))



    print ("Reduce bool any without passed axis parameter: ", session.run(red_bool_any_0))

    print ("Reduce bool any with passed axis=1: ", session.run(red_bool_any))
#from google.colab import drive

#drive.mount('/content/drive')

#from IPython.display import Image

#Image(filename= "/content/drive/My Drive/Colab Notebooks/segment.png",width=500,height=300)
import tensorflow as tf

import numpy as np





def convert(v, t=tf.float32):

    return tf.convert_to_tensor(v, dtype=t)



seg_ids = tf.constant([0, 0, 1, 2, 2])

tens1 = convert(np.array([(2, 5, 3, -5), (0, 3, -2, 5), (4, 3, 5, 3), (6, 1, 4, 0), (6, 1, 4, 0)]), tf.int32)

tens2 = convert(np.array([1, 2, 3, 4, 5]), tf.int32)





seg_sum = tf.segment_sum(tens1, seg_ids)

seg_sum_1 = tf.segment_sum(tens2, seg_ids)





with tf.Session() as session:

    print ("Segmentation sum tens1: ", session.run(seg_sum))

    print ("Segmentation sum tens2: ", session.run(seg_sum_1))
import numpy as np

import tensorflow as tf



def convert(v, t=tf.float32):

    return tf.convert_to_tensor(v, dtype=t)



x = convert(np.array([

    [2, 2, 1, 3],

    [4, 5, 6, -1],

    [0, 1, 1, -2],

    [6, 2, 3, 0]

]))



y = convert(np.array([1, 2, 5, 3, 7]))

z = convert(np.array([1, 0, 4, 6, 2]))



arg_min = tf.argmin(x, 1)

arg_max = tf.argmax(x, 1)

unique = tf.unique(y)

diff = tf.setdiff1d(y, z)



with tf.Session() as session:

    print ("Argmin = ", session.run(arg_min))

    print ("Argmax = ", session.run(arg_max))



    print ("Unique_values = ", session.run(unique)[0])

    print ("Unique_idx = ", session.run(unique)[1])



    print ("Setdiff_values = ", session.run(diff)[0])

    print ("Setdiff_idx = ", session.run(diff)[1])



    print (session.run(diff)[1])