import warnings
warnings.filterwarnings('ignore')
# Import tensorflow 
import tensorflow as tf
print("version: "+tf.__version__)
# lets see Hello world example
hello = tf.constant("Hello ")
world = tf.constant("world!")
# lets print
print(hello+world)
# lets have a look on type of hello and world variables
print(type(hello))
print(type(world))
with tf.Session() as sess:
    print( sess.run(hello+world) ) # concatenation
# lets play with numbers
a = tf.constant(5)
b = tf.constant(10)
type(a)
with tf.Session() as sess:
    print( sess.run(a+b) )
# A 4X4 Matrix with all values 10
matA = tf.fill((4,4),10) 
with tf.Session() as sess:
    print( sess.run(matA) )
# Zero values matrix
matB = tf.zeros((2,2))
with tf.Session() as sess:
    print( sess.run(matB) )
# Normal distribution Matrix
matN = tf.random_normal((2,2),mean=0,stddev=1.0)

with tf.Session() as sess:
    print(sess.run(matN))
# Uniform Random Distribution
matU = tf.random_uniform((2,2),minval=10,maxval=100)
with tf.Session() as sess:
    print(sess.run(matU))
ISess = tf.InteractiveSession()

print( ISess.run(tf.zeros((5,5)) ) )
# or
ones = tf.ones((5,5))
print( ones.eval() ) 

# Simple Matrix example
matS1 = tf.constant([[10,5],
                    [3, 9]  ])
matS2 = tf.constant([[10,5],
                    [3, 9]  ])

print( matS1.eval() )
print( matS1.get_shape())

print(tf.matmul(matS1,matS2).eval())
tf.get_default_graph
print( tf.constant(0) )
print( tf.constant(0, name="c") )
# Other than default graph
d = tf.get_default_graph
g = tf.Graph()
g
g is tf.get_default_graph
d is tf.get_default_graph
tensorVar = tf.Variable( initial_value=tf.zeros((2,2)) )

# initialize the variables
init = tf.global_variables_initializer()
init.run()

print( tensorVar )
print(tensorVar.eval())
ph = tf.placeholder(tf.float32) 
ph
matph1 = tf.placeholder(tf.int32, shape=(3,3))
matph2 = tf.placeholder(tf.int32, shape=(None,3))
# None is for no. of rows or examples in dataset

print(matph1)
print(matph2)
import numpy as np #for linear algebra 
np.random.seed(101) # It can be called again to re-seed the generator
tf.set_random_seed(101)
rand_a = np.random.uniform(0,100,(5,5))
rand_b = np.random.uniform(0,100,(5,1))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = a+b # tf.add(a,b)
mult_op = a*b #tf.multiply(a,b)
with tf.Session() as sess:
    s = sess.run( add_op, feed_dict={ a:rand_a, b:rand_b} )
    print(s)
    
    m = sess.run( mult_op, feed_dict={a:rand_a, b:rand_b } )
    print(m)
n_features = 10
n_dense_neurons = 3
# Placeholder for x
x = tf.placeholder(tf.float32,(None,n_features))

# Variables for w and b
b = tf.Variable(tf.zeros([n_dense_neurons]))

W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))
# Activation func
xW = tf.matmul(x,W)
z = tf.add(xW,b)
# tf.nn.relu() or tf.tanh()
a = tf.sigmoid(z)

# Variable Intialization
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(z,feed_dict={x : np.random.random([1,n_features])}))
    layer_out = sess.run(a,feed_dict={x : np.random.random([1,n_features])})

print(layer_out)
np.linspace(1,10,10)
# Artificial Data
xdata = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
print(xdata)
ydata = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
print(ydata)
m = tf.Variable(0.39)
b = tf.Variable(0.2)
# Cost or Residual 

error = 0
for x,y in zip(xdata,ydata):
    y_ = m*x + b
    error += (y-y_)**2

print(error)
# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
train
init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 100
    
    for i in range(epochs):
        
        sess.run(train)
        

    # Fetch Back Results
    final_slope , final_intercept = sess.run([m,b])
print(final_slope)
print(final_intercept)
import matplotlib.pyplot as plt

x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'r')

plt.plot(xdata,ydata,'*')
plt.show()