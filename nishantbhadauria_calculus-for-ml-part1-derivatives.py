import matplotlib.pyplot as plt
x = range(-10,10)
y = [x*x for x in x]
plt.scatter(x, y)
plt.show()
import matplotlib.pyplot as plt
x = range(-10,10)
y = [max(x,0) for x in x]
plt.scatter(x, y)
plt.show()
from sympy import *
x = Symbol('x')
f = x**2
fdash = f.diff(x)
fdash
from sympy import *
x = Symbol('x')
f = 1/x
fdash = f.diff(x)
fdash
import matplotlib.pyplot as plt
x = range(-10,10)
y = [1/x for x in x]
plt.scatter(x, y)
plt.show()
import tensorflow as tf
import numpy
x = tf.Variable([10.0])

with tf.GradientTape() as tape:
  y = x**2
fdash=tape.gradient(y, x)
fdash.numpy()
import tensorflow as tf
#Function with inputs x
def fu(x):
    return x ** 3.0 
#Reset the values of x
def reset():
    x = tf.Variable(10.0) 
    return x
x = reset()
for i in range(50):
    #Find derivative of x with respect to y using auto differentiation
    with tf.GradientTape() as tape:
        y = fu(x)
grads = tape.gradient(y, [x])

print (grads[0].numpy)
#Update x
newx=x.assign(x - 0.1*grads[0].numpy())
newx.numpy()
##chain rule###
from sympy import * 
f = symbols('f', cls=Function)
t= symbols('t')
x =  t**2
y = sin(t)
g = f(x,y)
Derivative(g,t).doit()
##sum rule###
from sympy import * 
f = symbols('f', cls=Function)
t= symbols('t')
x =  t**2
y = sin(t)
g = f(x)+f(y)
Derivative(g,t).doit()
#We are taking example of classical MNIST dataset which has 784 images of 0-9 digits. We have first starting input matrix of 784 
#columns and output of 10 categories y
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")
a_0 = tf.compat.v1.placeholder(tf.float32, shape=(None, 784))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 10))

# here is am constructing one hidden layers with w1 as weight 1 and b1 as bais 1 which passes the network to next layer
# w2 is output layer which has b2 as bais of 1 and output the 10 prediction 
middle = 50
w_1 = tf.Variable(tf.random.truncated_normal([784, middle]))
b_1 = tf.Variable(tf.random.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.random.truncated_normal([middle, 10]))
b_2 = tf.Variable(tf.random.truncated_normal([1, 10]))

# this is the activation function sigmoid which is 1/1+exp^(-x)
def sigmoid(x):
    return tf.compat.v1.div(tf.constant(1.0),
                  tf.compat.v1.add(tf.constant(1.0), tf.exp(tf.negative(x))))


z_1 = tf.compat.v1.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigmoid(z_1)
z_2 = tf.compat.v1.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigmoid(z_2)
diff = tf.subtract(a_2, y)##This becomes our cost function
## Derivative of the cost function###
def sigmaprime(x):
    return tf.multiply(sigmoid(x), tf.subtract(tf.constant(1.0), sigmoid(x)))
## we multiply diff with derivative of cost function , this becomes the input backward to the hidden layer
d_z_2 = tf.multiply(diff, sigmaprime(z_2))
d_b_2 = d_z_2
## Now the updated weight will be mutiplication of this new quantity into transpose of a1 because we are propogating backward 
#hence the dimension of matrix are reversed.
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)
d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
d_b_1 = d_z_1
d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)
print(d_w_1,d_z_1)
import sympy as sym
vars = sym.symbols('x1 x2') # Define x1 and x2 variables
f = sym.sympify(['x1**2', 'ln(x2)']) # Define functions of x1 and x2
J = sym.zeros(len(f),len(vars)) # Initialise Jacobian matrix
# Fill Jacobian matrix with entries
for i, fi in enumerate(f):
    for j, s in enumerate(vars):
        J[i,j] = sym.diff(fi, s)
print(J)
print(sym.Matrix.det(J))
from sympy import Function, hessian, pprint
from sympy.abc import x, y,z
g2 = x**3 + y**2+z
pprint(hessian(g2, (x, y,z)))

