import tensorflow as tf

'''creating a variable'''
x = tf.Variable(1 , name = 'x')
'''checking whether the variable is stored as node value in the default graph.'''
x.graph is tf.get_default_graph()
'''creating a temporary graph'''
graph = tf.Graph()
with graph.as_default():
    '''Creating a variable in temporary graph.'''
    x1 = tf.Variable(2 , name = 'x1')
    
x1.graph is graph 
x1.graph is tf.get_default_graph()
'''first lets reset the graph'''
tf.reset_default_graph()

'''variables'''
w = tf.constant(2)
x = w + 3
y = x + 5
z = x * 3

'''initializing a session'''
with tf.Session() as session:
    print(y.eval()) #10
    print(z.eval()) #15
with tf.Session() as session :
    '''Commanding TensorFlow to evaluate y and z , without evaluating w and x twice'''
    y_val , z_val = session.run([y , z ])
    print(y_val)
    print(z_val)
'''importing the Boston House pricing dataset'''
from sklearn.datasets import load_boston
housing = load_boston()
import numpy as np

m , n = housing.data.shape
'''adding or concating bias input feature'''
housing_plus_bias = np.c_[np.ones( (m , 1) ) , housing.data]
X = tf.constant(housing_plus_bias , dtype = tf.float32 , name = "X")
y = tf.constant(housing.target.reshape(-1 , 1) , dtype = tf.float32 , name = "y")

'''even create X transpose which will be usefull for theta evaluation.'''
XT = tf.transpose(X)
'''This line of code does not performs any kin of computaion , instead it creates a nodes
in the graph'''
theta = tf.matmul( tf.matmul( tf.matrix_inverse( tf.matmul(XT , X ) ) , XT ) , y)

'''Initializing session'''
with tf.Session() as session:
    '''evaluating theta'''
    theta_value = theta.eval()
print(theta_value)