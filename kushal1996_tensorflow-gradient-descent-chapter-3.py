import numpy as np 
import tensorflow as tf 
from sklearn.datasets import load_boston

'''For Normalizing the features i am using sklearns StandardScaler , you can use 
numpy or tensorflow to.'''
from sklearn.preprocessing import StandardScaler 

housing = load_boston()
data = housing.data 

'''normalizing the features'''
normalizer = StandardScaler()
normalizer.fit(data)
data = normalizer.transform(data)

'''adding or concating the bias input feature vector'''
m , n = data.shape
data_plus_bias = np.c_[np.ones( (m , 1) ) , data]

X = tf.constant(data_plus_bias , dtype = tf.float32 , name = 'X')
y = tf.constant(housing.target.reshape(-1 , 1) , dtype = tf.float32 , name = 'y')
'''number of iterations to converge theta'''
n_epochs = 1000 

'''Learning rate alpha'''
learning_rate = 0.01

'''random initialization of theta'''
theta = tf.Variable(tf.random_uniform([n + 1 , 1] , -1.0 , 1.0) , name = 'theta')

'''batch gradient descent'''
y_pred = tf.matmul(X , theta , name = 'predictions')
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error) , name = 'mse')
gradients = 2/m * tf.matmul(tf.transpose(X) , error)
training_op = tf.assign(theta , theta - learning_rate * gradients)


init = tf.global_variables_initializer()
with tf.Session() as session:
    '''initializing variables'''
    session.run(init)
    
    '''converging theta'''
    for epoch in range(n_epochs):
        if epoch % 100 == 0  :
            print('Epoch' , epoch , 'MSE =',mse.eval() )
        
        session.run(training_op)
    
    best_theta = theta.eval()
    print('best theta\n' , best_theta)
n_epochs = 1000 
learning_rate = 0.01

theta = tf.Variable(tf.random_uniform([n + 1 , 1] , -1.0 , 1.0) , name = 'theta')

y_pred = tf.matmul(X , theta , name = 'predictions')
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error) , name = 'mse')
'''using autodiff to compute gradients'''
gradients = tf.gradients(mse , [theta])[0]
training_op = tf.assign(theta , theta - learning_rate * gradients)


init = tf.global_variables_initializer()
with tf.Session() as session:
    '''initializing variables'''
    session.run(init)
    
    '''converging theta'''
    for epoch in range(n_epochs):
        if epoch % 100 == 0  :
            print('Epoch' , epoch , 'MSE =',mse.eval() )
        
        session.run(training_op)
    
    best_theta = theta.eval()
    print('best theta\n' , best_theta)
n_epochs = 1000 
learning_rate = 0.01

theta = tf.Variable(tf.random_uniform([n + 1 , 1] , -1.0 , 1.0) , name = 'theta')

y_pred = tf.matmul(X , theta , name = 'predictions')
error = y_pred - y 
mse = tf.reduce_mean(tf.square(error) , name = 'mse')

'''using optimizer'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
with tf.Session() as session:
    '''initializing variables'''
    session.run(init)
    
    '''converging theta'''
    for epoch in range(n_epochs):
        if epoch % 100 == 0  :
            print('Epoch' , epoch , 'MSE =',mse.eval() )
        
        session.run(training_op)
    
    best_theta = theta.eval()
    print('best theta\n' , best_theta)