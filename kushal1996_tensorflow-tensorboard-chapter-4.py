import numpy as np 
import tensorflow as tf 

'''creating a placeholder'''
x = tf.placeholder(tf.float32 ,shape = (None , 1024) ,  name = 'x') # None means any number of row's
y = tf.matmul(x , x , name = 'y')

with tf.Session() as session:
    rand_array = np.random.rand(1024 , 1024)
    print(session.run(y , feed_dict = {x : rand_array}))
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from datetime import datetime

tf.reset_default_graph()

housing = load_boston()
data = housing.data

m , n = data.shape

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

data_plus_bias = np.c_[np.ones((m , 1 )) , data ]


X = tf.placeholder( tf.float32  , name = 'X')
y = tf.placeholder( tf.float32  , name = 'y')

n_epoch = 1000
learning_rate = 0.01

theta = tf.Variable(tf.random_uniform([n + 1 , 1] , -1.0 , 1.0) , name = 'theta')
y_pred = tf.matmul(X , theta , name = 'prediction')
error = y_pred - y

mse = tf.reduce_mean(tf.square(error) , name = 'mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)

'''We are using timestamp in the logdir is because Tensorflow will merge stats from different run and it will mess 
   the visualization'''
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs_placeholder'
logdir = '{}/run-{}/'.format(root_logdir ,now)

'''The following line creates a node in the graph that will evaluate the MSE value and write it to a TensorBoard-compatible 
   binary log string called a summary'''
mse_summary = tf.summary.scalar('MSE' , mse)

'''This line creates a FileWriter that we will use to write the summaries to logfiles in the log directory.
   The First parameter indicates the path of the log directory (path/tf_logs/run-20190102091959) , the second
   parameter is optional , it is the graph we want visualize'''
file_writer = tf.summary.FileWriter(logdir , tf.get_default_graph())

init = tf.global_variables_initializer()

with tf.Session() as session :
    session.run(init)
    
    for epoch in range(n_epoch):
        
        if epoch % 100 == 0:
            
            '''EVALUATING mse_summary in execution phase'''
            summary_str = mse_summary.eval(feed_dict = {X : data_plus_bias , y : housing.target.reshape(-1  , 1)} )
            '''outputing the summary'''
            file_writer.add_summary(summary_str , epoch )
            
            print('Epoch =' , epoch , 'MSE = ' , mse.eval(feed_dict = {X : data_plus_bias , y : housing.target.reshape(-1  , 1)}))
        
        session.run(training_op , feed_dict = {X : data_plus_bias , y : housing.target.reshape(-1  , 1)})
    best_theta = theta.eval()
    
print(best_theta)

'''terminating file_writer'''
file_writer.close()
tf.reset_default_graph()

X = tf.placeholder( tf.float32  , name = 'X')
y = tf.placeholder( tf.float32  , name = 'y')

n_epoch = 1000
learning_rate = 0.01

theta = tf.Variable(tf.random_uniform([n + 1 , 1] , -1.0 , 1.0) , name = 'theta')
y_pred = tf.matmul(X , theta , name = 'prediction')

'''defining error and mse within name scope loss'''
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error) , name = 'mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs_loss'
logdir = '{}/run-{}/'.format(root_logdir ,now)

mse_summary = tf.summary.scalar('MSE' , mse)

file_writer = tf.summary.FileWriter(logdir , tf.get_default_graph())

init = tf.global_variables_initializer()

with tf.Session() as session :
    session.run(init)
    
    for epoch in range(n_epoch):
        
        if epoch % 100 == 0:
            
            summary_str = mse_summary.eval(feed_dict = {X : data_plus_bias , y : housing.target.reshape(-1  , 1)} )
            file_writer.add_summary(summary_str , epoch )
            
        session.run(training_op , feed_dict = {X : data_plus_bias , y : housing.target.reshape(-1  , 1)})
    best_theta = theta.eval()
    
file_writer.close()
tf.reset_default_graph()

def relu(x):
    w_shape = (int(x.get_shape()[1]) , 1)
    w = tf.Variable(tf.random_normal(w_shape) , name = 'weights')
    b = tf.Variable(0.0 , name = 'bias')
    z = tf.add(tf.matmul(x , w) , b , name = 'z')
    return tf.maximum(z , 0 , name =  'relu')

n_features = 3 
x = tf.placeholder(tf.float32 , shape = (None , n_features) , name = 'x')
relus = [relu(x) for i in range(5)]
output = tf.add_n(relus , name = 'ouput')

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs_relu'
logdir = '{}/run-{}/'.format(root_logdir ,now)

file_writer = tf.summary.FileWriter(logdir , tf.get_default_graph())
tf.reset_default_graph()

def relu(x):
    with tf.name_scope("relu") as scope:
        w_shape = (int(x.get_shape()[1]) , 1)
        w = tf.Variable(tf.random_normal(w_shape) , name = 'weights')
        b = tf.Variable(0.0 , name = 'bias')
        z = tf.add(tf.matmul(x , w) , b , name = 'z')
        return tf.maximum(z , 0 , name =  'relu')

n_features = 3 
x = tf.placeholder(tf.float32 , shape = (None , n_features) , name = 'x')
relus = [relu(x) for i in range(5)]
output = tf.add_n(relus , name = 'ouput')

now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs_relu'
logdir = '{}/run-{}/'.format(root_logdir ,now)

file_writer = tf.summary.FileWriter(logdir , tf.get_default_graph())