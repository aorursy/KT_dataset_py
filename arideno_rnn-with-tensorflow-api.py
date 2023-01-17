import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline
class TimeSeriesData():

    def __init__(self,num_points,xmin,xmax):

        self.xmin = xmin

        self.xmax = xmax

        self.num_point = num_points

        self.resolution = (xmax-xmin)/(num_points-1)

        self.x_data = np.linspace(xmin,xmax,num_points)

        self.y_true = np.sin(self.x_data)

    

    def ret_true(self,x_series):

        return np.sin(x_series)

    

    def next_batch(self,batch_size,steps,return_batch_ts=False):

        

        # Grab a random starting point for each batch

        rand_start = np.random.rand(batch_size,1)

        

        # Convert to be on time series

        ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))

        

        # Create batch time series on the x axis

        batch_ts = ts_start + np.arange(0.0,steps+1) * self.resolution

        

        # Create the Y data for the time series x axis from previous step

        y_batch = np.sin(batch_ts)

        

        # FORMATTION for RNN

        if return_batch_ts:

            return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1), batch_ts

        else:

            return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1)
ts_data = TimeSeriesData(250,0,10)
plt.plot(ts_data.x_data,ts_data.y_true)
num_time_steps = 30
y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)
plt.plot(ts.flatten()[1:],y2.flatten(),'*')
plt.plot(ts_data.x_data,ts_data.y_true,label='Sin(t)')

plt.plot(ts.flatten()[1:],y2.flatten(),'*',label='Single Training Instance')

plt.legend()

plt.tight_layout()
# TRAINING DATA
train_inst = np.linspace(5, 5 + ts_data.resolution*(num_time_steps+1),num_time_steps+1)
train_inst
plt.title('A TRAINING INSTANCE')



plt.plot(train_inst[:-1],ts_data.ret_true(train_inst[:-1]),'bo',markersize=15,alpha=0.5,label='INSTANCE')

plt.plot(train_inst[1:],ts_data.ret_true(train_inst[1:]),'ko',markersize=7,label='TARGET')

plt.legend()
num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.001
num_train_iterations = 2000
batch_size = 1
# PLACEHOLDERS
x = tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])
# RNN CELL LAYER
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(num_units=num_neurons,activation=tf.nn.relu),output_size=num_outputs)
outputs, states = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
# MSE

loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
# SESSION
saver = tf.train.Saver()
with tf.Session() as sess:

    sess.run(init)

    

    for iteration in range(num_train_iterations):

        X_batch, y_batch = ts_data.next_batch(batch_size,num_time_steps)

        

        sess.run(train,feed_dict={x:X_batch,y:y_batch})

        

        if iteration % 100 == 0:

            mse = loss.eval(feed_dict={x:X_batch,y:y_batch})

            print(iteration,'\tMSE',mse)

            

    saver.save(sess,'./rnn_time_series_model_codealong')
with tf.Session() as sess:

    saver.restore(sess,'./rnn_time_series_model_codealong')

    

    X_new = np.sin(np.array(train_inst[:-1].reshape(-1,num_time_steps,num_inputs)))

    y_pred = sess.run(outputs,feed_dict={x:X_new})
plt.title('TESTING THE MODEL')



# TRAINING INSTANCE

plt.plot(train_inst[:-1],np.sin(train_inst[:-1]),'bo',markersize=15,alpha=0.5,label='TRAINIGN INST')



# TARGET TO PREDICT

plt.plot(train_inst[1:],np.sin(train_inst[1:]),'ko',markersize=10,label='target')



# MODELS PREDICTION

plt.plot(train_inst[1:],y_pred[0,:,0],'r.',markersize=10,label='PREDICTIONS')



plt.xlabel('TIME')

plt.legend()

plt.tight_layout()
with tf.Session() as sess:

    saver.restore(sess,'./rnn_time_series_model_codealong')

    

    zero_seq_seed = [0.0 for i in range(num_time_steps)]

    

    for iteration in range(len(ts_data.x_data)-num_time_steps):

        X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1,num_time_steps,1)

        

        y_pred = sess.run(outputs,feed_dict={x:X_batch})

        

        zero_seq_seed.append(y_pred[0,-1,0])
plt.plot(ts_data.x_data,zero_seq_seed,'b-')

plt.plot(ts_data.x_data[:num_time_steps],zero_seq_seed[:num_time_steps],'r',linewidth=3)

plt.xlabel('TIME')

plt.ylabel('Y')
with tf.Session() as sess:

    saver.restore(sess,'./rnn_time_series_model_codealong')

    

    training_instance = list(ts_data.y_true[:30])

    

    for iteration in range(len(ts_data.x_data)-num_time_steps):

        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1,num_time_steps,1)

        

        y_pred = sess.run(outputs,feed_dict={x:X_batch})

        

        training_instance.append(y_pred[0,-1,0])
plt.plot(ts_data.x_data,training_instance,'b-')

plt.plot(ts_data.x_data[:num_time_steps],training_instance[:num_time_steps],'r',linewidth=3)

plt.xlabel('TIME')

plt.ylabel('Y')