!pip install ipython-autotime

%load_ext autotime
import os,re

import inspect

import functools



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.layers import Input,Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential,Model

import tensorflow_probability as tfp



import tensorflow.compat.v1 as tf1

tf1.disable_v2_behavior()
data_path = '../input/cc-live-proj/'



data = pd.read_csv(data_path + 'AI-DataTrain.csv')

data = data.drop(columns=['Unnamed: 0'])



test_data = pd.read_csv(data_path + 'AI-DataTest.csv')



print('Data shape: ',data.shape)

print('Test_data shape:', test_data.shape)
# data.loc[1000,:]=[None,1,0,0,None,0,1,1,1,1,None,1,0,0,None,0,1,1,1,1,None,1,0,0,None,0,1,1,1,1,None,1,0,0,None,0,1,1,1,1,None,1,0,0,None,0,1,1,1,1]

data.tail()
counts_data = data.apply(pd.Series.value_counts)

counts_data = counts_data.T

counts_data[0] = counts_data[0]/data.shape[0]

counts_data[1] = counts_data[1]/data.shape[0]

counts_data['Final Weights']=counts_data[0]/(counts_data[0]+counts_data[1])

counts_data.head()
train_data = data.iloc[0:900,:]

counts_train = train_data.apply(pd.Series.value_counts)

counts_train = counts_train.T

counts_train[0] = counts_train[0]/train_data.shape[0]

counts_train[1] = counts_train[1]/train_data.shape[0]

counts_train['Train Weights']=counts_train[0]/(counts_train[0]+counts_train[1])

counts_train.head()
x_train = counts_train[[0,1]].values



y_train = counts_train['Train Weights'].values

y_train = np.resize(y_train,(len(y_train),1))



x_valid = counts_data[[0,1]].values



y_valid = counts_train['Train Weights'].values

y_valid = np.resize(y_valid,(len(y_valid),1))



print('Train data shape: ',x_train.shape)

print('Train labels shape: ',y_train.shape)

print('Valid data shape: ',x_valid.shape)

print('Valid labels shape: ',y_valid.shape)
model = tf.keras.Sequential([

    tf.keras.layers.Dense(128,input_shape=(2,),activation=tf.nn.swish),

    tf.keras.layers.Dense(64,activation=tf.nn.swish),

    tf.keras.layers.Dense(1)

])

model.compile(

    optimizer='adam',

    loss = 'mse',

    metrics=[tf.keras.metrics.RootMeanSquaredError()]

)

model.summary()
EPOCHS = 50

history = model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_valid, y_valid))
epochs_range = range(EPOCHS)



plt.figure(figsize=(20, 6))



plt.subplot(121)

plt.plot(epochs_range,history.history['loss'], label='loss')

plt.plot(epochs_range,history.history['val_loss'], label='val_loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

# plt.ylim(0,1)

# plt.xlim(150,300)

plt.legend(loc='upper right')



plt.subplot(122)

plt.plot(epochs_range,history.history['root_mean_squared_error'], label='root_mean_squared_error')

plt.plot(epochs_range,history.history['val_root_mean_squared_error'], label='val_root_mean_squared_error')

plt.xlabel('Epoch')

plt.ylabel('Root mean squared error')

# plt.ylim(0,1)

# plt.xlim(150,300)

plt.legend(loc='upper right')



plt.suptitle('Train and validation loss and root mean square error',fontsize=24)
x = model.predict(x_train)

x.T
print('Easiest:',np.argmin(x.T)+1)

print('Toughest:',np.argmax(x.T)+1)
model.save('qws.h5')
m = tf.keras.models.load_model('qws.h5')

m.summary()
t = m.predict(x_valid).T

t
print('Easiest:',np.argmin(t)+1)

print('Toughest:',np.argmax(t)+1)
train_data = data.iloc[0:900,:].values.astype('float')

val_data = data.iloc[900:,:].values.astype('float')



print('Train data shape: ', train_data.shape[0])

print('Val data shape: ', val_data.shape[0])
train_data
data_shape = train_data.shape

learning_rate = 1e-3
def sigmoid(x):

    return tf1.sigmoid(x)





def log(x):

    return tf1.log(x)

    

    

def compute_cost(X,alpha,delta):

    offset = alpha-delta

    log_likelihood = tf1.reduce_sum(X * log(sigmoid(offset)) + (1-X) * log(1-sigmoid(offset)))

    return -log_likelihood
tf1.reset_default_graph()



X = tf1.placeholder(dtype='float' ,shape=data_shape, name="X")

alpha = tf1.Variable(initial_value=np.zeros((data_shape[0],1)), name="alpha", dtype='float')

delta = tf1.Variable(initial_value=np.zeros((1,data_shape[1])), name="delta", dtype='float')



out = tf1.layers.Dense(64,activation='relu')(X)

out = tf1.layers.Dense(1,activation='sigmoid')(out)



# log_likelihood = tf1.reduce_sum(X * tf1.log(tf1.sigmoid(alpha-delta)) + (1-X) * tf1.log(1-tf1.sigmoid(alpha-delta)))

cost = compute_cost(X,alpha,delta)



optimizer = tf1.train.AdamOptimizer(learning_rate)

training_op = optimizer.minimize(cost)
init = tf1.global_variables_initializer()

n_epochs = 100000
with tf1.Session() as sess:

    sess.run(init)



    for epoch in range(n_epochs):

        if epoch % 20000 == 0:

            print("Epoch: ", epoch, "\tCost =", cost.eval(feed_dict={X: train_data}))

        sess.run(training_op, feed_dict={X: train_data})

    

    best_alpha = alpha.eval()

    best_delta = delta.eval()
best_alpha.T
best_delta
print('output shape:',best_delta.shape)

print('MIN:',np.argmin(best_delta)+1,'\tMAX:',np.argmax(best_delta)+1)
# def sigmoid(x):

#     return tf1.sigmoid(x)



# def log(x):

#     return tf1.log(x)



# def compute_cost_2PL(X,discrimination,ability,difficulty):

#     eff = discrimination*(-ability+difficulty)

#     log_likelihood = tf1.reduce_sum(X * log(sigmoid(eff)) + (1-X) * log(1-sigmoid(eff)))

#     cost = -log_likelihood

#     return cost
# learning_rate = 1e-5

# data_shape = train_data.shape



# tf1.reset_default_graph()



# X = tf1.placeholder(dtype='float' ,shape=data_shape, name="X")

# ability = tf1.Variable(initial_value=np.zeros((data_shape[0],1)), name="ability", dtype='float')

# difficulty = tf1.Variable(initial_value=np.zeros((1,data_shape[1])), name="difficulty", dtype='float')

# discrimination = tf1.Variable(initial_value=np.zeros((1,data_shape[1])), name="discrimination", dtype='float')



# out = tf1.layers.Dense(64,activation='relu')(X)

# out = tf1.layers.Dense(1,activation='sigmoid')(out)



# # out1 = tf1.layers.Dense(64,activation='relu')(X)

# # out1 = tf1.layers.Dense(1,activation='sigmoid')(out1)



# # out2 = tf1.layers.Dense(64,activation='relu')(X)

# # out2 = tf1.layers.Dense(1,activation='sigmoid')(out2)



# cost = compute_cost_2PL(X,discrimination,ability,difficulty)



# optimizer = tf1.train.AdamOptimizer(learning_rate).minimize(cost)
# init = tf1.global_variables_initializer()

# n_epochs = 100000



# with tf1.Session() as sess:

#     sess.run(init)



#     for epoch in range(n_epochs):

#         if epoch % 20000 == 0:

#             print("Epoch: ", epoch, "\tCost =", cost.eval(feed_dict={X: train_data}))

#         sess.run(optimizer, feed_dict={X: train_data})

    

#     best_ability = ability.eval()

#     best_difficulty = difficulty.eval()

#     best_dicrimination = discrimination.eval()