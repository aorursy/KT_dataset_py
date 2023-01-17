# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Importing data
insurance = pd.read_csv("../input/insurance.csv")

#Data preparation
print("sex unique values:",insurance.sex.unique())
insurance.loc[insurance['sex'] == 'male','sex'] = 0
insurance.loc[insurance['sex'] == 'female','sex'] = 1
print("smoker unique values:",insurance.smoker.unique())
insurance.loc[insurance['smoker'] == 'yes','smoker'] = 0
insurance.loc[insurance['smoker'] == 'no','smoker'] = 1
print("region unique values:",insurance.region.unique())
region_cat = pd.get_dummies(insurance['region'], prefix='region')
df = pd.concat([insurance, region_cat],axis=1)
df = df.drop('region', axis=1)

features_cols = ['age','sex','bmi', 'children','smoker','region_northeast','region_northwest','region_southeast','region_southwest']
predict_cols = ['charges']

dfx = df[features_cols]
dfy = df[predict_cols]

train_dfx = dfx.sample(frac=0.8,random_state=200)
train_dfy = dfy.sample(frac=0.8,random_state=200)
test_dfx = dfx.drop(train_dfx.index)
test_dfy = dfy.drop(train_dfy.index)

#Normalization
train_x_mean = train_dfx.mean()
train_x_std = train_dfx.std()
test_x_mean = train_dfy.mean()
test_x_std = train_dfy.std()

train_dfx = (train_dfx - train_x_mean) / train_x_std
test_dfx = (test_dfx - train_x_mean) / train_x_std
train_dfy = (train_dfy - test_x_mean) / test_x_std
test_dfy = (test_dfy - test_x_mean) / test_x_std

print("train data contains", train_dfx.shape[0], "rows")
print("test data contains", test_dfx.shape[0], "rows")
learning_rate = 1e-5
training_epochs = 10000
batch_size = 500 # Mini-Batch Gradient Descent 
display = 1000
seed = 200
(hidden1_size, hidden2_size) = (100, 50)

X = tf.placeholder(tf.float32, shape=[None,train_dfx.shape[1]])
Y = tf.placeholder(tf.float32, shape=[None, train_dfy.shape[1]])
training_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(500).repeat().batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().batch(test_dfx.shape[0])
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,training_dataset.output_shapes)
dx, dy = iterator.get_next()
training_init_op = iterator.make_initializer(training_dataset)
test_init_op = iterator.make_initializer(test_dataset)
with tf.Session() as sess:
    W1 = tf.Variable(tf.random_normal([train_dfx.shape[1], hidden1_size], seed = seed))
    b1 = tf.Variable(tf.random_normal([hidden1_size], seed = seed))
    z1 = tf.nn.relu(tf.add(tf.matmul(dx,W1), b1))
    W2 = tf.Variable(tf.random_normal([hidden1_size, hidden2_size], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([hidden2_size], seed = seed))
    z2 = tf.nn.relu(tf.add(tf.matmul(z1,W2), b2))
    W3 = tf.Variable(tf.random_normal([hidden2_size, train_dfy.shape[1]], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([train_dfy.shape[1]], seed = seed))
    h = tf.add(tf.matmul(z2,W3), b3)                                    
    loss = tf.reduce_mean(tf.pow(h - dy, 2)) + 0.1 * tf.nn.l2_loss(W3)
    update = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(training_init_op, feed_dict={X: train_dfx.values, Y: train_dfy.values})
    for epoch in range(training_epochs):
        sess.run(update)
        if (epoch + 1) % display == 0:
            print("iter: {}, loss: {:.3f}".format(epoch + 1, sess.run(loss)))            
print("Training Finished!")
print("Train MSE:", sess.run(loss))
sess.run(test_init_op, feed_dict={X: test_dfx.values, Y: test_dfy.values})
print("Test MSE:", sess.run(loss))