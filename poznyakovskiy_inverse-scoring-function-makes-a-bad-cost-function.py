import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from tensorflow.python.framework import ops
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
del news_train_df
np.random.seed(1010)
days_past = np.sort(market_train_df.time.unique())
days_past = days_past[days_past > datetime.datetime(2009, 1, 1, tzinfo=days_past[0].tzinfo)]
days_shuffled = np.random.permutation(days_past)
days_test = np.sort(days_shuffled[0:int(len(days_past) * 0.02)])
days_train = np.sort(days_shuffled[int(len(days_past) * 0.02):])
train_df = market_train_df[market_train_df['time'].isin(days_train)].dropna(subset=['returnsOpenPrevMktres10', 'returnsOpenNextMktres10'])
test_df = market_train_df[market_train_df['time'].isin(days_test)].dropna(subset=['returnsOpenPrevMktres10', 'returnsOpenNextMktres10'])
def daily_score(test_daily):
    return np.sum(test_daily.confidenceValue * test_daily.returnsOpenNextMktres10 * test_daily.universe)
test_df = test_df.assign(confidenceValue = test_df.returnsOpenPrevMktres10)
sc = [daily_score(test_df[test_df.time == day]) for day in days_test]
print("Test score: ", np.mean(sc) / np.std(sc))
def score(Z, Y):
    x = tf.multiply(Z, Y)
    mu, var = tf.nn.moments(x, axes=1)
    return mu / tf.sqrt(var)
def cost(Z, Y):
    return -score(Z, Y)
ops.reset_default_graph()
X_train = [train_df['returnsOpenPrevMktres10']]
Y_train = [train_df['returnsOpenNextMktres10']]
X = tf.placeholder(tf.float32, [1, None], name="X")
Y = tf.placeholder(tf.float32, [1, None], name="Y")

# We initialize W with 1 and b with 0, so we are starting just with our benchmark
W = tf.get_variable('W', 1, initializer = tf.constant_initializer(1))
b = tf.get_variable('b', 1, initializer = tf.zeros_initializer())
Z = tf.add(tf.multiply(W, X), b)
cf = cost(Z, Y)

epoch_costs = []
optimizer = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(cf)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(200):
        _ , epoch_cost = sess.run([optimizer, cf], feed_dict={X: X_train, Y: Y_train})
        epoch_costs.append(epoch_cost)
        if epoch % 50 == 0:
            print('Costs after epoch %i: %f' % (epoch, epoch_cost))
    W_final = sess.run(W)
    b_final = sess.run(b)
    
plt.plot(np.squeeze(epoch_costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.show()
test_df = test_df.assign(confidenceValue = [(v * W_final + b_final)[0] for v in test_df.returnsOpenPrevMktres10])
sc = [daily_score(test_df[test_df.time == day]) for day in days_test]
print("Test score: ", np.mean(sc) / np.std(sc))
print("Weight: ", W_final)
print("Bias: ", b_final)
test_df.confidenceValue[0:10]
plt.hist(train_df.returnsOpenNextMktres10, bins=60, range=(-0.25, 0.25))
plt.show()
print("Mean returns: ", np.mean(train_df.returnsOpenNextMktres10))