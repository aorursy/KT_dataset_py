# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
df =pd.read_csv('../input/train.csv')
df.head()
df_h = df.loc[df["parentesco1"]==1]
df_h = df_h.fillna(0)

df_y =pd.DataFrame(df_h["Target"])
df_y.head()

df_x1 = df_h.drop(["Target"],1)
#세대주성별 교육년수 (논의필요)
df_x1 = df_x1.drop(["edjefa", "edjefe"],1) 
#중복정보 제거
df_x1 = df_x1.drop(["dependency","female","area2","hacdor","hacapo","bedrooms","r4h3","r4m3"],1) 
#수학적으로 의미가 없는 값 제거
df_x1 = df_x1.drop(["Id","SQBescolari", "SQBage", "SQBhogar_total", "SQBedjefe", "SQBhogar_nin", "SQBovercrowding","idhogar"],1) 
#세대주와의 관계열 제거
df_x1 = df_x1.drop(["parentesco1","parentesco2","parentesco3","parentesco4","parentesco5","parentesco6","parentesco7","parentesco8",
                   "parentesco9","parentesco10","parentesco11","parentesco12"],1)
df_x1 = df_x1.drop(["etecho1"],["etecho2"],["etecho3"],["eviv1"],["eviv2"],["eviv3"],1)
#집세 임시제거!!!!!!!!!!!!!!!!!!!!!!!!!!
df_x1 = df_x1.drop(["v2a1"],1)
df_x1.head()
df_x1['lent'] = df_x1['tamviv']-df_x1['tamhog']

#열 생성 이후 불필요한 열 제거 
df_x1 = df_x1.drop(["r4t1","r4t2","r4t3","tamhog","tamviv"],1) 
df_x1.head()
import tensorflow as tf
def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        # 6 was used in the paper.
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limints as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)
df_y["Target"]=df_y["Target"]-1
import tensorflow as tf
import random
import matplotlib.pyplot as plt

x_data = x_data = df_x1
sess = tf.Session()
y_data = tf.one_hot(df_y, depth = 4).eval(session=sess)
y_data = tf.reshape(y_data, shape=[-1,4]).eval(session=sess)
print(y_data)
tf.set_random_seed(999)  # reproducibility


# parameters
learning_rate = 0.001



X = tf.placeholder(tf.float32, [None, 101])
Y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)



W1 = tf.get_variable("W1", shape=[101, 64], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([64]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


W2 = tf.get_variable("W2", shape=[64, 64], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([64]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[64, 64], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([64]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)


W4 = tf.get_variable("W4", shape=[64, 64], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([64]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)


W5 = tf.get_variable("W5", shape=[64, 4], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([4]))
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)

hypothesis = tf.matmul(L4, W5) + b5



hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(optimizer, feed_dict={X: x_data, Y: y_data,keep_prob:0.7})
    if step % 1000 == 0 or step < 100:
        loss, acc = sess.run([cost, accuracy], feed_dict={
                             X: x_data, Y: y_data,keep_prob:0.7})
        print("Step: {:5}, \t Loss: {:.3f}, \t Acc: {:.2%}".format(
            step, loss, acc))
df2 =pd.read_csv('../input/test.csv')
df2_h = df2.fillna(0)
df2_x1 = df2_h
#세대주성별 교육년수 (논의필요)
df2_x1 = df2_x1.drop(["edjefa", "edjefe"],1) 
#중복정보 제거
df2_x1 = df2_x1.drop(["dependency","female","area2","hacdor","hacapo","bedrooms","r4h3","r4m3"],1) 
#수학적으로 의미가 없는 값 제거
df2_x1 = df2_x1.drop(["Id","SQBescolari", "SQBage", "SQBhogar_total", "SQBedjefe", "SQBhogar_nin", "SQBovercrowding","idhogar"],1) 
#세대주와의 관계열 제거
df2_x1 = df2_x1.drop(["parentesco1","parentesco2","parentesco3","parentesco4","parentesco5","parentesco6","parentesco7","parentesco8",
                   "parentesco9","parentesco10","parentesco11","parentesco12"],1)
df2_x1['lent'] = df2_x1['tamviv']-df2_x1['tamhog']
#집세 임시제거!!!!!!!!!!!!!!!!!!!!!!!!!!
df2_x1 = df2_x1.drop(["v2a1"],1)
df2_x1 = df2_x1.drop(["r4t1","r4t2","r4t3","tamhog","tamviv"],1) 
df2_x1 = df_x1.drop(["etecho1"],["etecho2"],["etecho3"],["eviv1"],["eviv2"],["eviv3"],1)
df2_x1 =  df2_x1.values.tolist()
test_data = df2_x1


pred_val = sess.run(hypothesis, feed_dict={X: test_data,keep_prob:1})
pred_idx = sess.run(tf.argmax(pred_val, 1))

pred_idx = pred_idx +1
submission = pd.DataFrame({'Id' : df2.Id, 'Target' : pred_idx})
submission.head()
submission.to_csv("submissions.csv", index =False)
