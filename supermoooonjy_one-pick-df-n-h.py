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
#df_nh = df.loc[df["parentesco1"]!=1]
#결측값은 0으로 제거해주기 
#df_nh = df_nh.fillna(0)
df_nh = df.fillna(0)
df_ny =pd.DataFrame(df_nh["Target"])
df_ny.head()
#불필요한 피쳐들 제거해 x1값 만들기 
# age, SQBescolari, SQBage, SQBhogar_total, SQBedjefe, SQBhogar_nin, SQBovercrowding,
# SQBdependency, SQBmeaned, agesq

#결과값 y에 해당하는 값 삭제 
df_x2 = df_nh.drop(["Target"],1)
#세대주성별 교육년수 (논의필요)
df_x2 = df_x2.drop(["edjefa", "edjefe"],1) 
#중복정보 제거
df_x2 = df_x2.drop(["dependency","female","area2","hacdor","hacapo","bedrooms","r4h3","r4m3"],1) 
#수학적으로 의미가 없는 값 제거
df_x2 = df_x2.drop(["Id","SQBescolari", "SQBage", "SQBhogar_total", "SQBedjefe", "SQBhogar_nin", "SQBovercrowding","idhogar"],1) 
#세대주와의 관계열 제거
df_x2 = df_x2.drop(["parentesco1","parentesco2","parentesco3","parentesco4","parentesco5","parentesco6","parentesco7","parentesco8",
                   "parentesco9","parentesco10","parentesco11","parentesco12"],1)
#집세 임시제거!!!!!!!!!!!!!!!!!!!!!!!!!!
df_x2 = df_x2.drop(["v2a1"],1)
df_x2.head()
#새로운열 추가 
# df_x1['rent_to_rooms'] = df_x1['v2a1']/df_x1['rooms']
# df_x1['r4t3_to_rooms'] = df_x1['r4t3']/df_x1['rooms']
# df_x1['rent_to_r4t3'] = df_x1['v2a1']/df_x1['r4t3']
# df_x1['v2a1_to_r4t3'] = df_x1['v2a1']/(df_x1['r4t3'] - df_x1['r4t1'])
df_x2['lent'] = df_x2['tamviv']-df_x2['tamhog']

#열 생성 이후 불필요한 열 제거 
df_x2 = df_x2.drop(["r4t1","r4t2","r4t3","tamhog","tamviv"],1) 
df_x2.head()
from tqdm import tqdm_notebook
import tensorflow as tf
import numpy as np
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
df_ny["Target"]=df_ny["Target"]-1
# Review : Learning rate and Evaluation
import tensorflow as tf
import random
import matplotlib.pyplot as plt

x_n_data = x_n_data = df_x2
sess = tf.Session()
y_n_data = tf.one_hot(df_ny, depth = 4).eval(session=sess)
y_n_data = tf.reshape(y_n_data, shape=[-1,4]).eval(session=sess)
print(y_n_data)
tf.set_random_seed(999)  # reproducibility


# parameters
learning_rate = 0.001



X = tf.placeholder(tf.float32, [None, 107])
Y = tf.placeholder(tf.float32, [None, 4])

W1 = tf.get_variable("W1", shape=[107, 64],
                     initializer=xavier_init(107, 64))
b1 = tf.Variable(tf.random_normal([64]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[64, 64],
                     initializer=xavier_init(64, 64))
b2 = tf.Variable(tf.random_normal([64]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[64, 4],
                     initializer=xavier_init(64, 4))
b3 = tf.Variable(tf.random_normal([4]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(optimizer, feed_dict={X: x_n_data, Y: y_n_data})
    if step % 1000 == 0 or step < 100:
        loss, acc = sess.run([cost, accuracy], feed_dict={
                             X: x_n_data, Y: y_n_data})
        print("Step: {:5}, \t Loss: {:.3f}, \t Acc: {:.2%}".format(
            step, loss, acc))
df2 =pd.read_csv('../input/test.csv')
df2_h = df2.fillna(0)
#결과값 y에 해당하는 값 삭제 
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
##새로운변수 추가 
df2_x1 =  df2_x1.values.tolist()
# Testing & One-hot encoding
test_data = df2_x1


pred_val = sess.run(hypothesis, feed_dict={X: test_data})
pred_idx = sess.run(tf.argmax(pred_val, 1))

# print("predict value : \n {} \n\npredict index : {}".format(pred_val, pred_idx))
print("test data : {} \n\npredict value : \n {} \n\npredict index : {}".format(test_data, pred_val, pred_idx))
submission = pd.DataFrame({'Id' : df2.Id, 'Target' : pred_idx+1})
submission.head()
submission.to_csv("submissions.csv", index =False)