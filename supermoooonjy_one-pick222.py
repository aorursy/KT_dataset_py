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
import tensorflow as tf
import numpy as np
train =pd.read_csv('../input/train.csv')
test =pd.read_csv('../input/test.csv')
train_h = train.loc[train["parentesco1"]==1]
def drop_feature(df_x1):
    #결과값 y에 해당하는 값 삭제 
    # df_x1 = df_x1.drop(["Target"],1)
    df_x1 = df_x1.fillna(0)
    #세대주성별 교육년수 
    df_x1['dependency'] = np.sqrt(df_x1['SQBdependency'])
    df_x1.loc[df_x1['edjefa'] == "no", "edjefa"] = 0
    df_x1.loc[df_x1['edjefe'] == "no", "edjefe"] = 0
    df_x1.loc[(df_x1['edjefa'] == "yes") & (df_x1['parentesco1'] == 1), "edjefa"] = df_x1.loc[(df_x1['edjefa'] == "yes") & (df_x1['parentesco1'] == 1), "escolari"]
    df_x1.loc[(df_x1['edjefe'] == "yes") & (df_x1['parentesco1'] == 1), "edjefe"] = df_x1.loc[(df_x1['edjefe'] == "yes") & (df_x1['parentesco1'] == 1), "escolari"]
    df_x1.loc[df_x1['edjefa'] == "yes", "edjefa"] = 4
    df_x1.loc[df_x1['edjefe'] == "yes", "edjefe"] = 4
    df_x1['edjefe'] = df_x1['edjefe'].astype("int")
    df_x1['edjefa'] = df_x1['edjefa'].astype("int")
    df_x1['dependency'] = df_x1['dependency'].astype("int")
    df_x1['edjef'] = np.max(df_x1[['edjefa','edjefe']], axis=1)
    df_x1['v2a1']=df_x1['v2a1'].fillna(0)
    df_x1['v18q1']=df_x1['v18q1'].fillna(0)
    df_x1['rez_esc']=df_x1['rez_esc'].fillna(0)
    df_x1.loc[df_x1.meaneduc.isnull(), "meaneduc"] = 0
    df_x1.loc[df_x1.SQBmeaned.isnull(), "SQBmeaned"] = 0
    #중복정보 제거
    df_x1 = df_x1.drop(["dependency","female","area2","hacdor","hacapo","bedrooms","r4h3","r4m3"],1) 
    #수학적으로 의미가 없는 값 제거
    df_x1 = df_x1.drop(["Id","SQBescolari", "SQBage", "SQBhogar_total", "SQBedjefe", "SQBhogar_nin", "SQBovercrowding","idhogar"],1) 
    #세대주와의 관계열 제거
    df_x1 = df_x1.drop(["parentesco1","parentesco2","parentesco3","parentesco4","parentesco5","parentesco6","parentesco7","parentesco8",
                       "parentesco9","parentesco10","parentesco11","parentesco12"],1)
    #집을 소유하고 있는 사람들에게 1값부여
    #df_x1["house"] = df_x1.loc[df_x1["v2a1"] == 0, "v2a1"]
    #df_x1["house"] = df_x1["house"].fillna(1)
    # 집세 임시 제거
    #df_x1['lent'] = (df_x1["v2a1"])
    df_x1 = df_x1.drop(["v2a1"],1)
    df_x1['lent'] = df_x1['tamviv']-df_x1['tamhog']
    df_x1 = df_x1.drop(["r4t1","r4t2","r4t3","tamhog","tamviv"],1) 
    # 중복 데이터 : 태블릿 수 삭제
    df_x1 = df_x1.drop(["v18q1"],1) 
    #벽,지붕, 바닥 좋고 나쁨 정도 중복 
    df_x1 = df_x1.drop(["epared1","epared2","epared3","etecho1","etecho2","etecho3"],1) 
    #모든 열이 0인 경우 삭제 
    df_x1 = df_x1.drop(["elimbasu5", "estadocivil1"],1) 
    #상관관계가 높은 사항들 제거
    df_x1 = df_x1.drop(["pisocemento","overcrowding","hhsize"],1)
    # 개인별 삭제 (실험 해 보아라~~~~): 정배픽
    #df_x1 = df_x1.drop(["age","lugar1","lugar6","sanitario2","coopele"],1)
    #정연픽
    # df_x1 = df_x1.drop(["hogar_nin","hogar_adul","hogar_mayor","hogar_total","paredblolad","meaneduc","qmobilephone"],1)
    df_x1 = df_x1.drop(["SQBdependency","SQBmeaned","agesq"],1)
    #진수픽
    
    return df_x1
train_h_drop = drop_feature(train_h)
#rain_h_drop.head()
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
df_x1 = train_h_drop.drop(["Target"],1)
df_y = train_h['Target']-1
tf.reset_default_graph()
# Review : Learning rate and Evaluation
import tensorflow as tf
import random
import matplotlib.pyplot as plt

placeholder_num = len(df_x1.columns)


x_data = df_x1
sess = tf.Session()
y_data = tf.one_hot(df_y, depth = 4).eval(session=sess)
y_data = tf.reshape(y_data, shape=[-1,4]).eval(session=sess)
print(y_data)
tf.set_random_seed(999)  # reproducibility



# parameters
learning_rate = 0.001
i =64


X = tf.placeholder(tf.float32, [None, placeholder_num])
Y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)


W1 = tf.get_variable("W1", shape=[placeholder_num, i], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([i]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


W2 = tf.get_variable("W2", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([i]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([i]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)


W4 = tf.get_variable("W4", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([i]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([i]))
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

W6 = tf.get_variable("W6", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([i]))
L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
L6 = tf.nn.dropout(L6, keep_prob=keep_prob)

W7 = tf.get_variable("W7", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([i]))
L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
L7 = tf.nn.dropout(L7, keep_prob=keep_prob)

W8 = tf.get_variable("W8", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([i]))
L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
L8 = tf.nn.dropout(L8, keep_prob=keep_prob)

W9 = tf.get_variable("W9", shape=[i, 4], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([4]))
L9 = tf.nn.relu(tf.matmul(L8, W9) + b9)

# W10 = tf.get_variable("W10", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
# b10 = tf.Variable(tf.random_normal([i]))
# L10 = tf.nn.relu(tf.matmul(L9, W10) + b10)
# L10 = tf.nn.dropout(L10, keep_prob=keep_prob)


# W11 = tf.get_variable("W11", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
# b11 = tf.Variable(tf.random_normal([i]))
# L11 = tf.nn.relu(tf.matmul(L10, W11) + b11)
# L11 = tf.nn.dropout(L11, keep_prob=keep_prob)

# W12 = tf.get_variable("W12", shape=[i, i], initializer=tf.contrib.layers.xavier_initializer())
# b12 = tf.Variable(tf.random_normal([i]))
# L12 = tf.nn.relu(tf.matmul(L11, W12) + b12)
# L12 = tf.nn.dropout(L12, keep_prob=keep_prob)

# W13 = tf.get_variable("W13", shape=[i, 4], initializer=tf.contrib.layers.xavier_initializer())
# b13 = tf.Variable(tf.random_normal([4]))
# L13 = tf.nn.relu(tf.matmul(L12, W13) + b13)

# W11 = tf.get_variable("W11", shape=[i, 4], initializer=tf.contrib.layers.xavier_initializer())
# b11 = tf.Variable(tf.random_normal([4]))
# L11 = tf.nn.relu(tf.matmul(L10, W11) + b11)

hypothesis = tf.matmul(L8, W9) + b9


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(5001):
    sess.run(optimizer, feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
    if step % 1000 == 0 or step < 100:
        loss, acc = sess.run([cost, accuracy], feed_dict={
                             X: x_data, Y: y_data, keep_prob: 0.7})
        print("Step: {:5}, \t Loss: {:.3f}, \t Acc: {:.2%}".format(
            step, loss, acc))
df2=drop_feature(test)
df2 =  df2.values.tolist()
test_data = df2


pred_val = sess.run(hypothesis, feed_dict={X: test_data, keep_prob: 1.0})
pred_idx = sess.run(tf.argmax(pred_val, 1))
pred_idx = pred_idx +1
submission = pd.DataFrame({'Id' : test.Id, 'Target' : pred_idx})
submission.head()
submission.to_csv("submissions.csv", index =False)
