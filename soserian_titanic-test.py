# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Initialize Figure and Axes object




train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

def getList(content, key):
    return content[key].values.tolist()

def get_vars():
    age_data = []
    sex_data = []
    survived_data = []
    pclass_data = []
    fare_data = []
    cabin_data = []
    emb_data = []
    sb_data = []
    par_data = []
    
    for i in range(len(tm_age_data)):
        if math.isnan(tm_age_data[i]) == False:
            age_data.append(tm_age_data[i])
        else:
            age_data.append(32)

        pdata = tm_pclass_data[i]
        pclass_data.append(4 - pdata)
        
        sex_data.append(2 if tm_sex_data[i] == 'female' else 1)
        survived_data.append([tm_survived_data[i]])
        emb_data.append(3 if tm_emb_data[i] == 'S' else 2 if tm_emb_data[i] == 'C' else 1)
        cabin_data.append(1 if type(tm_cabin_data[i]) != str else math.ceil(ord(tm_cabin_data[i][:1])/100+1))
#         if tm_fare_data[i] < 10:
#             fare_data.append(1)
#         elif tm_fare_data[i] > 70:
#             fare_data.append(3)
#         else:
#             fare_data.append(2)
        fare_data.append(tm_fare_data[i])
        sb_data.append(tm_sb_data[i])
        par_data.append(tm_par_data[i])

#     sex_data = [0 if i == 'female' else 1 for i in sex_data]
#     sex_data = np.add(sex_data, 1.0)
    par_data = np.add(par_data, 1.0)
    age_data = [9 if val < 16 else 2 if val < 30 else 3 if val < 40 else 4 for val in age_data]
    
    x_data = np.vstack((sex_data, pclass_data, age_data, fare_data, emb_data)).T
    return (age_data, sex_data, pclass_data, fare_data, cabin_data, emb_data, survived_data, x_data)

tm_age_data = getList(train, "Age")
tm_sex_data = getList(train, "Sex")
tm_pclass_data = getList(train, "Pclass")
tm_fare_data = getList(train, "Fare")
tm_survived_data = getList(train, "Survived")
tm_cabin_data = getList(train, "Cabin")
tm_emb_data = getList(train, "Embarked")
tm_sb_data = getList(train, "SibSp")
tm_par_data = getList(train, "Parch")


(age_data, sex_data, pclass_data, fare_data, cabin_data, emb_data, survived_data, x_data) = get_vars()
print(x_data)
x_len = len(x_data[0])
# age_data = np.array(age_data).reshape([-1,1]).tolist()
# print(age_data)
X = tf.placeholder(tf.float32, [None, x_len])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([x_len, 20], stddev=0.01))
b1 = tf.Variable(tf.zeros([20]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

W2 = tf.Variable(tf.random_normal([20, 20], stddev=0.01))
b2 = tf.Variable(tf.zeros([20]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

W3 = tf.Variable(tf.random_normal([20, 20], stddev=0.01))
b3 = tf.Variable(tf.zeros([20]))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))

W4 = tf.Variable(tf.random_normal([20, 10], stddev=0.01))
b4 = tf.Variable(tf.zeros([10]))
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))

W5 = tf.Variable(tf.random_normal([10, 1], stddev=0.01))
b5 = tf.Variable(tf.zeros([1]))
model = tf.add(tf.matmul(L4, W5), b5)

cost = tf.reduce_mean(tf.squared_difference(model, Y))
# cost = -tf.reduce_mean(Y * tf.log(model) + (1 - Y) * tf.log(1 - model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

fig, ax = plt.subplots()

ax.plot(range(len(x_data)), x_data)

plt.show()
for epoch in range(1000):
    total_cost = 0

    _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: survived_data})

    if epoch % 100 == 0:
        print(cost_val)

tm_age_data = getList(test, "Age")
tm_sex_data = getList(test, "Sex")
tm_pclass_data = getList(test, "Pclass")
tm_fare_data = getList(test, "Fare")
tm_cabin_data = getList(test, "Cabin")
tm_emb_data = getList(test, "Embarked")
tm_sb_data = getList(test, "SibSp")
tm_par_data = getList(test, "Parch")
# age_data = []
# sex_data = []
# survived_data = []

# for i in range(len(tm_age_data)):
#     if math.isnan(tm_age_data[i]) == False:
#         age_data.append(tm_age_data[i])
#     else:
#         age_data.append(32)
#     sex_data.append(tm_sex_data[i])

# sex_data = [1 if i == 'female' else 2 for i in sex_data]
# age_data = np.divide(age_data, 5.0)
# test_data = np.vstack((sex_data, age_data)).T
(age_data, sex_data, pclass_data, fare_data, cabin_data, emb_data, suvived_data, test_data) = get_vars()
result = sess.run(tf.cast(model > 0.5, dtype=tf.float32), feed_dict = {X: test_data})

sv = np.array(result).T[0].astype(int)

print(sv)
df_submission = pd.DataFrame({
   "PassengerId": test["PassengerId"],
   "Survived": sv
})
df_submission.to_csv('./result1.csv', index=False)





# Y = tf.placeholder(tf.float32, [None, 10])
# target = tf.argmax(Y, 1)
# print(sess.run(target, feed_dict={Y: survived_data}))
# input [1,2,3,4,5]
# ouput 0 

# Y = tf.placeholder(tf.float32, [None, 1])
# target = tf.argmax(Y, 1)
# print(sess.run(target, feed_dict={Y: survived_data}))
# [[1],[2],[111]]


# prediction = tf.argmax(model, 1)
# is_correct = tf.equal(tf.argmax(model, 1), target)
# print(sess.run(X))
# print(survived_data)
# print(sess.run(prediction,feed_dict={X: age_data}))
# print(sess.run(model, feed_dict={X: x_data}))
# print(sess.run(target, feed_dict={Y: survived_data}))

# fig, ax = plt.subplots()
# # Create violinplot
# ax.plot(range(len(x_data)), x_data)

# # Show the plot
# plt.show()
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# print(sess.run(accuracy, feed_dict={X: x_data, Y: survived_data}))

# predicted = tf.cast(model > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# sess.run([predicted, accuracy], feed_dict={X: x_data, Y: survived_data})