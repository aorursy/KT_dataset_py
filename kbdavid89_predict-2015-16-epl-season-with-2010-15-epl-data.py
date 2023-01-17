# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv

model = tf.compat.v1.global_variables_initializer()
data = read_csv('/kaggle/input/epl-data/totalengland.csv', sep=',')
xy = np.array(data, dtype=np.float32)
print(xy)
x_data = xy[:, 0:-1]
y_data = xy[:,[-1]]
print(x_data)
tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder(tf.float32, shape=[None,14])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random.normal([14,1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")
hypothesis = tf.matmul(X,W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000005)
train = optimizer.minimize(cost)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for step in range(100001):
    cost_, hypo_, _=sess.run([cost,hypothesis,train], feed_dict={X:x_data, Y:y_data})
    if step % 500 == 0 :
        print("#", step, "cost: ", cost_)
        print("-total point: ", hypo_[0])
saver = tf.compat.v1.train.Saver()
save_path = saver.save(sess, "./saved.cpkt")
print("model saved")
saver = tf.compat.v1.train.Saver()
model = tf.compat.v1.global_variables_initializer()
speed = float(input('speed: '))
play_passing = float(input('play_passing: '))
creation_passing = float(input('creation_passing: '))
creation_crossing = float(input('creation_crossing: '))
creation_shooting = float(input('creation_shooting: '))
pressure = float(input('pressure: '))
aggression = float(input('agression: '))
teamwidth = float(input('teamwidth: '))
wins = float(input('wins: '))
draws = float(input('draws: '))
loses = float(input('loses: '))
goals_scored = float(input('goals_scored: '))
goals_against = float(input('goals_against: '))
goal_diff = float(input('goal_diff: '))
with tf.compat.v1.Session() as sess: 
    sess.run(model)
    save_path = "./saved.cpkt"
    saver.restore(sess, save_path)

    data = ((speed, play_passing, creation_passing, creation_crossing, creation_shooting, pressure, aggression, teamwidth, wins, draws, loses, goals_scored, goals_against, goal_diff),)
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:14]
    dict = sess.run(hypothesis, feed_dict={X:x_data})
    print(dict[0])