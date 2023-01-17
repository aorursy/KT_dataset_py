

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
#print(os.listdir("../input"))

tf.__version__
a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
c = np.add(a, b)
print(c)
a = tf.constant([5, 3, 8])
b = tf.constant([3, -1, 2])
c = tf.add(a, b)
with tf.Session() as sess:
    result = sess.run(c)
    print(result)
a = tf.placeholder(dtype=tf.int32, shape=(None,))
b = tf.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)
with tf.Session() as sess:
    result = sess.run(c, feed_dict={a: [3, 4, 5],b: [-1, 2, 3]})
    print(result)
def compute_area(sides):
    # slice the input to get the sides
    a = sides[:,0]  # 5.0, 2.3
    b = sides[:,1]  # 3.0, 4.1
    c = sides[:,2]  # 7.1, 4.8
    
    # Heron's formula
    s = (a + b + c) * 0.5   # (a + b) is a short-cut to tf.add(a, b)
    areasq = s * (s - a) * (s - b) * (s - c) # (a * b) is a short-cut to tf.multiply(a, b), not tf.matmul(a, b)
    return tf.sqrt(areasq)
with tf.Session() as sess:
    # pass in two triangles\n",
    area = compute_area(tf.constant([
          [5.0, 3.0, 7.1],
         [2.3, 4.1, 4.8]
      ]))
    result = sess.run(area)
    print(result)
with tf.Session() as sess:
    # batchsize number of triangles, 3 sides
    sides = tf.placeholder(tf.float32, shape=(None, 3)) 
    area = compute_area(sides)
    result = sess.run(area, feed_dict ={sides:[[5.0, 3.0, 7.1],[2.3, 4.1, 4.8]]})
    print(result)
