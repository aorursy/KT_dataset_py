# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf



# 产生虚拟数据

x_data = np.float32(np.random.rand(2,100))

y_data = np.dot([0.100, 0.200], x_data) +0.300



#构造线性模型

b = tf.Variable(tf.zeros([1]))

w = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))

y = tf.matmul(w, x_data) + b

print(y)



#最小化方差

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)



#初始化变量

init = tf.initialize_all_variables()



#启动图

sess = tf.Session()

sess.run(init)



# 拟合平面

for step in range(0, 201):

    sess.run(train)

    if step % 20 ==0:

        print(step, sess.run(w), sess.run(b))
# tf 显示 hello tensorflow

hello = tf.constant("Hello, Tensorflow!")

sess = tf.Session()

print(sess.run(hello))
# tf 变量相加

a = tf.constant(10)

b = tf.constant(20)

print(sess.run(a+b))
matrix1 = tf.constant([[3., 3.]])

matrix2 = tf.constant([[2.], [2.]])

print(matrix1)

print(matrix2)



# 创建矩阵乘法

product = tf.matmul(matrix1, matrix2)



# 在会话中启动视图

sess = tf.Session()

result = sess.run(product)

print(result)



#任务完成，关闭会话

sess.close()
# 也可以使用with 模块，来自动完成关闭操作

with tf.Session() as sess:

    result = sess.run(product)

    print(result)
with tf.Session() as sess:

    with tf.device("/gpu:1"):

        result = sess.run(product)

        print(result)

        
# 交互式会话

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])

a = tf.constant([3.0, 3.0])



#初始化

x.initializer.run()



#减法

add = tf.add(x,a)

print(add.eval())



sub = tf.add(x,-a)

print(sub.eval())
#创建一个变量，初始化为0

val = tf.Variable(0, name='counter')

print(val)



# 创建一个op，使变量加一

one = tf.constant(1)

new_val = tf.add(val, one)

update = tf.assign(val, new_val)



#启动图后，biubiu先初始化op

init = tf.initialize_all_variables()



# 启动图，运行op

with tf.Session() as sess:

    sess.run(init)

    

    print(sess.run(val))

    

    for _ in range(3):

        sess.run(update)

        print(sess.run(val))
input1 = tf.constant(3.0)

input2 = tf.constant(2.0)

input3 = tf.constant(5.0)

intermed = tf.add(input2, input3)



add = tf.add(input1, intermed)



with tf.Session() as sess:

    result = sess.run([intermed, add])

    print(result)
# 先定义计算公式

input1 = tf.placeholder(tf.float32)

input2 = tf.placeholder(tf.float32)

output = tf.add(input1, input2)



# 再传入数值

with tf.Session() as sess:

    print(sess.run([output], feed_dict = {input1:[7.], input2:[2.]}))