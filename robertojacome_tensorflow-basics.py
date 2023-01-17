import tensorflow as tf



print(tf.__version__)
hello = tf.constant(name='hello_', value='Hello world!!!')

hello2 = tf.constant(value='Hello world 2!!! ')



with tf.Session() as sess:

    print (sess.run(hello))
a = tf.constant(34)

b= tf.constant(55)

c = a+b
print(c)



with tf.Session() as sess:

    print(sess.run(c))
mat = tf.constant([[3,6], [8,5], [1,4]])

vec = tf.constant([[3], [5]])
print(mat)
print(mat.shape)
print(mat.dtype)
out = tf.matmul(mat,vec)
print(out)
print(out.op)
with tf.Session() as sess:

    print(sess.run(out))
a=tf.constant(name='op', value=500)
print(a)
a=tf.constant(name='op', value=1000)
print(tf.get_default_graph())
graph = tf.get_default_graph()
b=graph.get_tensor_by_name('op:0')
with tf.Session() as sess:

    print(sess.run(b))
for op in graph.get_operations():

    print(op)
aa = tf.constant(43.45)

bb= tf.constant(23.67)

cc= aa + bb
with tf.Session() as sess:

    print (cc.eval())
with tf.Session() as sess:

    print (sess.run([cc,aa,bb]))
v_a = tf.constant(5)

v_b = tf.placeholder(tf.float32)
v_c = tf.Variable(5)
with tf.Session() as sess:

    sess.run(v_c.initializer)

    print(sess.run(v_c))
var1 = tf.get_variable(name='myvar1', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())

var2 = tf.get_variable(name='myvar2', shape=(), dtype=tf.float32, initializer=tf.ones_initializer())

var3 = tf.get_variable(name='myvar3', shape=(), dtype=tf.float32, initializer=tf.random_uniform_initializer())

with tf.Session() as sess:

    for i in range(10):

        sess.run(var1.initializer)

        var2.initializer.run()

        var3.initializer.run()

        print(sess.run([var1, var2, var3]))

    
c_a = tf.constant(123.0)

op1 = var1.assign(3)

op2 = var2.assign(var2+42)

op3 = var3.assign(c_a+var3)

with tf.Session() as sess:

    sess.run(var1.initializer)

    var2.initializer.run()

    var3.initializer.run()

    for i in range(10):

        sess.run(op1)

        sess.run(op2)

        sess.run(op3)

        print(sess.run([var1, var2, var3]))

    
def func(x, y):

    return x*x + y*y
x0, y0 = 2, 3

import numpy as np
for angle in np.linspace(0, 2*np.pi, 100):

    x, y = x0 + 0.1* np.cos(angle), y0 + 0.1*np.sin(angle)

    diff = func(x, y) - func(x0, y0)

    print('angle: ', angle, 'diff', diff)
np.tan(0.9519977738150889)
np.arctan(6/4)
g_x = tf.placeholder(tf.float32)

g_y = tf.placeholder(tf.float32)

fxy = g_x*g_x + g_y*g_y
grad = tf.gradients(fxy, [g_x, g_y])
with tf.Session() as sess:

    print(sess.run(grad, feed_dict={g_x:2, g_y:3}))
pts = [[2, 14], [3,12], [1,11], [3,15], [5,14], [4,12], [5,15], [2,11]]
import numpy as np

import matplotlib.pyplot as plt
pts = np.array(pts)

plt.scatter(pts[:,0], pts[:,1]) #all rows and column 0 .....
grid = [[m,c] for m in range(15) for c in range(15)]
grid
def get_loss(pts, m, c):

    loss = 0

    for pt in pts:

        diff = pt[1] - (m*pt[0] + c)

        loss += (diff) * (diff)

    return loss    



for (m,c) in grid:

    print('loss: ', get_loss(pts,m,c), ', m: ', m, ', c: ', c)
dummy_x = np.random.random(1000)
#y = 5x +3, m=5   c=3

dummy_y = 5 * dummy_x + 3+ 0.1*np.random.randn(1000)
plt.scatter(dummy_x, dummy_y, s=0.1)
r_x = tf.placeholder(shape=(1000,), dtype=tf.float32)

r_y = tf.placeholder(shape=(1000,), dtype=tf.float32)

m= tf.get_variable(name='slope',dtype=tf.float32,shape=(),initializer=tf.ones_initializer())

c= tf.get_variable(name='intercept',dtype=tf.float32,shape=(),initializer=tf.ones_initializer())

#objective function

yest = m*r_x+c

loss = tf.losses.mean_squared_error(r_y,yest)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
with tf.Session() as sess:

    init = tf.global_variables_initializer()

    init.run()

    for e in range(100):

        _, val_loss = sess.run([optimizer, loss], feed_dict={r_x:dummy_x, r_y:dummy_y})

        print('loss: ', val_loss, ', m: ', m.eval(), ', c: ', c.eval())
writer = tf.summary.FileWriter(logdir='log', graph=tf.get_default_graph())
abc = tf.constant(3)

xyz = tf.Variable(5)

print('names ', abc.name, xyz.name)
with tf.variable_scope('myscope'):

    abc = tf.constant(3)

    xyz = tf.Variable(5)

print('names ', abc.name, xyz.name)