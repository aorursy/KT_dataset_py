import numpy as np

import matplotlib.pyplot as plt

import pandas

import math

import os
import tensorflow.compat.v1 as tf1

#For Compatibility with older code.

tf1.disable_eager_execution()
X = np.array([0.5,2.3,2.9])

Y = np.array([1.4,1.9,3.2])
plt.scatter(X,Y)

W = 0.64

plt.plot(range(math.ceil(np.max(X))+1),[W*i for i in range(math.ceil(np.max(X))+1)])

plt.show()
def ss_loss_1(B):

    #Using the X&Y decalred at the top

    return np.sum(((Y - (W*X + B)))**2)



plt.plot([p/1000 for p in range(-10000,10000)],[ss_loss_1(i) for i in [p/1000 for p in range(-10000,10000)]])

plt.axvline(x = 0, c = 'r')

plt.axhline(y = 0, c = 'r')

plt.xlim([-1,5])

plt.ylim([-1,10])
# To clear the defined variables and operations of the previous cell

tf1.reset_default_graph()   

# We used a fixed value for m

m = tf1.constant(0.64)

#We initialize b at 0

b = tf1.Variable(0.0)
error = 0

for x,y in zip(X,Y):

    y_hat = m*x+b

    error += (y - y_hat)**2
optimizer = tf1.train.GradientDescentOptimizer(learning_rate = 0.1)

train = optimizer.minimize(error)

init = tf1.global_variables_initializer()
with tf1.Session() as sess:

    sess.run(init)

    training_steps = 1

    writer = tf1.summary.FileWriter('logs/', sess.graph)

    for i in range(training_steps):

        sess.run(train)

    slope,intercept = sess.run([m,b])
print (" Slope = ",slope," Intercept = ",intercept)
f,ax = plt.subplots(figsize = (20,6))

plt.plot([p/1000 for p in range(-10000,10000)],[ss_loss_1(i) for i in [p/1000 for p in range(-10000,10000)]])







plt.axhline(y=0)



with tf1.Session() as sess:

    sess.run(init)

    training_steps = 10

    

    err = sess.run(error)

    slope,intercept = sess.run([m,b])

    plt.scatter(intercept, err, c = 'r', s = 150)

    

    for i in range(training_steps):

        sess.run(train)

        err = sess.run(error)

        slope,intercept = sess.run([m,b])

        plt.scatter(intercept, err)





plt.scatter(intercept, err,c = 'g', s = 150)    

plt.axvline(x = 0, c = 'r')

plt.axhline(y = 0, c = 'r')

plt.xlim([-1,5])

plt.ylim([-1,10])

plt.show()

print ("Converged Value ",intercept)
f,ax = plt.subplots(figsize = (20,6))



with tf1.Session() as sess:

    sess.run(init)

    training_steps = 10

    

    err = sess.run(error)

    slope,intercept = sess.run([m,b])

    plt.plot(range(4),[slope*i + intercept for i in range(4)])

    

    for i in range(training_steps):

        sess.run(train)

        err = sess.run(error)

        slope,intercept = sess.run([m,b])

        plt.plot(range(4),[slope*i + intercept for i in range(4)])



plt.scatter(X,Y,s = 100)

plt.show()
# This time we randomize both variables needed, also we enclose them inside an array

params = tf1.Variable([[8.0,8.]],dtype=tf1.float32)
Xmat = np.vstack([X,np.ones(len(X))])

Xmat = Xmat.astype(np.float32)

y_hat = tf1.matmul(params,Xmat)

error = tf1.reduce_sum((Y - y_hat)**2)
optimizer = tf1.train.GradientDescentOptimizer(learning_rate = 0.01)

train = optimizer.minimize(error)

init = tf1.global_variables_initializer()
from mpl_toolkits.mplot3d import axes3d



def viz_loss(x1,x2):

    return (1.4 - (x1  + x2*0.32))**2 + (1.9 - (x1  + x2*1.4))**2 + (3.2 - (x1  + x2*1.8))**2 



a1 = np.linspace(-8, 8)

a2 = np.linspace(-8, 8)

A1, A2 = np.meshgrid(a1, a2)

Z = viz_loss(A1, A2)



fig = plt.figure(figsize = (15,10))

ax = fig.add_subplot(111, projection="3d")





with tf1.Session() as sess:

    sess.run(init)

    training_steps = 100

    for i in range(training_steps):

        sess.run(train)

        slope,intercept =  sess.run(params)[0]

        SumSq = sess.run(error)

        ax.scatter(slope,intercept,SumSq, c = 'red')





ax.plot_surface(A1, A2, Z, lw=10,cmap="coolwarm", rstride=1, cstride=1, alpha = 0.8)

ax.contour(A1, A2, Z, 10, cmap="coolwarm",linestyles="solid", offset=-1, alpha = 0.1)

ax.contour(A1, A2, Z, 10, colors="k", linestyles="solid", alpha = 0.1)

 

    

ax.view_init(0, 120)    

plt.show()