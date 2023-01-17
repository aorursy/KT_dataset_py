import os

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt





x_train = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],

                   [9.779],[6.182],[7.59],[2.167],[7.042],

                   [10.791],[5.313],[7.997],[3.1]],dtype = np.float32)

y_train = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],

                   [3.366],[2.596],[2.53],[1.221],[2.827],

                   [3.465],[1.65],[2.904],[1.3]],dtype=np.float32)



x_train
plt.plot(x_train,y_train,'ro')
x = tf.constant(x_train,name='x')

y = tf.constant(y_train,name='y')



w = tf.Variable(initial_value=tf.random_normal(shape=()),dtype= tf.float32,name='weight')

b=tf.Variable(initial_value=0,dtype = tf.float32,name='biase')

with tf.variable_scope('liner_model'):

    

    y_pred = w*x+b

sess = tf.InteractiveSession()# 开启交互式会话

sess.run(tf.global_variables_initializer())
y_pred_numpy=y_pred.eval(session=sess)

plt.plot(x_train,y_train,'bo',label='real')

plt.plot(x_train,y_pred_numpy,'ro',label='estimated')

plt.legend()
w.eval()
loss = tf.reduce_mean(tf.square(y-y_pred))

# 看看在当前模型参数下的误差是多少

print(loss.eval(session=sess))
w_grad,b_grad = tf.gradients(loss,[w,b])

print('w_grad:%.4f' % w_grad.eval(session=sess))

print('x_grad:%.4f'%b_grad.eval(session=sess))
lr = 1e-2

w_update = w.assign_sub(lr*w_grad)

b_update = b.assign_sub(lr*b_grad)

print ('w_update:%.4f'%w_update.eval(session=sess))

sess.run([w_update,b_update])
y_pred_numpy = y_pred.eval(session=sess)



plt.plot(x_train,y_train,'bo',label='real')

plt.plot(x_train,y_pred_numpy,'ro',label='estimated')

plt.legend()


sess.run(tf.global_variables_initializer())

for e in range(20):

    sess.run([w_update,b_update])

    y_pread_numpy = y_pred.eval(session=sess)

    loss_numpy = loss.eval(session=sess)



    print('epoch: {}, loss: {}'.format(e+1, loss_numpy))

    
print(w_update)

plt.plot(x_train, y_train, 'bo', label = 'real')

plt.plot(x_train, y_pred_numpy, 'ro', label = 'estimated')

plt.legend()