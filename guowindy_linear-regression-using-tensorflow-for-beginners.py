# Using tensorflow to do the linear regression !
# taken from https://github.com/davifrossard/iml/blob/master/03_TensorFlow/TensorFlow.pdf

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Generate Mock Data
#datax = np.linspace(0,10,100)[:, np.newaxis]
datax = np.linspace(1,8,100)[:, np.newaxis]
# print(datax.shape) # (100, 1)
datay = np.polyval([1,-14,59,-10], datax)+ 1.5*np.sin(datax)
# print(datay.shape) # (100, 1)
model_order = 5
datax = np.power(datax, range(model_order))
datax /= np.max(datax, axis = 0)
order = np.random.permutation(len(datax))
print(datax.shape,datay.shape)
portion = int(0.2* len(datax))
testx = datax[order[:portion]]
testy = datay[order[:portion]]
print(testx.shape,testy.shape)
trainx = datax[order[portion:]]
trainy = datay[order[portion:]]
datax[1:10,0]
print(datax.shape)
plt.plot(datax)
def linearRegTF(trainx, trainy, testx, testy,maxepoches, optprint = 0):
    ##### TensorFlow
    inputs = tf.placeholder(tf.float32, [None, model_order])
    outputs = tf.placeholder(tf.float32, [None,1])
    
    # model
    W = tf.Variable(tf.zeros([model_order ,1],dtype = tf.float32))
    y = tf.matmul(inputs,W)
    
    # Optimizer
    learning_rate = tf.Variable(0.5,trainable = False)
    costfun = tf.reduce_mean(tf.pow(y-outputs,2))
    sgd = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = sgd.minimize(costfun)
    
    epoches = 1
    lastcost = 0
    #maxepoches = 100000
    sess = tf.Session()
    cost_all = []
    print("Begin Training Now")
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)
        while True:
            sess.run(train_op,feed_dict = {inputs: trainx, outputs: trainy })
            if epoches%3000 == 0:
                cost = sess.run(costfun,feed_dict = {inputs:trainx, outputs: trainy})
                cost_all.append(cost)
                if optprint == 1:
                    print("Epoch: %d - Error: %.4f" % (epoches, cost))
                if abs(lastcost - cost) < 1e-3 :
                    print("Converged !")
                    break
                elif epoches > maxepoches:
                    print("Exceed the maximum epoches number !")
                    break
                lastcost = cost
            epoches += 1
        Weight = W.eval()
        # print("W = ", W.eval())
        print("Test Cost = ", costfun.eval({inputs: testx, outputs : testy}))
    return cost_all, Weight 

[cost_all , Weight] = linearRegTF(trainx, trainy, testx, testy, 50000, 1)
print("W = ", Weight.T)
ymodel = np.polyval(Weight[::-1],np.linspace(0,1,200))

f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 5))
ax1.plot(np.log(cost_all)) # exaggerate the slope 
ax1.set_title('logarithm of the cost')
ax2.plot(np.linspace(0,1,200),ymodel,c ='g', label = 'Model')
ax2.plot(datax[:,1],datay,c ='k',label = "raw")
ax2.scatter(trainx[:,1],trainy,c ='b', label = 'Train set')
ax2.scatter(testx[:,1],testy,c ='r', label = 'Test set')
ax2.grid()
ax2.legend(loc = 'upper left')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_xlim(0,1)
ax2.set_title('plot them all !')

plt.show()
# Analytical solution !! 
#  W X = Y  ==>  W = Y * pinv(X)
a = np.linalg.pinv(trainx)
b = trainy
print(a.shape, b.shape)
np.tensordot(a,b, axes=1).T
datax = np.linspace(0,8,100)[:, np.newaxis]
datay = np.polyval([1,-14,59,-10], datax)+ 1.5*np.sin(datax)
model_order = 5
datax = np.power(datax, range(model_order))
datax /= np.max(datax, axis = 0)  # normalization by divided by maximum
order = np.random.permutation(len(datax))
print(datax.shape,datay.shape)
portion = int(0.2* len(datax))
testx = datax[order[:portion]]
testy = datay[order[:portion]]
print(testx.shape,testy.shape)
trainx = datax[order[portion:]]
trainy = datay[order[portion:]]
#  TensorFlow
[cost_all , Weight] = linearRegTF(trainx, trainy, testx, testy, 50000)
ymodel = np.polyval(Weight[::-1],np.linspace(0,1,200))
#   plot the result
f, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 4))
ax1.plot((cost_all))# exaggerate the slope 
ax1.set_title('logarithm of the cost')
ax2.plot(np.linspace(0,1,200),ymodel,c ='g', label = 'Model')
ax2.plot(datax[:,1],datay,c ='k',label = "raw")
ax2.scatter(trainx[:,1],trainy,c ='b', label = 'Train set')
ax2.scatter(testx[:,1],testy,c ='r', label = 'Test set')
ax2.grid()
ax2.legend(loc = 'upper left')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_xlim(0,1)
ax2.set_title('plot them all !')
f.subplots_adjust(hspace=1)
plt.show()
