# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
Dataframe=pd.read_csv("../input/mnist-data/train(1).csv")
print(Dataframe.head(10))
print("Number of rows = "+str(len(Dataframe)))
print("Number of columns = "+str(len(Dataframe.columns)))
print(Dataframe['label'])
for u in Dataframe['label'].unique():
    print(u)
    Dataframe['label'+str('_')+str(u)] = list((Dataframe['label']==u)*1)
Dataframe= Dataframe.drop(['label'],axis=1)
print(Dataframe.head(10))
temp=Dataframe.as_matrix()
print(temp)
print(temp.shape)
for i in range(0,5):
        X=temp[i,0:temp.shape[1]-10]
        #print(X)
        I=X.reshape((28,28))
        
        #print(I)
        #print(I.shape)
        plt.imshow(I)
        plt.show()
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(temp[:,0:temp.shape[1]-10], temp[:,temp.shape[1]-10:], test_size=0.33)
print(test_y.shape)
import random
# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)
def batch_create(X,Y,batch_size=128,seed=0):
    m=X.shape[0]
    lst=random.sample(range(0,m),batch_size)
    batchX=X[lst,:]
    batchY=Y[lst]
    #print(X.shape)
    #print(Y.shape)
    return(batchX,batchY)

#Initializing neural network parameters
input_num_units = 28*28
output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01

#Initializing weights and bias
weights = {
    'output': tf.Variable(tf.random_normal([input_num_units, output_num_units], seed=seed))
}
biases = {
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

#Defining computation graph
output_layer = tf.add(tf.matmul(x, weights['output']), biases['output'])
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



#END OF DEFINING NEURAL NETWORK ARCHITECTURE

#Initializing variables
init = tf.initialize_all_variables()
#Running tensorflow Session
with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(temp.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_create(train_x,train_y)
            #print(batch_x.shape)
            #print(batch_y.shape)
            c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            avg_cost += c[1] / total_batch
            
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    
    print ("\nTraining complete!")    
    pred_train = np.argmax(output_layer.eval({x:train_x}),axis=1)
    print(pred_train.shape)
    #print(pred_train)
    train_yarg= np.argmax(train_y,axis=1)
    print(train_y.shape)
    train_acc = np.sum(pred_train==train_yarg)*1.0/train_y.shape[0]
    print('train accuracy : '+str(train_acc))
            
    pred_test = np.argmax(output_layer.eval({x:test_x}),axis=1)
    print(pred_test.shape)
    #print(pred_train)
    test_yarg= np.argmax(test_y,axis=1)
    print(test_y.shape)
    test_acc = np.sum(pred_test==test_yarg)*1.0/test_y.shape[0]
    print('test accuracy : '+str(test_acc))
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

from sklearn.cluster import KMeans
input_nodes=train_x.shape[1]
input_count=train_x.shape[0]
rbf_count=input_nodes+10
output_nodes=train_y.shape[1]

#Finding MU

kmeans = KMeans(n_clusters=rbf_count,verbose=1,n_init=1,max_iter=10,random_state=0).fit(train_x)
def cal_dist(x1,x2):
    dt=x1-x2
    dist=np.matmul(dt,dt.transpose())
    dist=np.sqrt(dist)
    return dist
import math

mu=kmeans.cluster_centers_
#rn=random.sample(range(0,input_count),rbf_count)
#mu=train_x[rn,:]
print(mu.shape)
sigma=np.ones(mu.shape[0])
sigma=sigma*1000

#Assigning Sigma
"""
for i in range(0,mu.shape[0]):
    print("SIGMA i = "+str(i))
    dtt=np.zeros(input_count)
    node=mu[i]
    for j in range(0,input_count):
        node2=train_x[j]
        dtt[j]=cal_dist(node,node2)
    dtt=np.argsort(dtt)
    summ=0
    for k in range(0,input_nodes):
         summ=summ+dtt[k]**2
    summ=summ/input_nodes
    summ=summ**0.5
    sigma[i]=summ
print(sigma)
"""

X_rbf=np.zeros((input_count,mu.shape[0]))
print(X_rbf.shape)

for i in range(0,input_count):
    print("TRAIN RBF i = "+str(i))
    for j in range(0,mu.shape[0]):
        r=cal_dist(train_x[i],mu[j])
        X_rbf[i][j]=np.exp(-1*(r**2)/(2*sigma[j]**2))
        
    

#print(X_rbf.shape)

print(train_y)
print(test_y)
X_rbf_test=np.zeros((test_x.shape[0],mu.shape[0]))
for i in range(0,test_x.shape[0]):
    print("TEST RBF i = "+str(i))
    for j in range(0,mu.shape[0]):
        r=cal_dist(test_x[i],mu[j])
        X_rbf_test[i][j]=math.exp(-1*(r**2)/(2*sigma[j]**2))
#Initializing neural network parameters
input_num_units = rbf_count
output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 100
batch_size = 128
learning_rate = 0.001

#Initializing weights and bias
weights = {
    'output': tf.Variable(tf.random_normal([input_num_units, output_num_units], seed=seed))
}
biases = {
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

#Defining computation graph
output_layer = tf.add(tf.matmul(x, weights['output']), biases['output'])
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



#END OF DEFINING NEURAL NETWORK ARCHITECTURE

#Initializing variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(input_count/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_create(X_rbf,train_y)
            #print(batch_x.shape)
            #print(batch_y.shape)
            c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            avg_cost += c[1] / total_batch
            
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    
    print ("\nTraining complete!")    
    pred_train = np.argmax(output_layer.eval({x:X_rbf}),axis=1)
    print(pred_train.shape)
    #print(pred_train)
    train_yarg= np.argmax(train_y,axis=1)
    print(train_y.shape)
    train_acc = np.sum(pred_train==train_yarg)*1.0/train_y.shape[0]
    print('train accuracy : '+str(train_acc))
    print(pred_train)
            
    pred_test = np.argmax(output_layer.eval({x:X_rbf_test}),axis=1)
    print(pred_test.shape)
    #print(pred_train)
    test_yarg= np.argmax(test_y,axis=1)
    print(test_y.shape)
    test_acc = np.sum(pred_test==test_yarg)*1.0/test_y.shape[0]
    print('test accuracy : '+str(test_acc))
    print(pred_test)
    
    print("TEST IMAGES")
    
    X1=plt.imread("../input/test-images-2/img_4.jpg")
    X1vect=np.reshape(X1,X1.shape[0]*X1.shape[1])
    X_rbf_1=np.zeros(mu.shape[0])
    for j in range(0,mu.shape[0]):
        r=cal_dist(X1vect,mu[j])
        X_rbf_1[j]=math.exp(-1*(r**2)/(2*sigma[j]**2))    
    plt.imshow(X1)
    plt.show()
    X_rbf_1=np.reshape(X_rbf_1,(1,X_rbf_1.shape[0]))
    pred_test = np.argmax(output_layer.eval({x:X_rbf_1}),axis=1)
    print("PREDICTION = "+str(pred_test))
    
    
    X2=plt.imread("../input/test-images-2/img_5.jpg")
    X2vect=np.reshape(X2,X2.shape[0]*X2.shape[1])
    X_rbf_2=np.zeros(mu.shape[0])
    for j in range(0,mu.shape[0]):
        r=cal_dist(X2vect,mu[j])
        X_rbf_2[j]=math.exp(-1*(r**2)/(2*sigma[j]**2))    
    plt.imshow(X2)
    plt.show()
    X_rbf_2=np.reshape(X_rbf_2,(1,X_rbf_2.shape[0]))
    pred_test = np.argmax(output_layer.eval({x:X_rbf_2}),axis=1)
    print("PREDICTION = "+str(pred_test))
    
    X3=plt.imread("../input/test-images-2/img_6.jpg")
    X3vect=np.reshape(X3,X3.shape[0]*X3.shape[1])
    X_rbf_3=np.zeros(mu.shape[0])
    for j in range(0,mu.shape[0]):
        r=cal_dist(X3vect,mu[j])
        X_rbf_3[j]=math.exp(-1*(r**2)/(2*sigma[j]**2))    
    plt.imshow(X3)
    plt.show()
    X_rbf_3=np.reshape(X_rbf_3,(1,X_rbf_3.shape[0]))
    pred_test = np.argmax(output_layer.eval({x:X_rbf_3}),axis=1)
    print("PREDICTION = "+str(pred_test))









