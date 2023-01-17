from mxnet import nd as np
#import numpy as np
from mxnet import autograd
import pandas as pd

data_train = pd.read_csv('../input/train.csv')
train_x=data_train[['R1C1','R1C2','R2C1','R2C2']].copy()
x_train=train_x.as_matrix()
x_train=np.array(x_train)
train_y=data_train[['IsStairs']].copy()
y_train=train_y.as_matrix()
y_train=np.array(y_train)
type(y_train)
#print (y_train.shape)

#np.delete(x_train, 4, 1)

data_test=pd.read_csv('../input/test.csv')
test_x=data_test[['R1C1','R1C2','R2C1','R2C2']].copy()
x_test=test_x.as_matrix()
x_test=np.array(x_test)
test_y=data_test[['IsStairs']].copy()
y_test=test_y.as_matrix()
y_test=np.array(y_test)
def relu(x):
    return np.maximum(x,np.zeros_like(x))

print(relu(np.array([8,3,-2,-9,1])))

def sigmoid(x):
        return (1 / (1 + np.exp(-x)))

#print(sigmoid(np.array([3, 5])))
def softmax(x):
    x= x-np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

#print(softmax(np.array([0.3,0.7,1.9,0.1])))
def sgd(params,lr):
    for param in params:
        param-=lr*param.grad
def crossentropy(y,yhat):
        #print(y,yhat)
        #return -np.nansum(y*np.log(yhat),axis=1)
        return -np.nansum(y*np.log(yhat))

#print(crossentropy([ 0.47570884],[1.0]))
#print(crossentropy(np.array([[1]]),np.array([[0]])))   
input_shape=4
h1_shape=3
output=2
lr=0.01
scale=0.01
w1=np.normal(shape=(input_shape,h1_shape),scale=scale)
w2=np.normal(shape=(h1_shape,output),scale=scale)
b1=np.normal(shape=h1_shape,scale=scale)
b2=np.normal(shape=output,scale=scale)
params=[w1,b1,w2,b2]
for param in params:
    param.attach_grad()
def net(x):
    int1=relu(np.dot(x,w1)+b1)
    int2=softmax(np.dot(int1,w2)+b2)
    return (int2)
#x=np.array([1,2,3,4])
#print(type(x))
#print(net(x))
#x=x_train[1]
#print(type(x))
#net(x)


def evaluate(data,lebel):
          numerator=0
          denominator=0
          num_iters= len(data)
          #while(1)
          for i in range (num_iters):
            x= data[i].reshape((-1,4))
            #print(x[i])
            y= lebel[i]
            #print(y)
            #output= net(x)  #softmax generated an array containing probabilities

            output= np.argmax(net(x), axis=1)
            #print(output,end=' ')
            #print(y)
            numerator+=sum(output==y)
            denominator+=len(x)

          #print (numerator/denominator)  
          print (numerator)
          print (denominator) 
epochs=10
batch_size=1
lr=0.001
num_iters= len(x_train) 
num_iters

cur_loss=0.0
for e in range(epochs):
 for i in range(0,num_iters):
    with autograd.record():
        net_output= net(x_train[i])
        #print(net_output,y_train[i])
        loss= crossentropy(y_train[i],net_output)
        #print(loss)
    loss.backward()
    cur_loss+=np.sum(loss).asscalar()
    sgd(params,lr)
    
evaluate(x_test,y_test)
