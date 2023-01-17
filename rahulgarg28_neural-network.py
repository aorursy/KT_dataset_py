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
data = pd.read_csv("../input/fashion-mnist_train.csv")
x = data.values
X = x[:,1:]
y = x[:,0]
X = X/255.0
test = pd.read_csv("../input/fashion-mnist_test.csv")
x_ = test.values
X_ = x_[:,1:]
y_ = x_[:,0]
X_ = X_/255.0
print (X.shape,y.shape)
print (X_.shape,y_.shape)
X_train = X
X_val = X_

y_train = y
y_val = y_
print (X_.shape,y_.shape)
INP = 784
H1_SIZE = 256
H2_SIZE = 64
OUT = 10
BATCH_SIZE = 200
EPOCH = 50
LR = 0.0003
def accuracy(y_pred,y_act):
    return ( 100.0*np.sum(y_pred==y_act)/y_pred.shape[0] )
def ini_weights():
    
    np.random.seed(0)
    model = {}
    model["W1"] = np.random.randn(INP,H1_SIZE)/np.sqrt(INP)
    model["B1"] = np.zeros((1,H1_SIZE))
    model["W2"] = np.random.randn(H1_SIZE,H2_SIZE)/np.sqrt(H1_SIZE)
    model["B2"] = np.zeros((1,H2_SIZE))
    model["W3"] = np.random.randn(H2_SIZE,OUT)/np.sqrt(H2_SIZE)
    model["B3"] = np.zeros((1,OUT))
    
    return model
def back_prop(model, x , a1, a2, y_out, y_act):
    delta4 = y_out
    delta4[range(y_act.shape[0]), y_act] -= 1
    dw3 = (a2.T).dot(delta4)
    db3 = np.sum(delta4, axis=0)
    delta3 = (1-np.square(a2))*delta4.dot(model["W3"].T)
    dw2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0)
    delta2 = (1-np.square(a1))*delta3.dot(model["W2"].T)
    dw1 = (x.T).dot(delta2)
    db1 = np.sum(delta2, axis=0)
    
    model["W1"] -= LR*dw1
    model["B1"] -= LR*db1
    model["W2"] -= LR*dw2
    model["B2"] -= LR*db2
    model["W3"] -= LR*dw3
    model["B3"] -= LR*db3
    
    return model
def predict(y_out):
    return np.argmax(y_out,axis=1)
def forward_prop(model,x):
    z1 = x.dot(model["W1"]) + model["B1"]
    a1 = np.tanh(z1)
    z2 = a1.dot(model["W2"]) + model["B2"]
    a2 = np.tanh(z2)
    z3 = a2.dot(model["W3"]) + model["B3"]
    h_x = np.exp(z3)
    y_out = h_x/ np.sum(h_x, axis=1, keepdims=True)
    
    return a1,a2,y_out
def loss(model, y_pred, y_act):
    correct_logprobs = -np.log(y_pred[range(y_act.shape[0]), y_act])
    l = np.sum(correct_logprobs)
    
    return(1.0/y_pred.shape[0])*l

def main():
    training_loss = []
    val_loss = []
    val_acc = []
    model = ini_weights()
    
    for e in range(EPOCH):
        print ("\n Epoch : %d" %(e+1))
        count = 0
        while (count + BATCH_SIZE) < y_train.shape[0]:
            batch_data = X_train[count:(count+BATCH_SIZE),:]
            batch_labels = y_train[count:(count+BATCH_SIZE),]
            count += BATCH_SIZE
            
            a1,a2,y_out = forward_prop(model,batch_data)
            model = back_prop(model,batch_data,a1,a2,y_out,batch_labels)
            
        _,_,p = forward_prop(model,X_train)
        training_loss.append(loss(model,p,y_train))
            
        print('training_loss : %.3f' %(loss(model,p,y_train)))
            
        _,_,p = forward_prop(model,X_val)
        pred = predict(p)
        val_loss.append(loss(model,p,y_val))
        val_acc.append(accuracy(pred,y_val))
        print('validation_loss : %.3f' %(loss(model,p,y_val)))
        print('validation_acc : %.3f' %(accuracy(pred,y_val)))
    print ("##############COMPLETED############")
    
    return training_loss,val_loss,val_acc
training_loss,val_loss,val_acc = main()
plt.plot(val_loss)
plt.plot(val_acc)







