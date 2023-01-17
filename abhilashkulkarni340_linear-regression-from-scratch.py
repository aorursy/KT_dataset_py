import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.describe()
train.dropna(inplace=True)
print(train.isnull().any())
train.describe()
plt.figure(figsize=(15,15))
plt.axis([0,100,-10,110])
plt.scatter(train['x'],train['y'])
plt.show()
def hypothesis_function(W,x):
    return W[0]+W[1]*x
def gradient_descent(W,X,Y):
    total1=0
    total2=0
    alpha=0.0001
    i=0
    for x in X:
        total1+=(hypothesis_function(W,x)-Y.values[i])
        total2+=(hypothesis_function(W,x)-Y.values[i])*x
        i+=1
    return [W[0]-alpha*(total1/len(X)),W[1]-alpha*(total2/len(X))]
def cost_function(W,X,Y):
    total=0
    i=0
    for x in X:
        total+=(hypothesis_function(W,x)-Y.values[i])**2
        i+=1
    
    return total/(2*len(X))
W=[0,0]
X=train['x']
Y=train['y']
err_list=[]
for i in range(50):
    err_list.append(cost_function(W,X,Y))
    W=gradient_descent(W,X,Y)
x_axis=[x for x in range(50)]
plt.plot(x_axis,err_list)
plt.show()
print(min(err_list))
x1=-10
x2=110
y1=hypothesis_function(W,x1)
y2=hypothesis_function(W,x2)
plt.figure(figsize=(15,15))
plt.plot([x1,x2],[y1,y2],color='red')
plt.scatter(train['x'],train['y'])
plt.show()
X_test=test['x']
Y_test=test['y']
cost_function(W,X_test,Y_test)
