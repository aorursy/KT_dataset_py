import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x= np.linspace (-1,1,1001)
y= 2 * x+ np.random.randn(*x.shape) * 0.33

plt.scatter(x,y)
def createline(x,w=0,b=0):
    return w*x+b  #mx+b line equation
plt.scatter(x,y)
w=2.0087843
b= -0.0057927
predict = createline(x,w,b)

plt.plot(x,predict, c='r')
plt.show()
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(1,activation='relu', input_dim=1))
model.compile(optimizer='rmsprop' , loss= 'mse' , metrics=['mae','accuracy', 'mse'])
history = model.fit(x,y, epochs=100, batch_size=100)
model.evaluate(x,y)
model.summary()
w,b=model.get_weights()
w
b
plt.scatter(x,y)
pline = createline(x,w[0],b)
plt.plot(x,pline,c='r')
plt.show()
x1=[0.001]
y1 = model.predict(x1)
plt.scatter(x,y)
plt.scatter(x1,y1)
plt.plot(x,pline,c='r')
plt.show()
def softmax(x):
    prob = []
    for i in x:
        b=i/np.sum(x)
        prob.append(b)
    return prob     
data = np.array([1,3,5,6,7])
softmax(data)
np.sum(softmax(data))
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
test = np.linspace(-10,10,101)
y1 = sigmoid(test)
y1
plt.plot(y1)
plt.show()
def relu(x):
    x[x<=0] = 0
    return x
y1 = relu(test)
y1
plt.plot(y1)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
a = np.linspace(-10,10,101)
b = np.tanh(a)
plt.plot(b)
plt.show()
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) **2).mean())

def rmse1(predictions,targets):
    differences = predictions - targets
    differences_squared = differences **2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt( mean_of_differences_squared)
    return rmse_val
rmse(y,pline)
import numpy as np
d = [0.000, 0.166, 0.333]
p = [0.000, 0.254, 0.998]

print("d is: " +str(["%.8f" % elem for elem in d]))
print("p is: " +str(["%.8f" % elem for elem in p]))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) **2).mean())

rmse_val = rmse(np.array(d), np.array(p))
print("rms error is: " + str(rmse_val))