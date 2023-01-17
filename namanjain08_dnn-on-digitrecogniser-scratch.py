# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
display(train)
X_train = train[: (int((len(train) * 0.8)))]
X_cv = train[(int((len(train) * 0.8))) :]
X_tr = np.array(X_train.drop(["label"],axis=1).copy()).T
Y_tr = np.array([X_train['label']])
X_cross = np.array(X_cv.drop(["label"],axis=1).copy()).T
Y_cross = np.array([X_cv['label']])
print(Y_tr.shape)
print(X_tr.shape)

print(Y_cross.shape)
print(X_cross.shape)
X_test = np.array(test).T
#Y_te = np.array([X_test['label']])
#print(Y_te.shape)
print(X_test.shape)
shape = (Y_tr.size, Y_tr.max() + 1)
one_hot = np.zeros(shape)

rows = np.arange(Y_tr.size)

one_hot[rows , Y_tr] = 1
Y_tr  = one_hot.T
print(Y_tr.shape)


shape = (Y_cross.size, Y_cross.max() + 1)
one_hot = np.zeros(shape)

rows = np.arange(Y_cross.size)

one_hot[rows , Y_cross] = 1
Y_cross  = one_hot.T
print(Y_cross.shape)
X_cross = X_cross/255
X_tr = X_tr/255
X_test = X_test/255
def relu(X):
  X[X<0] = 0
  return X

def softmax(X):
    
  
  X = X - np.max(X)

  A = np.exp(X)/(np.sum(np.exp(X) , axis = 0,keepdims = True ))
  return A

def intialise_param(layer_dims):
 
  parameters = {}
  L = len(layer_dims)
 
  for l in range(0 , L-1 ):
    parameters["W" + str(l+1)] = np.random.randn(layer_dims[l+1] , layer_dims[l]) * (np.sqrt(1/(layer_dims[l])))
    parameters["b" + str(l+1)] = np.zeros((layer_dims[l+1] , 1)) 

  return parameters   




def activation(A_prev ,parameters, X ,d):
 
  linear_cache = {}
  activation_cache = {}
  L = len(d)
  A_prev = X
  for l in range(1 ,L-1):
    linear_cache["Z" + str(l)] = np.dot(parameters["W" + str(l)] , A_prev) + parameters["b" + str(l)]
    activation_cache["A" + str(l)] = relu(linear_cache["Z" + str(l)])
    A_prev = activation_cache["A" + str(l)]

  return linear_cache , activation_cache 




def final_act(linear_cache ,activation_cache,parameters):
  L = len(parameters)//2
  ZL =  np.dot(parameters["W" + str(L)] , activation_cache["A" + str(L-1)]) + parameters["b" + str(L)]
  AL = softmax(ZL)
  
  return ZL,AL


def computecost(Y , X ,A):
  m = X.shape[1]
  

  J = -(1./m) * (np.sum((Y * np.log(A + 1e-8))))

  return J


def backprop(activation_cache , linear_cache ,parameters,A ,X ,Y):
  m = X.shape[1]
  
  grads = {}
  

  L = len(parameters)//2

  dZL = A - Y
  for l in reversed(range(1 , L + 1 )):
    if l == 1:
      activation_cache["A" + str(l - 1)] = X
    
    grads["dW" + str(l)] = (1./m) * (np.dot(dZL , activation_cache["A" + str(l-1)].T))
    grads["db" + str(l)] = (1./m) * (np.sum(dZL , axis = 1 ,keepdims = True))


    if l == 1:
      break
    
    dZL = (np.dot(parameters["W" + str(l)].T , dZL)) * (np.int64(activation_cache["A" + str(l-1)] >0))

  return grads

def gd_with_moment(layer_dims):
  v = {}
  s = {}

  L = len(layer_dims)
 
  for l in range(0 , L-1 ):
    v["dW" + str(l+1)] = np.zeros((layer_dims[l+1] , 1))
    v["db" + str(l+1)] = np.zeros((layer_dims[l+1] , 1))
    s["dW" + str(l+1)] = np.zeros((layer_dims[l+1] , 1))
    s["db" + str(l+1)] = np.zeros((layer_dims[l+1] , 1))

  return v ,s
def update_param(grads , parameters,alpha ,d , v,s  , beta1 , beta2  , moment ,t):

  

    
  L = len(parameters)//2

  if moment == 'momentum' :

    for l in range(1 , L):

      v["dW" + str(l)] = beta1 * v["dW" + str(l)]   +   (1 - beta1) * grads["dW" + str(l)]
      v["db" + str(l)] = beta1 * v["db" + str(l)]   +   (1 - beta1) * grads["db" + str(l)]

      parameters[ "W" + str(l)] = parameters[ "W" + str(l)] - alpha * v["dW" + str(l)]
      parameters["b" + str(l)] = parameters["b" + str(l)] - alpha * v["db" + str(l)]

  if moment == 'rmsprop' :

    for l in range(1 , L):

      s["dW" + str(l)] = beta2 * s["dW" + str(l)]   +   (1 - beta2) * (grads["dW" + str(l)] * grads["dW" + str(l)] )
      s["db" + str(l)] = beta2 * s["db" + str(l)]   +   (1 - beta2) * (grads["db" + str(l)] * grads["db" + str(l)])

      parameters[ "W" + str(l)] = parameters["W" + str(l)] - alpha * s["dW" + str(l)]
      parameters["b" + str(l)] = parameters["b" + str(l)] - alpha * s["db" + str(l)]

  if moment == 'adam' :

    for l in range(1 , L):

      v["dW" + str(l)] = beta1 * v["dW" + str(l)]   +   (1 - beta1) * grads["dW" + str(l)]
      v["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1,t))
      v["db" + str(l)] = beta1 * v["db" + str(l)]   +   (1 - beta1) * grads["db" + str(l)]
      v["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1,t))
      s["dW" + str(l)] = beta2 * s["dW" + str(l)]   +   (1 - beta2) * (grads["dW" + str(l)] * grads["dW" + str(l)] )
      s["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2,t))
      s["db" + str(l)] = beta2 * s["db" + str(l)]   +   (1 - beta2) * (grads["db" + str(l)] * grads["db" + str(l)])
      s["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2,t))

      parameters["W" + str(l)] = parameters["W" + str(l)] - alpha * (v["dW" + str(l)] / (np.sqrt(s["dW" + str(l)]) + 1e-8))
      parameters["b" + str(l)] = parameters["b" + str(l)] - alpha * (v["db" + str(l)] / (np.sqrt(s["db" + str(l)]) + 1e-8))


  if moment == 'none' :

    for l in range(1 , L):

      parameters["W" + str(l)] = parameters["W" + str(l)] - alpha * grads["dW" + str(l)]
      parameters["b" + str(l)] = parameters["b" + str(l)] - alpha * grads["db" + str(l)]           

  return parameters
  


def model(parameters ,X , v, s,Y ,num_iter ,alpha , d ,batch_size ,t):
  costs = []
  J = 100

  for i in range(1 , num_iter + 1):

    num =int(X_tr.shape[1]/batch_size)
    for k in range(0,num ):
      if k != int(X_tr.shape[1]/batch_size) :
        X_batch = X_tr[:,k * batch_size : (k+1) * batch_size]
        Y_batch = Y_tr[:,k*batch_size : (k+1) * batch_size]
  
      else:
        X_batch = X_train[:,k * batch_size : -1]
        Y_batch = Y_train[:,k*batch_size : -1]
      A_prev = np.copy(X_batch)
      linear_cache , activation_cache  = activation(A_prev ,parameters = parameters , X = X_batch, d = layer_dims)

      ZL, AL= final_act(linear_cache = linear_cache  ,activation_cache = activation_cache ,parameters = parameters)

      J = computecost(Y = Y_batch , X = X_batch ,A = AL)
 
   

      grads = backprop(activation_cache = activation_cache , linear_cache = linear_cache ,parameters = parameters , A = AL ,X = X_batch ,Y = Y_batch)

     
      t = t + 1
      parameters = update_param(grads = grads ,parameters = parameters, alpha = alpha , d = d ,v = v,s = s ,
                                beta1 = 0.9 , beta2 = 0.999 , moment = "none" , t = t)

        
      if i == 3  :
          alpha = alpha/(1 + 0.001 * i)
    

      if i % 1 == 0:
        print("Cost after {}th iteration is {} ".format(i, J))
        print(alpha)
      costs.append(float(J))    

  plt.plot(list(range(len(costs))) , costs)  
  plt.show()

  return parameters, AL



import matplotlib.pyplot as plt
m = X_tr.shape[1]
layer_dims = [784,700,500,300,150,100,10]
parameters = intialise_param(layer_dims)
v ,s = gd_with_moment(layer_dims = layer_dims )
parameters , AL = model(parameters = parameters ,X = X_tr,v = v, s = s,Y = Y_tr,num_iter = 6,alpha = 0.2,d = layer_dims,batch_size = 32 ,t =1)
A_prev = np.copy(X_tr)
linear_cache , activation_cache  = activation(A_prev ,parameters = parameters , X = X_tr, d = layer_dims)

ZL, AL= final_act(linear_cache = linear_cache  ,activation_cache = activation_cache ,parameters = parameters)

for i in range(0,AL.shape[1]):
  a = np.max(AL[:,i])
  AL[:,i] = (AL[:,i] == a)
fai = np.absolute(AL - Y_tr)  
cn = 0
for i in range(0,fai.shape[1]):
  if np.sum(fai[:,i]) == 0:
    cn += 1
print(cn)
print("Train_Accurcay: " + str((cn / AL.shape[1]) * 100))
A_prev = np.copy(X_cross)
linear_cache , activation_cache  = activation(A_prev ,parameters = parameters , X = X_cross, d = layer_dims)

ZL, AL= final_act(linear_cache = linear_cache  ,activation_cache = activation_cache ,parameters = parameters)


for i in range(0,AL.shape[1]):
  a = np.max(AL[:,i])
  AL[:,i] = (AL[:,i] == a)
fai = np.absolute(AL - Y_cross)  
cn = 0
for i in range(0,fai.shape[1]):
  if np.sum(fai[:,i]) == 0:
    cn += 1
print(cn)
print("CrossValidation_Accurcay: " + str((cn / AL.shape[1]) * 100))
A_prev = np.copy(X_test)
linear_cache , activation_cache  = activation(A_prev ,parameters = parameters , X = X_test, d = layer_dims)

ZL, AL= final_act(linear_cache = linear_cache  ,activation_cache = activation_cache ,parameters = parameters)
A_test = np.argmax(AL,axis = 0)
C = np.arange(AL.shape[1]) + 1
Output = np.zeros((AL.shape[1],2))
Output[:,0] = np.copy(C)
Output[:,1] = np.copy(A_test)
Output = Output.astype(int)

print(Output)
np.savetxt("newresults.csv",Output,delimiter = ",",fmt = "%d")
