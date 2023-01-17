#load modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

#get data

train = pd.read_csv("../input/train.csv" )

test = pd.read_csv("../input/test.csv")
#split training set into input (x_train) and output (y_train)  

x_train = train.iloc[: , 1:  ].values.T

y_train = train.iloc [ : , 0 ].values

x_test = test.iloc[: , : ].values.T

y_train = y_train.reshape(-1 , 1)

print(y_train.shape )
#normalizing 

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train = x_train/255

x_test = x_test/255

#one_hot_encoding

from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(sparse=False , categories = 'auto' )

y_train = one_hot.fit_transform(y_train)

y_train = y_train.T

print(y_train.shape )



#weight intialization

def init_params (l_dims) :  

  params = {}

  L= len(l_dims) 

  for l in range(1 , L ) :

    params['w'+ str(l)] = np.random.randn(l_dims[l] , l_dims[l-1]) * np.sqrt(1. / l_dims[l-1] )

    params['b'+ str(l)] = np.zeros ((l_dims[l] , 1)) * np.sqrt(1. /l_dims[l-1])        

  

  return params 

  
def init_adam(l_dims) :

  v = {}

  s = {}

  L= len(l_dims) 

  for l in range(1 , L ) :

      v["dw" + str(l)] = np.zeros(( l_dims[l] ,l_dims[l-1] ))

      v["db" + str(l)] = np.zeros((l_dims[l] , 1) )     

      s["dw" + str(l)] = np.zeros(( l_dims[l] ,l_dims[l-1] ))

      s["db" + str(l)] = np.zeros((l_dims[l] , 1) )      

  

  return v ,s
def relu (x) :  

     return x * (x > 0)
#derivative of relu function

def reLuD(x):

    return 1. * (x > 0)
def softmax(x):

    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=0)
def compute_loss(Y, Y_hat):

    

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))

    m = Y.shape[1]

    L = -(1./m) * L_sum



    return L
def forward_prop ( x , params , l_dims , keep_prop ) :

  L = len(l_dims)

  cache = {}

  cache ["a0"] = x 

  for l in range (1 , L) :

    z = np.dot( params['w' + str(l)] , cache['a' + str(l-1)] ) + params['b' + str(l)] 

    if( l == L-1 ) :

      a = softmax(z)

    else :  

      a = relu(z)

      #Dropout technique to prevent overfitting 

      d = np.random.rand( a.shape[0] ,  a.shape[1])

      d = d < keep_prop 

      a = a*d 

      a = a/keep_prop

    

    cache['z' + str(l)] = z 

    cache['a' + str(l)] = a 

    cache['d' +str(l) ] = d 

    

  return  a, cache 


def back_prop ( a , y , cache , params , l_dims , keep_prop ) :

    grads = {}

    L = len(l_dims)

    m = Y.shape[1]

    dz = a - Y

    for l in range(L-1 , 0 , -1 ) :

      dw = 1/m * np.dot ( dz , cache['a' + str(l-1)].T ) 

      db = 1/m * np.sum ( dz , axis = 1 , keepdims = True ) 

      if( l > 1) :

        da = np.dot(params['w' +str(l)].T , dz )

        da = da*cache["d"+str(l-1)] 

        da = da/keep_prop        

        dz =  da * reLuD(cache['z' + str(l-1)]) 

      grads["dw" + str(l)] = dw 

      grads["db" + str(l)] = db 



    return grads
#update weights with Adam algorithm 

def update_params( l_dims , params , grads , v , s , lr , beta1 , beta2 ,t ) :

  v_c = {} 

  s_c = {}

  for l in range ( 1 , len(l_dims) ) :

    v['dw' + str(l) ] = beta1*v['dw' + str(l)] + (1-beta1)*grads['dw'+ str(l) ]

    v['db' + str(l) ] = beta1*v['db' + str(l)] + (1-beta1)*grads['db'+ str(l) ]

    v_c["dw" + str(l)] = v["dw" + str(l)] / (1 - np.power(beta1, t))

    v_c["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t)) 

    

    s["dw" + str(l)] = beta2*s["dw" + str(l)] + (1-beta2) * np.power(grads['dw'+str(l)] , 2 )

    s["db" + str(l)] = beta2*s["db" + str(l)] + (1-beta2) * np.power (grads['db' + str(l)] , 2 ) 

    s_c["dw" + str(l)] = s["dw" + str(l)] / (1 - np.power(beta2, t))

    s_c["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t)) 

    params['w' + str(l)] = params['w' + str(l)] - lr * v_c["dw" + str(l)] / np.sqrt(s_c["dw" + str(l)] + 1e-8)

    params['b' + str(l)] = params['b' + str(l)] - lr * v_c["db" + str(l)] / np.sqrt(s_c["db" + str(l)] + 1e-8)

 

   



  return params ,v, s    
def accurcy (y_p , y ) :   

    m = y_p.shape[1]

    sum = 0

    for i in range(0,m) :

      if( np.argmax(y_p[: , i])  == np.argmax( y[: , i] ) ) :

         sum = sum+1

    return sum/m
l_dims = [x_train.shape[0] ,256, 10] #NN architecture 2

m = x_train.shape[1] #num of training examples 

params = init_params(l_dims) 

v,s = init_adam(l_dims)

keep_prop = 0.4 # Dropout parameter 

batch_size = 32 

epochs = 50 #num of epochs  

lr = 0.001 # learning rate 



for i in range(epochs):



    # shuffle training set

    permutation = np.random.permutation(m)

    X_train_shuffled = x_train[:, permutation]

    Y_train_shuffled = y_train[:, permutation]

     

    for j in range(math.ceil(m/batch_size )) :



        # get mini-batch

        begin = j * batch_size

        end = min(begin + batch_size, m - 1)

        X = X_train_shuffled[:, begin:end]

        Y = Y_train_shuffled[:, begin:end]

        m_batch = end - begin



        # forward and backward

        a,cache = forward_prop(X, params , l_dims , keep_prop )

        grads = back_prop(a, Y, cache ,params, l_dims , keep_prop)



        # updat parameters with gradient descent

        params ,v,s = update_params(l_dims, params , grads , v , s , lr , 0.9, 0.999 ,j+1 )

     

    # compute cost on training set 

    a , cache = forward_prop(x_train, params,l_dims , 1)

    train_loss = compute_loss(y_train, a)



    print( "epoch " + str(i) + " loss = " , train_loss )
#compute accuracy on training set 

a ,cache = forward_prop(x_train, params,l_dims ,1)

print(accurcy(a , y_train))

#predict

y_hat ,cache = forward_prop(x_test, params,l_dims,1 )

y_pred = np.argmax(y_hat,axis=0) #label test data 

m_test = x_test.shape[1] #num of test set examples 

output_file = "submission.csv"

id = []

for i in range(m_test ) :

    id.append(i+1)

    

    

submission = pd.DataFrame({'ImageId': id , 'Label': y_pred})

submission.to_csv('submission.csv', index=False)




