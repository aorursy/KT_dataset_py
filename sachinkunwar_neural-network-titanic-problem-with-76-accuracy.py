# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()
train_y=train_data.iloc[0:800,1]

val_y=train_data.iloc[800:891,1]

# tarin_y=train_y[0:800]

# val_y=train_y[800:891]

train_y=train_y.as_matrix()

train_y=train_y.reshape(1,-1)

val_y=val_y.as_matrix()

val_y=val_y.reshape(1,-1)

train_data.drop(['Survived'],axis=1,inplace=True)
print(train_y.shape)

val_y.shape

titanic_data=pd.concat([train_data,test_data],axis=0,ignore_index=True)

titanic_data.head()
print(titanic_data.tail())

print(len(titanic_data))

titanic_data.index
dummy_col=['PassengerId','Name','Ticket','Fare','Embarked']

passengerId=titanic_data['PassengerId']

titanic_data.drop(dummy_col,axis=1,inplace=True)

titanic_data.head()
cabin_mode=titanic_data['Cabin'].mode()

titanic_data.Cabin.fillna('C23 C25 C27',inplace=True)

titanic_data.head()
sex_mode=titanic_data['Sex'].mode()

titanic_data['Sex'].fillna(value=sex_mode, inplace=True)

titanic_data.head()
age_mean=titanic_data['Age'].mean()

titanic_data['Age'].fillna(age_mean, inplace=True)

titanic_data.head(20)
sex = {'male': 1,'female': 0} 

titanic_data.Sex = [sex[item] for item in titanic_data.Sex] 
cabin_count=titanic_data['Cabin'].value_counts()

# print(cabin_count.index)

cabin_index=list(cabin_count.index)

# l[0]

cabin_cnt=list(cabin_count)

# cabin={}

# for i in range(len(titanic_data)):

#     titanic_data.iloc[i,]
titanic_data.Cabin.replace(to_replace=cabin_index,value=cabin_cnt,inplace=True)
titanic_data
for col in titanic_data.columns:

    titanic_data[col]=titanic_data[col]/titanic_data[col].max()

#     titanic_data[col]=fun(np.array(titanic_data[col]))

# titanic_data['Age']=fun(np.array(titanic_data['Age']))
titanic_data
train_x=titanic_data[:800]

val_x=titanic_data[800:891]

test_x=titanic_data[891:]
train_x=train_x.as_matrix()

test_x=test_x.as_matrix()

val_x=val_x.as_matrix()
print("shape of train_x = "+str(train_x.shape))

print("shape of test_x = "+str(test_x.shape))

print("shape of val_x = "+str(val_x.shape))
train_x=train_x.T

test_x=test_x.T

val_x=val_x.T
print("shape of train_x = "+str(train_x.shape))

print("shape of test_x = "+str(test_x.shape))

print("shape of val_x = "+str(val_x.shape))
def initialize_parameters(layer_dims):

    np.random.seed(1)

    L=len(layer_dims)

    parameters={}

    for l in range(1,L):

        parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01

        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

    return parameters



def relu(Z):

#     A=[max(0,i) for i in x ]

    A=np.where(Z>0,Z,0)

    return A,Z

def sigmoid(Z):

    A=1/(1+np.exp(-Z))

    return A,Z



def linear_forward(A,W,b):

    Z=np.dot(W,A)+b

    cache=(A,W,b)

    return Z,cache



def linear_activation_forward(A,W,b,activation):

    Z,linear_cache=linear_forward(A,W,b)

    if activation=='relu':

        A,activation_cache=relu(Z)

    elif activation=='sigmoid':

        A,activation_cache=sigmoid(Z)

    cache=(linear_cache,activation_cache)

    return A,cache

def L_model_forward(X,parameters):

    L=len(parameters)//2

    A=X

    caches=[]

    for l in range(1,L):

        A_prev=A

        A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],'relu')

        caches.append(cache)

    A_prev=A

    AL,cache=linear_activation_forward(A_prev,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')

    caches.append(cache)

    return AL,caches

def compute_cost(AL,Y):

    m=Y.shape[1]

    cost=-(np.dot(Y,np.log(AL).T)+np.dot((1-Y),np.log(1-AL).T))/m

    cost=np.squeeze(cost)

    return cost

def relu_backward(dA,activation_cache):

    Z=activation_cache

    return dA*np.where(Z>0,1,0)

    

def sigmoid_backward(dA,activation_cache):

    Z=activation_cache

    z,cache=sigmoid(Z)

    return dA*(z*(1-z))

def linear_backward(dZ,cache):

    A_prev,W,b=cache

    m=A_prev.shape[1]

    

    dW=np.dot(dZ,A_prev.T)/m

    db=np.sum(dZ,axis=1,keepdims=True)/m

    dA_prev=np.dot(W.T,dZ)

    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):

    linear_cache,activation_cache=cache

    if activation=='relu':

        dZ=relu_backward(dA,activation_cache)

        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    elif activation=='sigmoid':

        dZ=sigmoid_backward(dA,activation_cache)

        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    return dA_prev,dW,db

def L_model_backward(AL,Y,caches):

    grads={}

    L=len(caches)

    m=Y.shape[1]

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache=caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L-1)):

        current_cache=caches[l]

        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)]=linear_activation_backward(dAL, current_cache, 'relu')

    return grads

def update_parameters(parameters, grads, learning_rate):

    """

    Update parameters using gradient descent

    

    Arguments:

    parameters -- python dictionary containing your parameters 

    grads -- python dictionary containing your gradients, output of L_model_backward

    

    Returns:

    parameters -- python dictionary containing your updated parameters 

                  parameters["W" + str(l)] = ... 

                  parameters["b" + str(l)] = ...

    """

    

    L = len(parameters) // 2 # number of layers in the neural network



    # Update rule for each parameter. Use a for loop.

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]

        parameters["b" + str(l+1)] = parameters["b"+ str(l+1)]-learning_rate*grads["db"+ str(l+1)]

    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False):#lr was 0.009

    """

    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    

    Arguments:

    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).

    learning_rate -- learning rate of the gradient descent update rule

    num_iterations -- number of iterations of the optimization loop

    print_cost -- if True, it prints the cost every 100 steps

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """



    np.random.seed(1)

    costs = []                         # keep track of cost

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)       

        grads = L_model_backward(AL, Y, caches)        

        parameters = update_parameters(parameters, grads, learning_rate)        

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

        if print_cost and i % 100 == 0:

            costs.append(cost)

            

    # plot the cost

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations (per tens)')

    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

    

    return parameters

def predict(parameters,test_X):

    prediction, Z=L_model_forward(test_X,parameters)

    prediction=np.where(prediction<0.5,0,1)

    return prediction

    



def score(parameters,X,Y):

    prediction=predict(parameters,X)

    preds=(prediction==Y)

    count=0

    for i in range(preds.shape[1]):

        if preds[0,i]==True:

            count=count+1

    return (count*100/float(Y.shape[1]))

layer_dims=(train_x.shape[0],20,10,1)  

parameters = L_layer_model(train_x, train_y, layer_dims, learning_rate =0.001,num_iterations = 50000, print_cost = True)

train_score=score(parameters,train_x,train_y)

print('Accuracy on training data = '+str(round(train_score,2)))

test_score=score(parameters,val_x,val_y)

print('Accuracy on validation data = '+str(round(test_score,2)))

Survived=predict(parameters,test_x)

list1=passengerId[891:1309]

list2=Survived.squeeze().tolist()

output=pd.DataFrame({'PassengerId':list1,

                     'Survived':list2})

output.to_csv('submission.csv',index=False)
