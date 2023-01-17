import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



%matplotlib inline
#LOADING THE DATA From data.csv

df = pd.read_csv('../input/data.csv')

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})



#Complete data set:

X = df[df.columns[2:32]]

Y = df['diagnosis']

Y = Y.values.reshape(Y.shape[0],1)





#train set (80%):

train_X = X.loc[0:454,X.columns[0:]]

train_Y = Y[0:455]



#test set (20%):

test_X = X.loc[0:143,X.columns[0:]]

test_Y = Y[0:144]

#training set:



mean = train_X.mean()

std_error = train_X.std()

train_X = (train_X - mean)/std_error



#test set:

mean = test_X.mean()

std_error = test_X.std()

test_X = (test_X - mean)/std_error
print("Shape of train_X",train_X.shape)

print("Shape of test_X",test_X.shape)

print("Shape of train_Y",train_Y.shape)

print("Shape of test_Y",test_Y.shape)
def sigmoid(z):

    return 1/(1+np.exp(-z))
sigmoid(np.array([1,2,3,3]))
def random_init(dim):

    w = np.zeros((dim,1))

    b = 0

    

    return w,b

(random_init(train_X.shape[1]))
def propo(w,b,X,Y):

    

    m = X.shape[0]

    

    #forward propogation

    z = np.dot(X,w) + b

    a = sigmoid(z)

    cost = -np.sum(Y*np.log(a) - (1-Y)*np.log(1-a))/m

    

    

    #backpropogation:

    dz = a-Y

    dw  = np.dot(np.transpose(X),dz)/m

    db = np.sum(dz)/m

    

    grad = {

        "dw":dw,

        "db":db

    }

    return grad,cost
def optim(w,b,X,Y,learning_rate,num_iteration):

    costs = []

    

    for i in range(num_iteration):

        grads, cost=propo(w,b,X,Y)

        

        dw = grads["dw"]

        db = grads["db"]

        

        #updating w and b

        w  = w - learning_rate*dw

        b  = b - learning_rate*db

          

        if(i%100==0):

            costs.append(cost)

        

    params= {

        "w":w,

        "b":b

    }

    grads = {

        "dw":dw,

        "db":db

    }

    return params,grads,costs
#random init of w,b

w,b = random_init(train_X.shape[1])



#forward, backward & grad. descent:



params,grads,costs = optim(w,b,train_X,train_Y,0.01,2000)



print(params)

print(grads)

print(costs)
# plt.plot(cost_all,range(len(cost_all)))

costs = np.squeeze(costs)

plt.plot(costs)

plt.xlabel('No. of iteration')

plt.ylabel('Cost')

plt.show()
def predict(w,b,X):

    a = sigmoid(np.dot(X,w) + b)

    return a
def oneORzero(x):

    if(x>=0.5):

        return 1

    elif(x<0.5):

        return 0
# Accuracy for training set:

temp = predict(params["w"],params["b"],train_X)

train_prediction = np.array(list(map(oneORzero,temp)))

train_prediction = train_prediction.reshape((train_prediction.shape[0],1))



# Accuracy for test set:

temp = predict(params["w"],params["b"],test_X)

test_prediction = np.array(list(map(oneORzero,temp)))

test_prediction = test_prediction.reshape((test_prediction.shape[0],1))



print("Training set accuracy = ",(100 - np.mean(np.abs(train_prediction - train_Y))*100))

print("Test set accuracy = ",(100 - np.mean(np.abs(test_prediction - test_Y))*100))