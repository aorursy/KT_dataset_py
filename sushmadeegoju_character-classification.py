# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/A_Z Handwritten Data.csv")

data=np.asarray(data)
def splitting(temp):

    count=[0]*26

    for i in range(len(temp)):

        count[temp[i][0]]+=1

    train=[]

    dev=[]

    test=[]

    start=0

    for i in range(len(count)):

        train1=round(0.8*count[i])

        val1=test1=round(0.1*count[i])

        end=start+train1+val1+test1

        if(end<=len(temp)+2):

            for j in range(start,end):

                if(j<(start+train1)):

                    train.append(temp[j][:])

        

                elif(j<(start+train1+val1)):

                    dev.append(temp[j][:])

               

                elif(j<(start+train1+val1+test1)):

                    if(j<372450):

                        test.append(temp[j][:])

                

            start=end

    return train,dev,test
full_train,dev,test=splitting(data)
def train_temp(full_train):

    count=[0]*26

    for i in range(len(full_train)):

        count[full_train[i][0]]+=1

    train=[]

    k=0

    end=0

    start=0

    for i in range(len(count)):

        end=round(start+0.1*count[i])

        if(end<len(full_train)):

            for j in range(start,end):

                train.append(full_train[j][:])

            start=start+count[i]

    return train

        
train=train_temp(full_train)
def bitmap_conversion(data,threshold):

    for i in range(len(data)):

            for j in range(1,785):

                if(data[i][j]>=threshold):

                    data[i][j]=1

                else:

                    data[i][j]=0

    return data
def bitmap_threshold(data,threshold,iterations):

    temp1=[]

    temp2=[]

    avg=threshold

    while(iterations>0):

        for i in range(len(data)):

            for j in range(len(data[0])):

                if(data[i][j]>=avg):

                    temp1.append(data[i][j])

                else:

                    temp2.append(data[i][j])

        avg1=average(temp1)

        avg2=average(temp2)

        avg=(avg1+avg2)/2

        iterations-=1

    return avg
def average(temp_list):

    return round(np.sum(temp_list)/len(temp_list))
threshold=bitmap_threshold(dev,128,5) #113

X=bitmap_conversion(train,threshold)
def vec_to_matrix(data,dset="train"):

    temp=np.zeros((28,28))

    count=0

    if(dset=="train"):

        k=1

        for i in range(1,785,28):

            for j in range(0,28):

                temp[count][j]=data[k]

                k+=1

            count+=1

    elif(dset=="test"):

        k=1

        for i in range(0,784,28):

            for j in range(0,28):

                temp[count][j]=data[k]

                k+=1

            count+=1

    return temp
def add(temp,r,c):

    res=0

    for i in range(r,r+7):

        for j in range(c,c+7):

            res+=temp[i][j]

    return res
def zoning(data,dset="train"):

    features=np.zeros((len(data),16))

    if(dset=="train"):

        for i in range(len(data)):

            temp=vec_to_matrix(data[i],dset)

            r=0

            count=0

            for j in range(0,4):

                c=0

                for k in range(0,4):

                    features[i][count]=add(temp,r,c)

                    count+=1

                    c+=7

                r+=7

            features[i]/=13

    elif(dset=="test"):

        for i in range(len(data)):

            print(data[i].shape)

            temp=vec_to_matrix(data[i],dset)

            r=0

            count=0

            for j in range(0,4):

                c=0

                for k in range(0,4):

                    features[i][count]=add(temp,r,c)

                    count+=1

                    c+=7

                r+=7

            features[i]/=13

    return features.T
def gety_labels(Y):

    temp=np.zeros((len(Y),26))

    for i in range(len(Y)):

        k=Y[i][0]

        temp[i][k]=1

    return temp.T
X_train=zoning(X,"train")

Y_train=gety_labels(train)
test_bmp=bitmap_conversion(test,threshold)

X_test=zoning(test_bmp,"train")

Y_test=gety_labels(test_bmp)
def activation(z,act="sigmoid"):

    if(act=="sigmoid"):

        return (1/(1+np.exp(-z)))

    elif(act=="relu"):

        return np.maximum((0.01*z),z)

    elif(act=="tanh"):

        return np.tanh(z)

    elif(act=="softmax"):

        t=np.exp(z)

        a=np.divide(t,np.sum(t)+0.0001)

        return a
def layer_sizes(X,Y):

    n_x=X.shape[0]

    n_h=10

    n_y=Y.shape[0]

    return (n_x,n_h,n_y)
layer_sizes(X_train,Y_train)
def initialize_parameters(n_x,n_h,n_y):

    W1=np.random.randn(n_h,n_x)*0.01

    b1=np.zeros((n_h,1))

    W2=np.random.randn(n_y,n_h)*0.01

    b2=np.zeros((n_y,1))

    

    parameters={"W1":W1,

               "b1":b1,

               "W2":W2,

               "b2":b2}

    return parameters
(n_x,n_h,n_y)=layer_sizes(X_train,Y_train)

parameters=initialize_parameters(n_x,n_h,n_y)
def forward_prop(X,parameters):

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    

    Z1 = np.dot(W1,X)+b1

    A1 = activation(Z1,"relu")

    Z2 = np.dot(W2,A1)+b2

    A2 = activation(Z2,"sigmoid")

    

    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache
A2,cache=forward_prop(X_train,parameters)
def max_index(arr):

    maxm=arr[0]

    for i in range(len(arr)):

        if(maxm<=arr[i]):

            maxm=arr[i]

            index=i

    return index
def accuracy(A2,Y_train):

    y_hat=A2.T

    res=0

    for i in range(len(Y_train[0])):

        maxm=max_index(y_hat[i])

        if(Y_train[maxm][i]==1):

            res+=1

    return (res/len(Y_train[0]))
def elt_mult(A1,A2):

    A=A1.T

    Y=A2.T

    res=0

    for i in range(len(A)):

        k=np.multiply(A[i],Y[i])

        res+=np.sum(k)

    return res
def compute_cost(A2,Y,parameters):

    m = Y.shape[1] 

    A1=np.log(A2)

    logprobs = elt_mult(A1,Y)

    cost = -(logprobs/m)

    return cost
compute_cost(A2,Y_train,parameters)
def compute_cost2(A2,Y,parameters):

    m = Y.shape[1] 

    logprobs = np.sum(np.sum(np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y))))

    logprobs = np.squeeze(logprobs) 

    cost = -(logprobs/m)

    return cost
compute_cost2(A2,Y_train,parameters)
def g1prime(A,Z):

    return (np.divide(A,Z))
def back_prop(parameters,cache,Y,X):

    

    m=Y.shape[1]

    W1=parameters["W1"]

    W2=parameters["W2"]

    A1=cache["A1"]

    A2=cache["A2"]

    Z1=cache["Z1"]

    

    dZ2=A2-Y

    dW2=(1/m)*np.dot(dZ2,A1.T)

    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)

    dZ1=np.multiply(np.dot(W2.T,dZ2),g1prime(A1,Z1))

    dW1=(1/m)*np.dot(dZ1,X.T)

    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)

    

    grads={"dW1":dW1,

          "dW2":dW2,

          "db1":db1,

          "db2":db2}

    return grads
def update_parameters(parameters,grads,alpha=1.2):

    W1=parameters["W1"]

    W2=parameters["W2"]

    b1=parameters["b1"]

    b2=parameters["b2"]

    

    dW1=grads["dW1"]

    dW2=grads["dW2"]

    db1=grads["db1"]

    db2=grads["db2"]

    

    W1=W1-(alpha*dW1)

    W2=W2-(alpha*dW2)

    b1=b1-(alpha*db1)

    b2=b2-(alpha*db2)

    

    parameters={"W1":W1,

               "b1":b1,

               "W2":W2,

               "b2":b2}

    

    return parameters
def nn(X_train,Y_train,n_h,iterations=1000):

    n_x,n_h,n_y=layer_sizes(X_train,Y_train)

    parameters=initialize_parameters(n_x,n_h,n_y)

    costs=[]

    decay_rate=1.5

    initial_alpha=1

    for i in range(iterations):

        A2,cache=forward_prop(X_train,parameters)

        cost=compute_cost(A2,Y_train,parameters)

        grads=back_prop(parameters,cache,Y_train,X_train)

        learning_rate=(1/(1+(decay_rate*i)))*initial_alpha

        parameters=update_parameters(parameters,grads,learning_rate)

        if(i%100==0):

            print(cost)

            costs.append(cost)

    return parameters,costs
parameters,costs=nn(X_train,Y_train,10,1000)
import matplotlib.pyplot as plt

x=[1,2,3,4,5,6,7,8,9,10]

plt.plot(x, costs)

plt.show()
A2,cache=forward_prop(X_train,parameters)

accuracy_train=accuracy(A2,Y_train)
accuracy_train
A2_test,cache_test=forward_prop(X_test,parameters)

accuracy_test=accuracy(A2_test,Y_test)
accuracy_test