# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.framework import ops



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

        end=round(start+0.2*count[i])

        if(end<len(full_train)):

            for j in range(start,end):

                train.append(full_train[j][:])

            start=start+count[i]

    return train
train=train_temp(full_train)
def bitmap_conversion(data,threshold):

    temp=np.zeros((len(data),785))

    for i in range(len(data)):

            for j in range(1,785):

                if(data[i][j]>=threshold):

                    temp[i][j]=1

                #else:

                   # data[i][j]=0

    return temp.T
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
#threshold=bitmap_threshold(dev,128,5) #113.5

X=bitmap_conversion(train,threshold=113.5)
test_bmp=bitmap_conversion(test,113.5)
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
X_train2=zoning(X,"train")

Y_train2=gety_labels(train)
X_test=zoning(test_bmp,"train")

Y_test=gety_labels(test_bmp)
def create_placeholders(n_x, n_y):

 

    X = tf.placeholder(tf.float32,[n_x,None],name="X")

    Y = tf.placeholder(tf.float32,[n_y,None],name="Y")

    

    

    return X, Y
def initialize_parameters():

    

    

   

    W1 = tf.get_variable("W1",[10,785],initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.get_variable("b1",[10,1],initializer=tf.zeros_initializer())

    W2 = tf.get_variable("W2",[10,10],initializer=tf.contrib.layers.xavier_initializer())

    b2 = tf.get_variable("b2",[10,1],initializer=tf.zeros_initializer())

    W3 = tf.get_variable("W3",[26,10],initializer=tf.contrib.layers.xavier_initializer())

    b3 = tf.get_variable("b3",[26,1],initializer=tf.zeros_initializer())

    



    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2,

                  "W3": W3,

                  "b3": b3}

    

    return parameters

    
def forward_propagation(X, parameters):

    """

    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    

    Arguments:

    X -- input dataset placeholder, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"

                  the shapes are given in initialize_parameters



    Returns:

    Z3 -- the output of the last LINEAR unit

    """

    

    # Retrieve the parameters from the dictionary "parameters" 

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']

    

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:

    Z1 = tf.add(tf.matmul(W1,X),b1)                                             # Z1 = np.dot(W1, X) + b1

    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)

    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, a1) + b2

    A2 = tf.nn.relu(Z2)                                               # A2 = relu(Z2)

    Z3 = tf.add(tf.matmul(W3,A2),b3)                                              # Z3 = np.dot(W3,Z2) + b3

    ### END CODE HERE ###

    

    return Z3
def compute_cost(Z3, Y):

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    

    return cost
def model(X_train, Y_train, X_test, Y_test,

          num_epochs = 1000, print_cost = True):

   

    ops.reset_default_graph()                         #to keep consistent results

    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)

    n_y = Y_train.shape[0]                            # n_y : output size

    costs = []                                        # To keep track of the cost

    

    # Create Placeholders of shape (n_x, n_y)

    X, Y = create_placeholders(n_x, n_y)



    # Initialize parameters

    parameters = initialize_parameters()

    

    # Forward propagation: Build the forward propagation in the tensorflow graph

    Z3 = forward_propagation(X, parameters)

    

    # Cost function: Add cost function to tensorflow graph

    cost = compute_cost(Z3, Y)

    

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.

    learning_rate=tf.placeholder(tf.float32,name="learning_rate")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    

    # Initialize all the variables

    init = tf.global_variables_initializer()

    decay_rate=1

    initial_alpha=3

    # Start the session to compute the tensorflow graph

    with tf.Session() as sess:

        

        # Run the initialization

        sess.run(init)

        

        # Do the training loop

        for epoch in range(num_epochs):

            alpha=(1/(1+(decay_rate*epoch)))*initial_alpha

            _, epoch_cost = sess.run([optimizer,cost],feed_dict={X:X_train,Y:Y_train,learning_rate:alpha})

            epoch_cost/=m

           

            # Print the cost every epoch

            if print_cost == True and epoch % 1000 == 0:

                print(alpha)

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 100 == 0:

                costs.append(epoch_cost)

                

        # plot the cost

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # lets save the parameters in a variable

        parameters = sess.run(parameters)

        print ("Parameters have been trained!")



        # Calculate the correct predictions

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        

        return parameters
X.shape


parameters = model(X, Y_train, test_bmp, Y_test)
parameters = model(X_train, Y_train, X_test, Y_test)