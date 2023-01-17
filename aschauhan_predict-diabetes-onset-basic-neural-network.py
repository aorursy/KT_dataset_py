# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from scipy.optimize import minimize

import scipy.optimize as opt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import the data csv into dataset and have a look

diab = pd.read_csv('../input/diabetes.csv')

diab.head()
# rename column names for ease and look at basic data description

diab.columns = ['pr', 'gl', 'bp', 'st', 'ins', 'bmi', 'dpf', 'age', 'out']

diab.describe()
# replacing zeros with mean values for some columns

diab.loc[diab['gl']==0, 'gl'] = np.ceil(np.mean(diab['gl']))

diab.loc[diab['bp']==0, 'bp'] = np.ceil(np.mean(diab['bp']))

diab.loc[diab['st']==0, 'st'] = np.ceil(np.mean(diab['st']))

diab.loc[diab['ins']==0, 'ins'] = np.ceil(np.mean(diab['ins']))

diab.loc[diab['bmi']==0, 'bmi'] = np.ceil(np.mean(diab['bmi']))

diab.describe()
# for training a basic neural network we need to add a bias variable to the dataset

# adding one additional column with value 1 for the bias variable

diab['x0'] = 1

cols = diab.columns.tolist()

cols = cols[-1:] + cols[:-1]

diab = diab[cols]

diab.head()
# create test and training datasets

diab_train = diab.sample(frac = 0.8, random_state = 1234)

diab_test = diab.drop(diab_train.index)

# sort the training and test dataset by index for ease in looping

diab_train = diab_train.reset_index(drop=True)

diab_test = diab_test.reset_index(drop=True)

print(diab_train.shape, diab_test.shape)

diab_train.head()
# some key functions to be used in setting up the neural network prediction

# sigmoid function

def sigmoid(z):

    return (1.0/(1.0 + np.exp(-1.0*z)))



# sigmoid gradient function

def sigmoid_grad(z):

    return np.multiply(sigmoid(z), 1- sigmoid(z))



# define the cost function for the neural network for a given input layer units, hidden layer units

# and regularization factor lambda



# training neural network with only 1 hidden layer



# theta (a combination of theta1 and theta2) will be passed as an unrolled flattened array

# x and y will be matrices and x already has ones added to it



def nncostfunc(theta, hidden_layer, x, y, lmbda=0):

    m = x.shape[0]

    n = x.shape[1]

    theta1 = theta[0:(hidden_layer*n)].reshape(hidden_layer, n)

    theta2 = theta[(hidden_layer*n):theta.size].reshape(1, hidden_layer+1)

    

    cost = 0

    theta1_g = np.zeros([theta1.shape[0], theta1.shape[1]])

    theta2_g = np.zeros([theta2.shape[0], theta2.shape[1]])

    

    for i in range(0,m):

        xtemp = x[i,:].reshape(1,n)

        # hidden layer activation values

        a2 = sigmoid(np.dot(theta1, xtemp.T))

        a2 = np.vstack([1, a2])

        

        # output layer activation

        a3 = sigmoid(np.dot(theta2, a2))

        

        # cost for the ith example

        ytemp = y[i,:].reshape(1,1)

        cost+= (-1.0/m) * (np.dot(ytemp.T, np.log(a3)) + np.dot((1-ytemp).T, np.log(1-a3)))

        

        # error terms and gradient of theta by backpropagation

        delta_3 = a3 - ytemp

        

        delta_2 = np.multiply(np.dot(theta2.T, delta_3), np.multiply(a2, 1-a2))

        delta_2 = delta_2[1:delta_2.shape[0],:]

        

        theta1_g = theta1_g + np.dot(delta_2, xtemp)

        theta2_g = theta2_g + np.dot(delta_3, a2.T)

    

    # apply regularization - to be added later

    

    cost+= (lmbda/(2*m))*(np.sum(np.sum(np.square(theta1[:,1:theta1.shape[1]]))) + np.sum(np.sum(np.square(theta2[:,1:theta2.shape[1]]))))

    

    theta1_g[:,0] = (1.0/m)* theta1_g[:,0]

    theta1_g[:,1:theta1_g.shape[1]] = ((1.0/m)*theta1_g[:,1:theta1_g.shape[1]]) + ((lmbda/m)*theta1[:,1:theta1.shape[1]])

    

    theta2_g[:,0] = (1.0/m)* theta2_g[:,0]

    theta2_g[:,1:theta2_g.shape[1]] = ((1.0/m)*theta2_g[:,1:theta2_g.shape[1]]) + ((lmbda/m)*theta2[:,1:theta2.shape[1]])

    

    # flatten the gradient as the optimization function only uses flattened array for the grad

    theta1_g = np.array(theta1_g.ravel())

    theta2_g = np.array(theta2_g.ravel())

    grad = np.append(theta1_g, theta2_g)

    

    return cost, grad





# random initialization of theta function

def rand_init(lin, lout):

    w = np.zeros([lout, lin + 1])

    epsilon = 0.12

    

    w = (np.random.rand(lout, lin + 1)*2*epsilon) - epsilon

    

    return w



# function to predict the output from given x values, x should already have first column as 1s

def predict(theta1, theta2, x):

    h1 = sigmoid(np.dot(x, theta1.T))

    h1 = np.hstack((np.ones([x.shape[0],1]),h1))

    

    h2 = sigmoid(np.dot(h1, theta2.T))

    

    pred = np.zeros((x.shape[0],1))

    for i in range(0,x.shape[0]):

        if h2[i,0] > 0.5:

            pred[i,0] = 1

        else:

            pred[i,0] = 0

            

    return pred
# minimize the cost function to get thetas for the training dataset - train the neural network!



# we also want to plot the testing and training accuracy with the hidden layer size to see 

# how it impacts the model



# transforming the input training dataset to matrix format for the ease of vectorization

x = diab_train[diab_train.columns[[0,1,2,3,4,5,6,7,8]]].as_matrix()

y = diab_train[diab_train.columns[[9]]].as_matrix()



def acc_nn(hl_min, hl_max, il):

    

    columns = ['hl', 'test_acc', 'train_acc']

    plot_data = pd.DataFrame(index=range(hl_min,hl_max+1), columns=columns)

    

    for hl in range(hl_min,hl_max+1):

    

        # initialize random values to theta1 and theta2

        t1_in = rand_init(il,hl)

        t2_in = rand_init(hl,1)

        t1_in = np.array(t1_in.ravel())

        t2_in = np.array(t2_in.ravel())

        t_final = np.append(t1_in, t2_in)



        # minimize the cost obtained from the cost function for given x and y and the theta values initialized above

        result = opt.fmin_tnc(func=nncostfunc, x0=t_final, args=(hl, x, y, 0.2))

        print("cost for nn with hidden layers ", hl, " is :", nncostfunc(result[0], hl, x, y, 0.2)[0])



        # predict the output for training and testing datasets

        x_train = diab_train[diab_train.columns[[0,1,2,3,4,5,6,7,8]]].as_matrix()

        y_train = diab_train[diab_train.columns[[9]]].as_matrix()



        x_test = diab_test[diab_test.columns[[0,1,2,3,4,5,6,7,8]]].as_matrix()

        y_test = diab_test[diab_test.columns[[9]]].as_matrix()



        theta1 = np.matrix(result[0][0:(hl*(il+1))].reshape(hl, il+1))

        theta2 = np.matrix(result[0][(hl*(il+1)):result[0].size].reshape(1, hl+1))



        pred_train = predict(theta1, theta2, x_train)

        pred_test = predict(theta1, theta2, x_test)



        train_acc = 0

        test_acc = 0



        for i in range(0, pred_train.shape[0]):

            if pred_train[i,0] == y_train[i,0]:

                train_acc+=1



        for i in range(0, pred_test.shape[0]):

            if pred_test[i,0] == y_test[i,0]:

                test_acc+=1 

                

        # append info to the plot_Data data

        plot_data.loc[hl,'hl'] = hl

        plot_data.loc[hl,'test_acc'] = np.ceil(100.0*test_acc/y_test.shape[0])

        plot_data.loc[hl,'train_acc'] = np.ceil(100.0*train_acc/y_train.shape[0])

        

    # plot the accuracies vs hidden layer size

    fig = plt.figure(figsize=(15, 8))

    ax = fig.add_subplot(111)

    line, = ax.plot(plot_data['hl'], plot_data['test_acc'], lw=2, label = "test accuracy")

    for i, txt in enumerate(plot_data['test_acc']):

        ax.annotate(txt, xy=(plot_data.iloc[i, 0],plot_data.iloc[i,1]))

    

    ax1 = fig.add_subplot(111)

    line, = ax1.plot(plot_data['hl'], plot_data['train_acc'], lw=2, label = "training accuracy")

    for i, txt in enumerate(plot_data['train_acc']):

        ax1.annotate(txt, xy=(plot_data.iloc[i, 0],plot_data.iloc[i,2]))

    #plot_data.plot(x='k')

    plt.legend(loc="upper left")

    plt.show()

    

acc_nn(8, 15, 8) 
y_train = diab_train[diab_train.columns[[9]]]

y_test = diab_test[diab_test.columns[[9]]]

print("Baseline Training and Test accuracies : ", np.ceil(100.0*y_train[y_train['out']==0].shape[0]/y_train.shape[0]), np.ceil(100.0*y_test[y_test['out']==0].shape[0]/y_test.shape[0]))