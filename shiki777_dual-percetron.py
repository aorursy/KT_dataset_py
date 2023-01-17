import pandas as pd

import numpy as np

import time

from sklearn.metrics import f1_score,accuracy_score,recall_score
def load_Data(filename):

    fr = open(filename,'r')

    x,y = [],[]

    for line in fr.readlines():

        curline = line.strip().split(',')

        x.append([int(num) / 255 for num in curline[1:]])

        y.append(1 if int(curline[0]) <= 4 else -1)

        if len(x) == 100:

            break

    return x,y 
def percetron(x_train,y_train):

    data_mat = np.mat(x_train,dtype = np.float32)

    label_mat = np.mat(y_train).T



    len_data_mat = data_mat.shape[0]

    

    b = 0

    lr = 1e-4

    alpha = np.zeros((len_data_mat,))

    G_matrix = np.dot(data_mat,data_mat.T)



    print("start to train...")

    for epoch in range(30):

        start_train = time.time()

        for i in range(len_data_mat):

            x_i = data_mat[i]

            y_i = label_mat[i]

            sum = 0

            

            for j in range(len_data_mat):

                sum += alpha[j] * label_mat[j] * G_matrix[i,j]

                

            if (sum + b) * y_i <= 0:

                alpha[i] += lr

                b += y_i*lr

                

        print('Time of epoch {} consume:{:.2f} seconds:'.format(epoch + 1,time.time() - start_train))

    return G_matrix,alpha,b
def test(x,y,alpha,b,G_matrix):

    data_mat = np.mat(x,dtype = np.float32)

    label_mat = np.mat(y).T

    

    len_data_mat = len(data_mat)

    correct = 0

    

    for i in range(len_data_mat):



        x_i = data_mat[i]

        y_i = label_mat[i]

        

        sum = 0

        for j in range(len_data_mat):

            sum += alpha[j] * label_mat[j] * G_matrix[i,j]

        if y_i * (sum + b) > 0:

            correct += 1



    return correct/len_data_mat
x_train,y_train = load_Data('/kaggle/input/mnist_train.csv')

x_val,y_val = load_Data('/kaggle/input/mnist_test.csv')

print("x_train_length:",len(x_train),"x_val_length:",len(x_val))

print("y_train_length:",len(y_train),"y_val_length:",len(y_val))
G,alpha,b = percetron(x_train,y_train)
acc_train = test(x_train,y_train,alpha,b,G)

print("accuracy:",acc_train)
acc_val = test(x_val,y_val,alpha,b,G)

print("accuracy:",acc_val)