import numpy as np

import time 
def loadData(filename):

    fr = open(filename,'r')

    x,y = [],[]

    for line in fr.readlines():

        curline = line.strip().split(',')

        x.append([int(num) for num in curline[1:]])

        y.append(int(curline[0]))  

    x = np.array(x)

    y = np.array(y)

    return x,y
def get_Lapsmoothing(x_train,y_train):

    label_num = 10

    dim = 784

    colorway = 256

    

    P_x_y = np.zeros((label_num,dim,colorway))

    P_y = np.zeros((10))

    sum_label = np.zeros((10))

    

    for i in range(label_num):

        sum_label[i] = np.sum(y_train == i)

        P_y[i] = (sum_label[i] + 1) / (y_train.shape[0] + label_num)

    P_y = np.log(P_y)

    

    for i in range(label_num):

        label_i = x_train[y_train == i]

        for j in range(dim):

            for k in range(colorway):

                P_x_y[i,j,k] = (np.sum(label_i[:,j] == k) + 1) / (sum_label[i] + colorway)                

    P_x_y = np.log(P_x_y)



    return P_x_y,P_y

def predict(P_x_y,P_y,x):

    label_num = 10

    dim = 784

    colorway = 256

    p = np.zeros((10))

    for i in range(label_num):

        p[i] = P_y[i]

        for j in range(dim):

            p[i] += P_x_y[i][j][x[j]]

    return np.argmax(p)
def test(x_val,y_val,P_x_y,P_y):

    start = time.time()    

    correct = 0

    size = x_val.shape[0]

    for i in range(size):

        pred = predict(P_x_y,P_y,x_val[i])

        if pred == y_val[i]: correct += 1    

    print("Time of training consumes:{:.2f} Accuracy is:{:.2f}".format(time.time() - start , correct/size))

    return correct/size
x_train,y_train = loadData('/kaggle/input/mnist-percetron/mnist_train.csv')

x_val,y_val = loadData('/kaggle/input/mnist-percetron/mnist_test.csv')

print("start to train...")

P_x_y,P_y = get_Lapsmoothing(x_train,y_train)

print("start to test...")

acc = test(x_val,y_val,P_x_y,P_y)