import numpy as np

import time
def loadData(filename):

    fr = open(filename,'r')

    x,y = [],[]

    for line in fr.readlines():

        curline = line.strip().split(',')

        x.append([int(num) for num in curline[1:]])

        y.append(int(curline[0]))

    return x,y
def cal_dict(x1,x2):

    assert len(x1) == len(x2)

    return np.sqrt(np.sum(np.square(x1 - x2)))
def get_topK(x_train,y_train,x_test,K):

    dict = []

    for x in x_train:

        dict.append(cal_dict(x,x_test))

    idxs = np.array(dict).argsort()[:K]

    labels = []

    for idx in idxs:

        labels.append(y_train[idx])

    return labels
def test(x_train,y_train,x_val,y_val,K):

    start = time.time()

    x_train_mat = np.mat(x_train)

    x_val_mat = np.mat(x_val)

    correct = 0

    for i in range(len(x_val[:100])):

        pred_labels = get_topK(x_train_mat,y_train,x_val_mat[i],K)

        pred_label = max(pred_labels,key = pred_labels.count)

        if pred_label == y_val[i]:

            correct += 1

    print("Time of training consumes:{:.2f} Accuracy is:{:.2f}".format(time.time() - start , correct / len(x_val[:100])))

    return correct / len(x_val[:100])
x_train,y_train = loadData('/kaggle/input/mnist-percetron/mnist_train.csv')

x_val,y_val = loadData('/kaggle/input/mnist-percetron/mnist_test.csv')

acc = test(x_train,y_train,x_val,y_val,25)