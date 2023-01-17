import numpy as np

import pandas as pd

import random as rd

import struct

from PIL import Image
def readMNISTdata(ubyte_file, status):

    with open(ubyte_file, 'rb') as f:

        buf = f.read()

    offset = 0



    if status == 'data':

        magic, imageNum, rows, cols = struct.unpack_from('>IIII', buf, offset)

        offset += struct.calcsize('>IIII')

        imageSize = rows * cols

        images = np.empty((imageNum,imageSize))

        fmt = '>' + str(imageSize) + 'B'

        for i in range(imageNum):

            images[i] = np.array(struct.unpack_from(fmt, buf, offset))

            offset += struct.calcsize(fmt)

        return images

    elif status == 'label':

        magic, LabelNum = struct.unpack_from('>II', buf, offset)

        offset += struct.calcsize('>II')

        Labels = np.zeros((LabelNum))

        for i in range(LabelNum):

            Labels[i] = np.array(struct.unpack_from('>B', buf, offset))

            offset += struct.calcsize('>B')

        return Labels

    else: return -1
train_images = readMNISTdata('../input/train-images-idx3-ubyte','data')

train_labels = readMNISTdata('../input/train-labels-idx1-ubyte','label')

test_images =  readMNISTdata('../input/t10k-images-idx3-ubyte','data')

test_labels = readMNISTdata('../input/t10k-labels-idx1-ubyte','label')
train_ind = np.arange(0,60000)

rd.shuffle(train_ind)

test_ind = np.arange(0,10000)

rd.shuffle(test_ind)

train_size = 500

test_size = 500

train_images = train_images[train_ind[0:train_size],...]

train_labels = train_labels[train_ind[0:train_size]]

test_images = test_images[test_ind[0:test_size],...]

test_labels = test_labels[test_ind[0:test_size]]
def Euc_distance(x,y):

    dist = np.linalg.norm(x-y)

    return dist
def Man_distance(x,y):

    dist = np.linalg.norm(x-y, ord=1)

    return dist
def Cos_distance(x,y):

    dist = np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)

    return dist
def KNN(train_images, train_labels, test_images, test_labels):

    train_size = train_labels.size

    test_size = test_labels.size

    count = 0

    distance = np.zeros(train_size)

    for i in range(test_size):

        for j in range(train_size):

            distance[j] = Euc_distance(test_images[i,...],train_images[j,...])

        index = np.argsort(distance)

        predict = train_labels[index[0]]

        if predict == test_labels[i]:

            count = count + 1

    acc = count / test_size

    return acc
def KNN_tangent(train_images, train_labels, test_images, test_labels):

    train_size = train_labels.size

    test_size = test_labels.size

    count = 0

    learning_rate = 6e-7

    training_epochs = 50

    for i in range(test_size):

        # Unit vector for each test point

        #Tr=np.zeros((784,3))

        pattern = test_images[i,...].reshape((28,28))

        pattern_x = np.hstack((np.zeros((28,1)),pattern[...,0:27]))

        pattern_y = np.vstack((pattern[1:,...],np.zeros((1,28))))

        #Rotate the image by 3 degrees

        img = pattern.astype(np.uint8)

        img = Image.fromarray(img)

        pattern_r = np.array(img.rotate(3))

        Tr = np.hstack((pattern_x.reshape((784,1))-test_images[i,...].reshape((784,1)), \

                              pattern_y.reshape((784,1))-test_images[i,...].reshape((784,1)), \

                              pattern_r.reshape((784,1))-test_images[i,...].reshape((784,1))))

        # Tangent distance

        distances = np.zeros(train_size)

        for j in range(train_size):

            alpha = np.zeros((3,1))

            # Gradient descent

            for ite in range(training_epochs):

                alpha = alpha - learning_rate*np.matmul(Tr.T,(test_images[i,...].reshape((784,1))+np.matmul(Tr,alpha)-train_images[j,...].reshape((784,1))))

            distances[j] = Euc_distance(test_images[i,...].T+np.matmul(Tr,alpha), train_images[j,...].T)

        index = np.argsort(distances)

        predict = train_labels[index[0]]

        if predict == test_labels[i]:

            count = count + 1

    acc = count / test_size

    return acc
acc = KNN(train_images, train_labels, test_images, test_labels)

print(acc)
acc = KNN_tangent(train_images,train_labels,test_images,test_labels)

print(acc)