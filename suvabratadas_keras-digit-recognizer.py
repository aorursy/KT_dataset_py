# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# from __future__ import print_function

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from keras.models import Sequential, load_model

from keras import Input

from keras.layers import Reshape, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from keras.utils import np_utils



import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def loadtraindata(path, normalize=1.0, trainFraction=0.70):

    # read table of features and label. label is first column of data

    mydataframe = pd.read_csv(path)

    (M, N) = mydataframe.shape



    M_train = int(M*trainFraction)

    M_validation = M - M_train

    print("{:.2f} %".format(100.0*float(M_train/M)))

    

    trainDF      = mydataframe.sample(M_train)

    validationDF = mydataframe.drop(trainDF.index)

    

    # convert dataframe to Matrix

    X_train = trainDF.values[:,1:N]/normalize

    y_train = trainDF.values[:,0]

    

    X_validation = validationDF.values[:,1:N]/normalize

    y_validation = validationDF.values[:,0]

    

    return X_train, y_train, X_validation, y_validation
path = "/kaggle/input/digit-recognizer/train.csv"

normalize    = 255

(X_train, y_train, X_test, y_test) = loadtraindata(path, normalize, 0.90)



y_train = np_utils.to_categorical(y_train)

y_test  = np_utils.to_categorical(y_test)



image_width  = 28

image_height = 28

first_layer_kernel_width  = 3

first_layer_kernel_height = 3

first_layer_filters = 60



second_layer_kernel_width  = 3

second_layer_kernel_height = 3

second_layer_filters = 120



third_layer_kernel_width  = 3

third_layer_kernel_height = 3

third_layer_filters = 240



first_dense_layer_size    = 300

second_dense_layer_size   = 150

third_dense_layer_size    =  75

nLabels                   =  10



Epochs = 200



model_CNN = Sequential()

model_CNN.add(Reshape((image_width, image_height, 1), input_shape=(X_train.shape[1],)))

model_CNN.add(Conv2D(filters=first_layer_filters, 

                     kernel_size=(first_layer_kernel_width, first_layer_kernel_height),

                     padding="same",

                     input_shape=(image_width,image_height,1), activation='relu'))

model_CNN.add(MaxPooling2D(pool_size=(2,2)))

model_CNN.add(Dropout(0.40))

model_CNN.add(Conv2D(filters=second_layer_filters, 

                     kernel_size=(second_layer_kernel_width, second_layer_kernel_height), 

#                      padding="same",

                     input_shape=(int(image_width/2),int(image_height/2),1), activation='relu'))

model_CNN.add(MaxPooling2D(pool_size=(2,2)))

model_CNN.add(Dropout(0.40))

model_CNN.add(Conv2D(filters=third_layer_filters, 

                     kernel_size=(third_layer_kernel_width, third_layer_kernel_height), 

                     padding="same",

                     input_shape=(int(image_width/4),int(image_height/4),1), activation='relu'))

model_CNN.add(MaxPooling2D(pool_size=(2,2)))

model_CNN.add(Flatten())

model_CNN.add(Dropout(0.40))

model_CNN.add(Dense(first_dense_layer_size, activation="relu"))

model_CNN.add(Dropout(0.40))

model_CNN.add(Dense(second_dense_layer_size, activation="relu"))

model_CNN.add(Dropout(0.40))

model_CNN.add(Dense(third_dense_layer_size, activation="relu"))

model_CNN.add(Dropout(0.40))

model_CNN.add(Dense(nLabels, activation="softmax"))

model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model_CNN.summary()

history_CNN = model_CNN.fit(X_train, y_train, epochs=Epochs, validation_data=(X_test, y_test))

model_CNN.save('Convolution_model')
fig, ax = plt.subplots(1,2,figsize=(14, 4.75))



ax[0].plot(history_CNN.history['accuracy'], 'b')

ax[0].plot(history_CNN.history['val_accuracy'], 'r')

ax[0].set_title('Model Accuracy CNN', fontsize=16)

ax[0].set_xlabel("Epoch", fontsize=14)

ax[0].set_ylabel("Accuracy", fontsize=14)

ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

ax[0].legend(['train', 'validation'], loc='lower right')



ax[1].plot(history_CNN.history['loss'], 'b')

ax[1].plot(history_CNN.history['val_loss'], 'r')

ax[1].set_title('Model Loss CNN', fontsize=16)

ax[1].set_xlabel("Epoch", fontsize=14)

ax[1].set_ylabel("Loss", fontsize=14)

ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

ax[1].legend(['train', 'validation'], loc='upper right')



fig.show()
def loadtestdata(path, normalize=1.0):

    # read table of features. No label in data

    mydataframe = pd.read_csv(path)

    (M, N) = mydataframe.shape



    # convert dataframe to Matrix

    X_test = mydataframe.values[:,0:N]/normalize

    

    return X_test, M, N
path = "/kaggle/input/digit-recognizer/test.csv"

normalize    = 255

X_test_data, M, N = loadtestdata(path, normalize)

print(X_test_data.shape)

prediction = np.zeros((M,1))



predict_CNN = model_CNN.predict(X_test_data)

for ii in range(M):

    prediction[ii,0] = np.float(np.argmax(predict_CNN[ii,:]))

print(prediction.shape)

    

h = open('Prediction.csv', 'w')

h.write('ImageId,Label\n')

for i in range(M):

    h.write('{},{}\n'.format(i+1,np.int(prediction[i])))

h.close()
def visualize(path, row, Nshow, normalize, iLabel=0, iTrans=1):    

    A, M, N = loadtestdata(path, normalize)

    if (iLabel == 1):

        w = np.sqrt(N-1)

    else:

        w = np.sqrt(N)



    iw = np.int_(w)

    print('M = {:4d} N = {:4d} w = {:12.4f}'.format(M,N,w))

    

    show = np.sqrt(Nshow)

    ishow = np.int_(show)

    while (np.mod(Nshow,ishow) > 0):

        ishow += 1

        

    if (iLabel == 0):

        prediction = np.zeros((Nshow,1))

        X_test_data = A[row-1:row-1+Nshow,:]

        predict_CNN = model_CNN.predict(X_test_data)

        for ii in range(Nshow):

            prediction[ii,0] = np.float(np.argmax(predict_CNN[ii,:]))        

       

    jshow = np.int_(Nshow/ishow)

    print('ishow = {} jshow = {} Nshow = {}'.format(ishow,jshow,Nshow))

    

    mat = np.zeros((iw*ishow,iw*jshow))



    if (iLabel == 1):

        print('Labels are:')

    else:

        print('Predictions are:')



    for l in range(jshow):

        for k in range(ishow):

            irow = row + ishow*l + k

            istart = k*iw

            jstart = l*iw

            for j in range(iw):

                for i in range(iw):

                    col = iw*j + i

                    if (iLabel == 1):

                        mat[i+istart,j+jstart] = A[irow-1,col+1]

                    else:

                        mat[i+istart,j+jstart] = A[irow-1,col]

                        

            if (iLabel == 1):

                if (k < ishow-1): print('{:4d}'.format(int(A[irow-1,0])),end="")

                else:             print('{:4d}'.format(int(A[irow-1,0])))

            else:

                if (k < ishow-1): print('{:4d}'.format(int(prediction[irow-row,0])),end="")

                else:             print('{:4d}'.format(int(prediction[irow-row,0])))                



    fig, ax = plt.subplots()

    if (iTrans == 1):

        im = ax.imshow(np.transpose(mat), cmap=plt.cm.viridis)

    else:

        im = ax.imshow(mat, cmap=plt.cm.viridis)

    plt.show()
path = "/kaggle/input/digit-recognizer/test.csv"

normalize    = 255

row = 1

Nshow = 49

visualize(path, row, Nshow, normalize)
def get_Conv_Layer_Output(kernel, bias, inputs, padding="same", stride=1):

    (W, H) = inputs.shape

    (kernel_width, kernel_height) = kernel.shape



    if (padding=="same"):

        Wm = W + kernel_width-1

        Wn = W + kernel_height-1

        padded_data = np.zeros((Wm, Wn))

        i0 = np.int(kernel_width/2)

        j0 = np.int(kernel_height/2)

        for i in range(Wm-kernel_width+1):

            for j in range(Wn-kernel_height+1):

                padded_data[i0+i,j0+j] = inputs[i,j]

    else:

        Wm = W

        Wn = W

        padded_data = np.copy(inputs)

    

    outputs = np.zeros((Wm-kernel_width+1,Wn-kernel_height+1))    

    for i in range(Wm-kernel_width+1):

        for j in range(Wn-kernel_height+1):

            outputs[i,j] = bias + np.sum(padded_data[i:i+kernel_width,j:j+kernel_height] * kernel[:])

            if (outputs[i,j] < 0): outputs[i,j] = 0.00



    return outputs            
def get_MaxPool_Layer_Output(inputs,pool_size=(2,2)):

    (W, H) = inputs.shape

    pool_width  = pool_size[0]

    pool_height = pool_size[1]

    

    Wo = np.int(W/pool_width)

    Ho = np.int(H/pool_height)



    outputs = np.zeros((Wo,Ho))

    for i in range(Wo):

        i1 = i*pool_width

        i2 = (i+1)*pool_width

        for j in range(Ho):

            j1 = j*pool_height

            j2 = (j+1)*pool_height

            outputs[i,j] = np.max(inputs[i1:i2,j1:j2])



    return outputs    
model_Weights = model_CNN.get_weights()

# print(model_Weights[0].shape)

# print(model_Weights[1].shape)

# for i in range(5):

#     for j in range(5):

#         print('{:10.4f}'.format(kernel_0[i,j]), end="")

#     print('')

# print(bias_0)



for weight in model_Weights:

    print(weight.shape)



path = "/kaggle/input/digit-recognizer/test.csv"

data = 4

normalize = 255

A, M, N = loadtestdata(path, normalize)

input_data = A[data-1,:]

N = input_data.shape

W = np.int(np.sqrt(N))

inputs = np.reshape(input_data, (W,W))

fig0, ax0 = plt.subplots(1,1,figsize=(3,3))

im0 = ax0.imshow(inputs, cmap=plt.cm.viridis)

fig0.show()



fig1, ax1 = plt.subplots(1,6,figsize=(14, 4.75))

for channels in range(6):

    kernel_0 = model_Weights[0][0:5,0:5,0,channels]

    bias_0   = model_Weights[1][channels]

    layer_output = get_Conv_Layer_Output(kernel_0, bias_0, inputs, "same")

    im1 = ax1[channels].imshow(layer_output, cmap=plt.cm.viridis)

plt.show()



fig2, ax2 = plt.subplots(1,6,figsize=(12, 2.00))

for channels in range(6):

    kernel_0 = model_Weights[0][0:5,0:5,0,channels]

    bias_0   = model_Weights[1][channels]

    layer_output1 = get_Conv_Layer_Output(kernel_0, bias_0, inputs, "same")

    layer_output2 = get_MaxPool_Layer_Output(layer_output1)

    im2 = ax2[channels].imshow(layer_output2, cmap=plt.cm.viridis)

fig2.tight_layout(pad=3.0)

plt.show()
