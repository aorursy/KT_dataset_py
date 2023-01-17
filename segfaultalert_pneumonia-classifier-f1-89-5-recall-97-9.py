# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
import cv2 
import glob
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout,GlobalAveragePooling2D
#from keras import backend as K

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/chest_xray/chest_xray/train/NORMAL"))
train_dir_normal = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/*.jpeg"
train_dir_pneumonia = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/*.jpeg"
val_dir_normal = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val/NORMAL/*.jpeg"
val_dir_pneumonia = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val/PNEUMONIA/*.jpeg"
test_dir_normal = "../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/*.jpeg"
test_dir_pneumonia = "../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/*.jpeg"
# Any results you write to the current directory are saved as output.
%matplotlib inline
#Helper functions !! :
def shuffle_in_unison(a,b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]
def convert_to_one_hot(vec, num):
    Y = np.eye(num)[vec.reshape(-1)].T
    return Y
    
def load_data():
    """A function to load the whole dataset and the labels"""
    train_data = []
    val_data = []
    test_data = []

    train_labels = [] 
    val_labels = []
    test_labels = []



    # groupping the file names into arrays and then we're gonna read each img and put it in the above array 
    train_files_normal = glob.glob(train_dir_normal) 
    train_files_pneumonia = glob.glob(train_dir_pneumonia)
    val_files_normal = glob.glob(val_dir_normal) 
    val_files_pneumonia = glob.glob(val_dir_pneumonia)

    test_files_normal = glob.glob(test_dir_normal) 
    test_files_pneumonia = glob.glob(test_dir_pneumonia)




    for num, file in enumerate(train_files_normal + train_files_pneumonia):
        img = cv2.imread(file, 1) 
        img = cv2.resize(img, (150,150))
        train_data.append(img)
        if(num+1 <= 1341):
            train_labels.append(0)
        else:
            train_labels.append(1)
    
    for num, file in enumerate(val_files_normal + val_files_pneumonia):
        img = cv2.imread(file, 1)
        img = cv2.resize(img, (150,150))
        val_data.append(img)
        if(num+1 <= 8):
            val_labels.append(0)
        else:
            val_labels.append(1)
    
    for num, file in enumerate(test_files_normal + test_files_pneumonia):
        img = cv2.imread(file, 1)
        img = cv2.resize(img, (150,150))
        test_data.append(img)
        if(num+1 <= 234):
            test_labels.append(0)
        else:
            test_labels.append(1)
    
    
    X_train = np.array(train_data)
    X_val  = np.array(val_data)
    X_test = np.array(test_data)
    
    Y_train = np.array(train_labels)
    Y_val = np.array(val_labels)
    Y_test = np.array(test_labels) 
    
    X_train, Y_train = shuffle_in_unison(X_train,Y_train)
    X_val, Y_val = shuffle_in_unison(X_val,Y_val)
    X_test, Y_test = shuffle_in_unison(X_test,Y_test)


    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test 
            

#print(train_labels[0:1345])

X_train,Y_train,X_val,Y_val,X_test,Y_test = load_data()

# Further preparing the data 
X_train = X_train/ 255 
X_val = X_val/ 255 
X_test = X_test/ 255

Y_train = convert_to_one_hot(Y_train, 2).T
Y_val = convert_to_one_hot(Y_val, 2).T
Y_test = convert_to_one_hot(Y_test, 2).T

print("shape of X_train : " + str(X_train.shape))
print("shape of X_val : " + str(X_val.shape))
print("shape of X_test : " + str(X_test.shape))
print("shape of Y_train : " + str(Y_train.shape))
print("shape of Y_val : " + str(Y_val.shape))
print("shape of Y_test : " + str(Y_test.shape))
plt.imshow(X_train[150])
print("y = " + str(np.squeeze(Y_train[150])))
# loading a pre-trained model(VGG16)
base_model = applications.VGG16(include_top = False, weights = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5') 
base_model.summary()
sgd = optimizers.Adam(lr = 1e-4)
# now we build a function to train the model and add a bottleneck model to our pre-trained one 
def Train_model(X_train, Y_train, X_test, Y_test, optimizer, base_model, batch_size = 64, num_epochs = 10):
    
    X = base_model.output
    X = Dropout(0.5)(X)
    X = GlobalAveragePooling2D()(X)
    X = Dense(128, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dense(2, activation='sigmoid')(X)
    model = Model(inputs = base_model.input, outputs = X)
    
    for layer in base_model.layers:
        layer.trainable = False 
        
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    model.fit(X_train,Y_train,
              batch_size = batch_size,
              epochs = num_epochs,
              validation_data = (X_test,Y_test))
    
    return model 
    
    
    
model = Train_model(X_train, Y_train, X_test, Y_test, sgd, base_model, batch_size = 64, num_epochs = 10)
model.save('model.h5')
s_model = model.save('model.h5')
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)
y_true = np.argmax(Y_test, axis = 1) 
conf_mat = confusion_matrix(y_true, y_pred) 
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat = conf_mat ,  figsize=(5, 5))
plt.show()
# recall = 375 /(375+15) = .961 -> 96.1% 
# precision = 375 / (375+64) = .854 -> 85.4% 
# F1 = 2/(1/recall + 1/precision) = 90.4%
