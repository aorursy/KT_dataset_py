import numpy as np
import h5py
import tensorflow as tf
import cv2
import os
import keras
from keras import regularizers
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from data_process import *
# from utils import *
import math
np.random.seed(13)
%matplotlib inline
def load_dataset(): 
    X = []
    Y = []

    paths = os.listdir("../input/signus/dataset/Dataset")
#     print(paths)
    for p in paths:
        root = "../input/signus/dataset/Dataset"
        cwd = os.path.join(root, p)
        folder = os.listdir(cwd)
#         print(folder)

        for files in folder:
#             print(files)
            loc = os.path.join(cwd, files)
#             print(os.listdir(loc))
            img = cv2.imread(loc)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (100,100))
            # print(img.shape)
            flat_img = img.reshape(-1).T
            # print(flat_img.shape)
            X.append(flat_img)
            Y.append(int(p))


    X = np.array(X)
    Y = np.array(Y)
    # print(Y.shape)
    X = X / 255

    return X, Y

x, y = load_dataset()
# print(x.shape)
# x=x[:2000,:]
# y=y[:2000]
print(x.shape)
print(y.shape)


X_train, X_test, Y_train_orig, Y_test_orig = train_test_split(x, y, test_size = 0.15, shuffle = True)
print(X_train.shape)
print(Y_train_orig.shape)
print(X_test.shape)
print(Y_test_orig.shape)

index = 1003
plt.imshow(X_train[index].reshape(100,100,3), cmap = "binary")
print(Y_train_orig[index])
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y
Y_train = convert_to_one_hot(Y_train_orig, 10)
Y_test = convert_to_one_hot(Y_test_orig, 10)
print(Y_train.shape)
print(Y_test.shape)
print("Number of training examples = ", X_train.shape[0])
print("Number of testing examples = ", X_test.shape[0])

# X_train = X_train.T
# X_test = X_test.T


print("Shape of X_train data = ", X_train.shape)
print("Shape of X_test data = ", X_test.shape)
print("Shape of Y_train data = ", Y_train.shape)
print("Shape of Y_test data = ", Y_test.shape)

train_x = X_train.reshape(-1, 100, 100 ,3)
test_x = X_test.reshape(-1, 100, 100 ,3)
print(train_x.shape, test_x.shape)
clf = Sequential()
clf.add(Conv2D(32, kernel_size=(3, 3),activation="relu",input_shape=(100,100,3),padding='same'))
# clf.add(LeakyReLU(alpha=0.1))
clf.add(MaxPooling2D((3, 3),padding='same'))
clf.add(Conv2D(64, (3, 3),activation="relu",padding='same'))
# clf.add(LeakyReLU(alpha=0.1))
clf.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
clf.add(Conv2D(128, (3, 3),activation="relu",padding='same'))
# clf.add(LeakyReLU(alpha=0.1))                  
clf.add(MaxPooling2D(pool_size=(3, 3),padding='same'))
clf.add(Flatten())
clf.add(Dense(128,activation="relu"))
# clf.add(LeakyReLU(alpha=0.1))                  
clf.add(Dense(10, activation='softmax'))

clf.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

clf.summary()

clf_train = clf.fit(train_x, Y_train, batch_size=32,epochs=100,verbose=1,validation_data=(test_x, Y_test))
print(clf_train)

clf.save_weights("signus_model.h5")
val_loss = clf_train.history["val_loss"]
val_acc = clf_train.history["val_acc"]
loss = clf_train.history["loss"]
acc = clf_train.history["acc"]

plt.plot(val_acc)
plt.plot(acc)
plt.title("model accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["val_acc","acc"])
plt.show()
plt.plot(val_loss)
plt.plot(loss)
plt.title("model loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["val_loss","loss"])
plt.show()
pred = clf.predict_classes(test_x)

for index in range(200):
    #index = 194
    print(pred[index])
    im_view = plt.imshow(test_x[index].reshape(100,100,3))
    plt.show(im_view)
file = os.listdir('../input/testforsignus')
print(file)
image = cv2.imread('../input/testforsignus/' + str(file[0]))

image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
image = cv2.resize(image,(100,100))
image=image.reshape(1,100,100,3)
image=image/255
pediction_single = clf.predict_classes(image)
print(pediction_single)
#     print(image[1])
im_show = plt.imshow(image[0])
plt.show(im_show)
