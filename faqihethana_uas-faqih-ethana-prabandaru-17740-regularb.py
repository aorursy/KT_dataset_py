import numpy as np
import pandas as pd 
import os
import cv2
import skimage.transform
import skimage.io
import matplotlib.pyplot as plt
from random import seed
from random import random
from statistics import mean
from shutil import move

data_dir1 = '/kaggle/input/flowers-recognition/flowers/flowers/sunflower'
data_dir2 = '/kaggle/input/flowers-recognition/flowers/flowers/dandelion'
data_dir3 = '/kaggle/input/flowers-recognition/flowers/flowers/rose'

def ambilGambar(folder):
    images = []
    labels = []
    x = 0
    if folder == data_dir1:
        for i in range(100):
            y = 0
            labels.append(y)
    elif folder == data_dir2:
        for i in range(100):
            y = 1
            labels.append(y)
    elif folder == data_dir3:
        for i in range(100):
            y = 2
            labels.append(y)
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (320,240), interpolation = cv2.INTER_AREA) 
        if img is not None:
            images.append(img)
        x += 1
        if x == 100: break
    
    return images, labels

image_sun, label_sun = ambilGambar(data_dir1)
image_dan, label_dan = ambilGambar(data_dir2)
image_ros, label_ros = ambilGambar(data_dir3)

image = image_sun + image_dan + image_ros
target = label_sun + label_dan + label_ros

print(len(image))
print(len(target))
plt.figure(figsize = (20,20))
for i in range(3):
    img = image[100*i]
    plt.subplot(1,3,i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("kelas ke-" + str(target[100*i]))
def sigmoid(x):
    return 1.0 /(1.0+np.exp(-x))
from random import seed
from random import random
seed(1)
 
def initBobot(layer):
    nLayer = len(layer)
    bias = [np.random.randn(y,1) for y in layer[1:]]
    w = [np.random.randn(y,x) for x,y in zip(layer[:-1], layer[1:])]
    return [bias, w, nLayer]
def cost_func(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
def feedforward(x):
    weight = x
    weightNet = [x]
    dotProd = []
    for b, w in zip(Theta[0], Theta[1]):
        dots = np.dot(w, weight) + b
        dotProd.append(dots)
        weight = sigmoid(dots)
        weightNet.append(weight)
        
    return dotProd, weightNet
def backpropagation(x,y):
    deltaBias = [np.zeros(b.shape) for b in Theta[0]]
    deltaW = [np.zeros(w.shape) for w in Theta[1]]
    
    dotProd, weight = feedforward(x)
    
    loss = cost_func(weight[-1], y)
    delta_cost = weight[-1] - y
    
    delta = delta_cost
    
    deltaBias[-1] = delta
    deltaW[-1]= np.dot(delta, weight[-2].T)
    
    for x in range(2, Theta[2]):
        dots = dotProd[-x]
        deltaA = delta_sigmoid(dots)
        delta = np.dot(Theta[1][-x + 1].T, delta) * deltaA
        deltaBias[-x] = delta
        deltaW[-x] = np.dot(delta, weight[-x - 1].T)
        
    return(loss, deltaBias, deltaW)
def prediksi(X):
    prediksi = np.array([])
    labels = ["sunflower","dandelion","rose"]
    for x in X:
        dots, weight = feedforward(x)
        prediksi = np.append(prediksi, np.argmax(weight[-1]))
    prediksi = np.array([labels[int(p)] for p in prediksi])
    return prediksi
def akurasi(X, y):
    tmp = 0
    for x, _y in zip(X, y):
        dots, weight = feedforward(x)
        
        if np.argmax(weight[-1]) == np.argmax(_y):
            tmp += 1
    ak = (float(tmp) / X.shape[0]) * 100
    return ak
from sklearn.model_selection import train_test_split as split
def train_test_split (X, Y, trainSize):
    X_train, X_test, Y_train, Y_test = split(X, Y, train_size = trainSize)
    return X_train, X_test, Y_train, Y_test
def one_hot_encode(target):
    kelas = np.max(target) + 1
    encoded = np.eye(kelas)[target]
    return encoded
def normalisasiPixel(data):
    return data/255
image = np.array(image)
image = normalisasiPixel(image)
image.shape
nNeurons = image.shape[1] * image.shape[2] * image.shape[3]
nNeurons
X_train, X_test, Y_train, Y_test = train_test_split(image, target, 0.8)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 0.8)
X_train_CNN = X_train
X_test_CNN = X_test
X_val_CNN = X_val
Y_train_CNN = Y_train
Y_test_CNN = Y_test
Y_val_CNN = Y_val
def reshape_x(data,a):
    data = data.reshape(-1,a,1)
    return data
X_train = reshape_x(X_train, nNeurons)
X_test = reshape_x(X_test, nNeurons)
X_val = reshape_x(X_val, nNeurons)
Y_train = np.array(Y_train)
y_train_enc = one_hot_encode(Y_train)
y_train_enc = y_train_enc.reshape(-1, 3, 1)
alpha = 0.1
epoch = 300
Theta = initBobot([X_train[0].size, 128, y_train_enc[0].size])
len(Theta)
def getBatch(X, y, size):
    for idx in range(0, X.shape[0], size):
        batch = zip(X[idx:idx+size],
                   y[idx:idx+size])
        yield batch
def delta_sigmoid(x):
    return sigmoid(x) * (1- sigmoid(x))
batch_size = 32
nBatch = int(X_train.shape[0] / batch_size)
Y_val = np.array(Y_val)
y_val_enc = one_hot_encode(Y_val)
y_val_enc = y_val_enc.reshape(-1, 3, 1)
historyLoss = []
historyAcc = []
for j in range(epoch):
    batch_iter = getBatch(X_train, y_train_enc, batch_size)
    for i in range(nBatch):
        batch = next(batch_iter)
        deltaBias = [np.zeros(b.shape) for b in Theta[0]]
        deltaW = [np.zeros(w.shape) for w in Theta[1]]
        for batch_X, batch_Y in batch:
            loss, delta_deltaBias, delta_deltaW = backpropagation(batch_X, batch_Y)
            deltaBias = [db + ddb for db, ddb in zip(deltaBias, delta_deltaBias)]
            deltaW = [dw + ddw for dw, ddw in zip(deltaW, delta_deltaBias)]
    Theta[1] = [w - (alpha/batch_size)*delw for w, delw in zip(Theta[1], deltaW)]
    Theta[0] = [b - (alpha/batch_size)*delb for b, delb in zip(Theta[0], deltaBias)]
    historyLoss.append(loss)
    acc = akurasi(X_val, y_val_enc)
    historyAcc.append(acc)
    print("\nEpoch : %d\tLoss: %f\tAkurasi: %f\n"%(j, loss, acc))
Y_test_arr = np.array(Y_test)
y_test_enc = one_hot_encode(Y_test_arr)
y_test_enc = y_test_enc.reshape(-1, 3, 1)
akurasi(X_test, y_test_enc)
print(prediksi(X_test))
plt.plot(range(len(historyLoss)),historyLoss)
plt.xlabel('epoch')
plt.show()
plt.plot(range(len(historyAcc)),historyAcc)
plt.xlabel('epoch')
plt.show()
len(Theta[0])
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
Y_train_CNN = to_categorical(Y_train_CNN,num_classes = 3)
Y_test_CNN = to_categorical(Y_test_CNN,num_classes = 3)
Y_val_CNN = to_categorical(Y_val_CNN,num_classes = 3)
plt.figure(figsize = (20,20))
for i in range(3):
    img = X_train_CNN[64*i]
    plt.subplot(1,3,i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(Y_train_CNN[64*i])
plt.show()
panjang = 320
lebar = 240
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3),padding="Same",activation="relu" , input_shape = (lebar,panjang,3)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(3,activation="softmax"))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
epoch = 300
batch_size = 32
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False, 
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=60,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode = "reflect"
    ) 
datagen.fit(X_train_CNN)
history = model.fit_generator(datagen.flow(X_train_CNN,Y_train_CNN,batch_size=batch_size),epochs= epoch,validation_data=(X_val_CNN,Y_val_CNN),
                              steps_per_epoch=X_train_CNN.shape[0] // batch_size)
print("Test Accuracy: {0:.2f}%".format(model.evaluate(X_test_CNN,Y_test_CNN)[1]*100))
plt.plot(range(len(history.history['accuracy'])),history.history['accuracy'])
plt.xlabel('epoch')
plt.show()
plt.plot(range(len(history.history['loss'])),history.history['loss'])
plt.xlabel('epoch')
plt.show()