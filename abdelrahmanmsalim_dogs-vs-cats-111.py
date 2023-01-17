
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame, Series
import random
from tqdm import tqdm
import os
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
from keras.utils import to_categorical
import glob
from matplotlib import pyplot as plt
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Flatten, MaxPool2D
from keras.optimizers import adam
from keras import regularizers
from keras.utils import plot_model
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Dense, Dropout
from keras import backend as K

# Any results you write to the current directory are saved as output.
train_path = '../input/train/*.jpg'
x_train_adres = glob.glob(train_path)

m_train = len(x_train_adres)
y_train = np.zeros((m_train,1))
for i,ca in enumerate(x_train_adres):
    if 'cat' in ca:
        y_train[i] = 1
print(y_train.shape)
  
# print(y_train)
# print(x_train_adres[m_train-1])

wid = 100
n = wid*wid*3
x_train = np.zeros((m_train, wid, wid, 3), dtype = np.float32)
# for i in tqdm(range(5)):
for i in tqdm(range(len(x_train_adres))):
    if i%1000 ==0:
        print(i)
    img = cv2.imread(x_train_adres[i])
#     plt.imshow(img)
#     plt.show()  
#     print(y_train[i])
    
    img = (cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(wid,wid),interpolation=cv2.INTER_CUBIC))/255
    x_train[i] = img
#     print(x_train[i])
    del img

# img = cv2.resize(x_train[1000], (wid,wid))
# plt.imshow(img)
# plt.show()  
# print(y_train[1000])
acc= []
val_acc= []
loss= []
val_loss= []

lamda = .0001
inputs = Input(shape = (wid,wid,3))

x = Conv2D(16, kernel_size=(3,3), activation = 'relu', kernel_regularizer=regularizers.l2(lamda))(inputs)
x = MaxPool2D()(x)
x = Conv2D(32, kernel_size=(3,3), activation = 'relu', kernel_regularizer=regularizers.l2(lamda))(x)
x = MaxPool2D()(x)
x = Conv2D(64, kernel_size=(3,3), activation = 'relu', kernel_regularizer=regularizers.l2(lamda))(x)
x = MaxPool2D()(x)
x = Conv2D(128, kernel_size=(3,3), activation = 'relu', kernel_regularizer=regularizers.l2(lamda))(x)
x = MaxPool2D()(x)
x = Conv2D(256, kernel_size=(3,3), activation = 'relu', kernel_regularizer=regularizers.l2(lamda))(x)
x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(.5)(x)

x = Dense(256, activation='relu')(x)
x = Dropout(.5)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(.5)(x)

output = Dense(1,  activation = 'sigmoid')(x)

model = Model(inputs, output)
opt = adam(lr=.001, beta_1=0.9, beta_2=0.999)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train,
              batch_size=64,
              epochs=20,
              validation_split = .1, 
              shuffle = True)


acc += history.history['acc'] 
val_acc += history.history['val_acc'] 

plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


loss += history.history['loss'] 
val_loss += history.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# print(history.params)
# history = model.fit(x_train, y_train,
#               batch_size=64,
#               epochs=50,
#               validation_split = .1)


# acc += history.history['acc'] 
# val_acc += history.history['val_acc'] 

# plt.plot(acc)
# plt.plot(val_acc)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# loss += history.history['loss'] 
# val_loss += history.history['val_loss']
# plt.plot(loss)
# plt.plot(val_loss)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')
#print(model.get_weights())
test_path = '../input/test/*.jpg'
x_test_adres = glob.glob(test_path)
print(x_test_adres[0])
m_test = len(x_test_adres)
y_test = np.zeros((m_test,1))

print(y_test.shape)
  

x_test = np.zeros((m_test, wid, wid, 3), dtype = np.float32)
# for i in tqdm(range(5)):
print('Processing...')

for i, name in enumerate(x_test_adres):
    if i%1000 ==0:
        print(i)
    
    img = cv2.imread(x_test_adres[i])
#     plt.imshow(img)
#     plt.show()  

    img = (cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(wid,wid),interpolation=cv2.INTER_CUBIC))/255
    na = int(''.join([i for i in name if i.isdigit()]))
#     print(na)
    x_test[na-1] = img
    del img
print('Predicting...')
y_test = model.predict(x_test)
print(y_test)
plt.imshow(x_test[10005])
plt.show() 
print(y_test[10005])

frame = pd.DataFrame({'label': y_test.T.squeeze()})
frame = frame.reset_index(drop=True)
frame.index += 1 
frame.to_csv("Dogs Vs. Cats.csv", index_label='id')

# print(y_test[10000])
print(frame)

