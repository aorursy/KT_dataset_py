import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras import Model
from keras import Input
from keras.utils import plot_model
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import backend as K

from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import os

%matplotlib inline
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from google.colab import drive
drive.mount('/content/drive')
imgloc_train = []
label_train = []

imgloc_val = []
label_val = []

for dirname, _, filenames in os.walk('/content/drive/My Drive/Project Data/Train/Adults'):
    for filename in filenames[:-50]:
        imgloc_train.append((os.path.join(dirname, filename)))
        label_train.append(0)
    for filename in filenames[-50:]:
        imgloc_val.append((os.path.join(dirname, filename)))
        label_val.append(0)
        
for dirname, _, filenames in os.walk('/content/drive/My Drive/Project Data/Train/Teenagers'):
    for filename in filenames[:-50]:
        imgloc_train.append((os.path.join(dirname, filename)))
        label_train.append(1)
    for filename in filenames[-50:]:
        imgloc_val.append((os.path.join(dirname, filename)))
        label_val.append(1)
        
for dirname, _, filenames in os.walk('/content/drive/My Drive/Project Data/Train/Toddler'):
    for filename in filenames[:-50]:
        imgloc_train.append((os.path.join(dirname, filename)))
        label_train.append(2)
    for filename in filenames[-50:]:
        imgloc_val.append((os.path.join(dirname, filename)))
        label_val.append(2)
img_train = []
img_val = []

for i in range(0, len(imgloc_train)):
    img1 = cv2.imread(imgloc_train[i],1)
    img2 = np.array(img1)
    img2 = cv2.resize(img2,(224,224))
    img_train.append(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
    
for i in range(0, len(imgloc_val)):
    img1 = cv2.imread(imgloc_val[i],1)
    img2 = np.array(img1)
    img2 = cv2.resize(img2,(224,224))
    img_val.append(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
img_train = np.array(img_train)
label_train = np.array(label_train).reshape(-1,1)

img_val = np.array(img_val)
label_val = np.array(label_val).reshape(-1,1)
img_train.shape
i = 1900
plt.imshow(img_train[i])
print(label_train[i,0])
# plt.imshow(img_val[i])
x_train = img_train/255
x_val = img_val/255
enc_y = OneHotEncoder(handle_unknown='ignore')
enc_y.fit(label_train)
y_train = enc_y.transform(label_train).toarray()
y_val = enc_y.transform(label_val).toarray()
x_train.shape
y_train.shape
reg = l2(1e-2)

model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', input_shape = x_train[0].shape,
                 kernel_regularizer = reg))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(64, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', kernel_regularizer = reg))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(64, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', kernel_regularizer = reg))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(128, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', kernel_regularizer = reg))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(128, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', kernel_regularizer = reg))

model.add(Flatten())
model.add(Dense(1024, activation = 'relu', kernel_regularizer = reg))
model.add(Dense(256, activation = 'relu', kernel_regularizer = reg))
model.add(Dense(3, activation = 'softmax', kernel_regularizer = reg))

# model.load_weights('LeNet.h5')
model.summary()
adam = Adam(learning_rate = 0.000001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['categorical_accuracy'])
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('LeNet-Reg-loss-colab.h5', monitor='loss', mode='min', save_best_only=True, verbose=1)
mc_acc = ModelCheckpoint('LeNet-Reg-acc-colab.h5', monitor='val_categorical_accuracy', mode='max', save_best_only=True, verbose=1)
callbacks = [es, mc, mc_acc]

history = model.fit(
    x = x_train,
    y = y_train,
    epochs=100, batch_size=32,
    validation_data = (x_val,y_val),
    verbose = 1, callbacks = callbacks)
model1 = load_model('LeNet-Reg-loss-colab.h5')
model2 = load_model('LeNet-Reg-acc-colab.h5')
model1.save('/content/drive/My Drive/Project Data/LeNet-Reg-loss-colab.h5')
model2.save('/content/drive/My Drive/Project Data/LeNet-Reg-acc-colab.h5')
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()
plt.figure()
plt.plot(history.history['categorical_accuracy'], label='train')
plt.plot(history.history['val_categorical_accuracy'], label='val')
plt.legend()
plt.show()
#loss 0.92
#acc 0.57
model.evaluate(x_train,y_train)
ymodel = model.predict(x_val)
ymodel = enc_y.inverse_transform(ymodel)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
confusion_matrix(label_val,ymodel)
100*recall_score(label_val,ymodel, average='weighted')
img_val[(label_val != ymodel)[:,0]].shape
unmatched = img_val[(label_val != ymodel)[:,0]]
y_act = label_val[(label_val != ymodel)[:,0]]
y_pred = ymodel[(label_val != ymodel)[:,0]]
for i in range(unmatched.shape[0]):
  print("actual label : ", y_act[i])
  print("predicted label : ", y_pred[i])
  plt.figure()
  plt.imshow(unmatched[i])
  plt.show()
case = y_act == 0
print("y_act = 0 ==>", np.sum(case))
print("y_act = 0 and y_pred = 1 ==>", np.sum(y_pred[case] == 1))
print("y_act = 0 and y_pred = 2 ==>", np.sum(y_pred[case] == 2))
print("----------------")
case = y_act == 1
print("y_act = 1 ==>", np.sum(case))
print("y_act = 1 and y_pred = 0 ==>", np.sum(y_pred[case] == 0))
print("y_act = 1 and y_pred = 2 ==>", np.sum(y_pred[case] == 2))
print("----------------")
case = y_act == 2
print("y_act = 2 ==>", np.sum(case))
print("y_act = 2 and y_pred = 0 ==>", np.sum(y_pred[case] == 0))
print("y_act = 2 and y_pred = 1 ==>", np.sum(y_pred[case] == 1))
print("----------------")
