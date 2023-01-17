# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from matplotlib import pyplot as plt
import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
np.random.seed(7)

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)
#https://pinkwink.kr/1121
img_rows = 28
img_cols = 28

mnist_train = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
mnist_test = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')

mnist = pd.concat([mnist_train, mnist_test], axis=0)

# Get all mnist as training
mnist_train_label = mnist['label']
mnist_train_label = mnist_train_label

mnist_train_img = mnist.drop('label', axis=1).to_numpy()
mnist_train_img = mnist_train_img / 255  # scale
print(mnist_train_img.shape)
mnist_train_img.resize(70000, 28, 28, 1)
print(mnist_train_img.shape)

trainX = mnist_train_img[:60000 , :]
trainY = mnist_train_label[:60000]
testX = mnist_train_img = mnist_train_img[60000: , :]
testY = mnist_train_label[60000:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(trainX,trainY,test_size=0.25, random_state = 42)
input_shape = (28, 28, 1)

batch_size = 128
num_classes = 10
epochs = 20
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
testY = np_utils.to_categorical(testY)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1, 
                 validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
n = 1
m = x_test[n].reshape(28, 28)
m = np.swapaxes(m, 1, 0)

plt.imshow(m, cmap='Greys', interpolation='nearest')
plt.show()

print('The Answer is ', model.predict(x_test[n].reshape((1, 28, 28, 1))).argmax(axis=1))
print(y_test[n].argmax())
pred = model.predict(testX).argmax(axis=1)
y_real = testY.argmax(axis=1)
F_EA = (pred == y_real)
F_EA1 = np.where(F_EA == False)
i = 0
i_num = len(F_EA1[0])
badx = testX[F_EA1[0][0]].reshape(1, 784)
bady = np.array([[testY[F_EA1[0][0]].argmax()]])
while i != i_num:
    m = testX[F_EA1[0][i]].reshape(1, 784)
    n = np.array([[testY[F_EA1[0][i]].argmax()]])
    badx = np.concatenate((badx, m), axis=0)
    bady = np.concatenate((bady, n), axis=0)
    i += 1
col = np.concatenate((bady,badx), axis=1)
res1 = np.expand_dims(col,axis=0)
f = np.hstack((res1))
df = pd.DataFrame(f) 
df.to_csv('DTCsubmission.csv', index=False)
sub = pd.read_csv('DTCsubmission.csv')
sub.head(10)
wrong_result
import random

predicted_result = model.predict(badx.reshape(badx.shape[0],28,28,1))
predicted_labels = np.argmax(predicted_result, axis=1)

test_labels = bady

wrong_result = []

for n in range(0, len(test_labels)):
    if predicted_labels[n] != test_labels[n]:
        wrong_result.append(n)

#samples = random.choices(population=wrong_result, k=25)
samples = wrong_result[50:72]

count = 0
nrows = ncols = 5

plt.figure(figsize=(12,8))

for n in samples:
    count += 1
    plt.subplot(nrows, ncols, count)
    m = badx[n].reshape(28, 28)
    m = np.swapaxes(m, 1, 0)
    plt.imshow(m, cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n])
    plt.title(tmp)

plt.tight_layout()
plt.show()

########### 검증코드
datat = pd.read_csv('MNIST_TEST.csv')
Xtest = datat.iloc[: , 1:].values
Xtest = Xtest.reshape(362, 28, 28)
Xtest = np.transpose(Xtest, axes=(0, 2, 1))
Xtest = Xtest.reshape(362, 784)
Ytest = datat.iloc[: , 0].values.reshape(datat.shape[0], 1)

from keras.utils import np_utils
testY = np_utils.to_categorical(Ytest)
pred = model.predict(Xtest).argmax(axis=1)
y_real = testY.argmax(axis=1)

F_EA = (pred == y_real)
F_EA1 = np.where(F_EA == True)

print(len(F_EA1[0]) / F_EA.shape[0] * 100)

n = 39
m = Xtest[n].reshape(28, 28)
#m = np.swapaxes(m, 1, 0)
#m = np.transpose(m, (1, 0))

plt.imshow(m, cmap='Greys', interpolation='nearest')
plt.show()
print('Predict ', model.predict(Xtest[n].reshape((1, 784))).argmax(axis=1))
print(testY[n].argmax())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping


seed = 0
np.random.seed(1212)
df = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')

df_features = df.iloc[:, 1:785]
df_label = df.iloc[:, 0]


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_label,
                                                test_size = 0.2,

                                                random_state = 1212)
X_train_arr = np.array(X_train)
X_test_arr = np.array(X_test)
Y_train_arr = np.array(Y_train)
Y_test_arr = np.array(Y_test)

X_train_arr

X_train_reshape = X_train_arr.reshape(X_train_arr.shape[0], 784).astype('float32') / 255
X_test_reshape = X_test_arr.reshape(X_test_arr.shape[0], 784).astype('float32') / 255

Y_train_reshape = np_utils.to_categorical(Y_train_arr, 10)
Y_test_reshape = np_utils.to_categorical(Y_test_arr, 10)

X_train_reshape
Y_train_reshape

model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

# 모델 최적화 설정
MODEL_DIR = './model2/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./model2/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train_reshape, Y_train_reshape, validation_data=(X_test_reshape, Y_test_reshape), epochs=100, batch_size=256, verbose=0, callbacks=[early_stopping_callback,checkpointer])
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test_reshape, Y_test_reshape)[1]))
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss")
legend = ax[0].legend(loc='best')
ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best')

########### 검증코드
datat = pd.read_csv('/kaggle/input/2dconv/test.csv')
Xtest = datat.iloc[: , 1:].values
Xtest = Xtest.reshape(362, 28, 28)
Xtest = np.transpose(Xtest, axes=(0, 2, 1))
Xtest = Xtest.reshape(362, 784)
Ytest = datat.iloc[: , 0].values.reshape(datat.shape[0], 1)

from keras.utils import np_utils
testY = np_utils.to_categorical(Ytest)
pred = model.predict(Xtest).argmax(axis=1)
y_real = testY.argmax(axis=1)

F_EA = (pred == y_real)
F_EA1 = np.where(F_EA == True)

print(len(F_EA1[0]) / F_EA.shape[0] * 100)

n = 5
m = Xtest[n].reshape(28, 28)
#m = np.swapaxes(m, 1, 0)
#m = np.transpose(m, (1, 0))

plt.imshow(m, cmap='Greys', interpolation='nearest')
plt.show()
print('Predict ', model.predict(Xtest[n].reshape((1, 784))).argmax(axis=1))
print(testY[n].argmax())