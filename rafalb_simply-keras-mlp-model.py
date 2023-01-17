# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
def convertData(df_train, df_test):
    # return numpy array as follow y_train, X_train, y_test, X_test
    y_train, X_train = df_train.iloc[:, 0].values, df_train.iloc[:, 1:].values
    X_test = df_test.values
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_train, X_train, X_test = y_train.astype(float), X_train.astype(float), X_test.astype(float)
    #check data
    assert(X_train.shape[1] == 784)
    assert(X_test.shape[1] == 784)
    assert(y_train.shape == (X_train.shape[0], 1))
    
    return y_train, X_train, X_test
y_train, X_train, X_test = convertData(df_train, df_test)
fig = plt.figure(figsize=(10,10))
for idx in range(25):
    plt.subplot(5,5, idx+1)
    plt.imshow(X_train[idx].reshape((28,28)), cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[idx]))
    
plt.tight_layout()
def plot_digit(digit, dem=28, font_size=12):
    max_ax = font_size * dem
    
    fig = plt.figure(figsize=(14,14))
    plt.xlim([0, max_ax])
    plt.ylim([0, max_ax])
    plt.axis('off')

    for idx in range(dem):
        for jdx in range(dem):
            t = plt.text(idx*font_size, max_ax - jdx*font_size, digit[jdx][idx], fontsize=font_size, color="#000000")
            c = digit[jdx][idx] / 255.
            t.set_bbox(dict(facecolor=(c, c, c), alpha=0.5, edgecolor='#f1f1f1'))
            
    plt.show()
print(y_train[0])
plot_digit(X_train[0].reshape((28,28)))
print(y_train[200])
plot_digit(X_train[200].reshape((28,28)))
#scalling aour data

X_train /= 255
X_test /= 255
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras import backend as K
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

np.random.seed(1)
#Split data
X_train = X_train.reshape(-1, 28,28, 1)

X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.3)

#one hot encoding for target variable
y_train = keras.utils.np_utils.to_categorical(y_train)
y_dev = keras.utils.np_utils.to_categorical(y_dev)

print("Shape of train images data: " + str(X_train.shape) + "\n" +
      "Shape of target variable: " + str(y_train.shape))
print("Shape of dev images data: " + str(X_dev.shape) + "\n" +
      "Shape of dev target variable: " + str(y_dev.shape))

X_test.shape[1:]
def gold_function(x):
    return ((x + K.sqrt(K.pow(x, 2) + 4)) / 2 )
num_pixels = X_train.shape[1]
num_classes = y_train.shape[1]
model = Sequential([
    Dense(1024, input_dim=num_pixels, activation='relu'),
    Dropout(0.5),
    Dense(512, kernel_initializer='glorot_normal', activation='relu'),
    Dropout(0.2),
    Dense(256, kernel_initializer='glorot_normal', activation=gold_function),
    Dropout(0.2),
    Dense(2128, kernel_initializer='glorot_normal', activation=gold_function),
    Dense(num_classes, kernel_initializer='glorot_normal', activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
          batch_size=64, epochs=20, verbose=1,
          validation_data=(X_dev, y_dev))
input_size = X_train.shape
cnnModel = Sequential([
    
    Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)),
    MaxPool2D(pool_size=(2,2)),
    
    Conv2D(filters = 32, kernel_size = (3,3), activation ='relu'),
    MaxPool2D(pool_size=(2,2)),
    
    Flatten(),
    
    Dropout(0.5),
    Dense(128, kernel_initializer='glorot_normal', activation=gold_function),
    
    Dropout(0.2),
    Dense(num_classes, kernel_initializer='glorot_normal', activation='softmax')
])

cnnModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnnModel.summary()
X_dev.shape
historyCnn = cnnModel.fit(X_train, y_train,
          batch_size=1024, epochs=10, verbose=1,
          validation_data=(X_dev, y_dev))
score = cnnModel.evaluate(X_dev, y_dev, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("MLP Error: %.2f%%" % (100-score[1]*100))
plt.plot(historyCnn.history['val_' + 'acc'])
def draw_learning_curve(history, key = 'acc'):
    #learning curve function 
    plt.figure(figsize=(15,8))
    plt.plot(history.history[key])
    plt.plot(history.history['val_' + key])
    plt.title('Learning Curve')
    plt.xlabel(key.title())
    plt.ylabel('Epoch')
    plt.legend(['train', 'test'], loc = 'best')
    plt.show()
draw_learning_curve(history, 'acc')
draw_learning_curve(history, 'loss')
X_test.shape
probabilities = model.predict_classes(X_test)
imageId = [idx+1 for idx in range(len(probabilities))]
y_predict = pd.DataFrame({'ImageId': imageId, 'Label': probabilities})
y_predict.head()
y_predict.to_csv('Prediction', index=False)
