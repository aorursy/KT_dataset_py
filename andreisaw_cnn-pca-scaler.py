import warnings
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from skimage.exposure import equalize_hist
from skimage.filters import gaussian

%matplotlib inline

warnings.filterwarnings('ignore')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
size = 80
shape =  2 * (size,)
X, y = shuffle(np.load('../input/x_train.npy'), np.load('../input/y_train.npy'))
data_test = np.load('../input/x_test.npy')
X[0].reshape(shape)
data_test[0].reshape(shape)
def prepare(img):
    img = img.reshape(shape)
    img = equalize_hist(img)
    img = gaussian(img, sigma=1)
    img - img.mean()
    
    return img.flatten()
    
def transform(X):
    height, width = X.shape
    for i in range(height):
        X[i] = prepare(X[i]) 
        
    return X
X = transform(X)
data_test = transform(data_test)
d=np.concatenate((X,data_test))
d.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(d)
X=scaler.transform(X)
data_test=scaler.transform(data_test)
X=X.reshape(X.shape[0], size,size,1)
data_test=data_test.reshape(data_test.shape[0], size,size,1)
X.shape,data_test.shape
width, height = 8, 8

plt.figure(figsize=(16, 20))
for n, (image, name) in enumerate(zip(X, y), 1):
    if n > width * height:
        break
    
    plt.subplot(height, width, n)
    plt.title(name)
    plt.imshow(image.reshape(shape), cmap='gray')
width, height = 8, 8

plt.figure(figsize=(16, 20))
n=1
for image in data_test:
    if n > width * height:
        break
    
    plt.subplot(height, width, n)
    plt.imshow(image.reshape(shape), cmap='gray')
    n+=1
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
le=LabelEncoder()
y_transform=le.fit_transform(y)
lb=LabelBinarizer()
y_lb=lb.fit_transform(y_transform)
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(X, y_lb, test_size=0.4, random_state=0)
trainX.shape, testX.shape
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D 
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
def createModel():
    
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(size, size, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (2, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40, activation='softmax'))
    
    return model
my_network = createModel()
import keras
batch_size = 128
epochs = 50

my_network.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.rmsprop(),
                   metrics=['accuracy'])
print('The crash is right after this')
history = my_network.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, validation_data=(testX, testY))
my_network.evaluate(testX, testY)
my_network.save_weights('try_cnn2.h5')
prediction= my_network.predict(data_test)
prediction
y_classes = [np.argmax(y, axis=None, out=None) for y in prediction]
y_classes[:10]
preds=le.inverse_transform(y_classes)
preds[:10]
with open('predictions.csv', 'w') as out:
    print('Id,Name', file=out)
    for pair in enumerate(preds, 1):
        print('%i,%s' % pair, file=out)
