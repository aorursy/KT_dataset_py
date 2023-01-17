import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
seed = 7
np.random.seed(seed)
tpure = np.load("../input/train_images_pure.npy")
tnoisy = np.load("../input/train_images_noisy.npy")
trotated = np.load("../input/train_images_rotated.npy")
tboth = np.load("../input/train_images_both.npy")
tlabel = pd.read_csv("../input/train_labels.csv")
tlabel
plt.subplot(221)
plt.imshow(tpure[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(tpure[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(tpure[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(tpure[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
plt.subplot(221)
plt.imshow(tnoisy[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(tnoisy[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(tnoisy[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(tnoisy[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
for i in range(15,19):
    plt.subplot(221+(i%5))
    plt.imshow(trotated[i], cmap=plt.get_cmap('gray'))
plt.show()
plt.subplot(221)
plt.imshow(tboth[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(tboth[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(tboth[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(tboth[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
K.set_image_dim_ordering('th')
def DataPrep(db):
    db = db.reshape(db.shape[0], 1, 28, 28).astype('float32')
    db = db / 255
    db = np_utils.to_categorical(db)
    return db
tlabel = np_utils.to_categorical(tlabel['label'])
tpure = DataPrep(tpure)
trotated = DataPrep(trotated)
def deepCNN():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(tlabel.shape[1], activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
CNNmodel = deepCNN()
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
Xtrain,Xvalidation,Ytrain,Yvalidation = train_test_split(tpure,tlabel, test_size = 0.2)
CNNmodel.fit(Xtrain, Ytrain, validation_data=(Xvalidation,Yvalidation), epochs=20, 
          batch_size=200, verbose=1, callbacks = callbacks)
