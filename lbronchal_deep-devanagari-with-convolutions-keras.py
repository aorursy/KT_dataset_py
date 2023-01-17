import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
dataset = pd.read_csv("../input/dataset.csv")
dataset.head()
X = dataset.iloc[:,:-1]

Y_d = dataset.iloc[:,-1]
num_pixels = X.shape[1]

num_classes = 46

img_width = 32

img_height = 32

img_depth = 1
X_images = X.values.reshape(X.shape[0], img_width, img_height)
for i in range(1, 9):    

    plt.subplot(240+i)

    plt.axis('off')

    plt.imshow(X_images[i-1], cmap=plt.get_cmap('gray'))

plt.show()
dataset.iloc[:,1024].value_counts()
rows_to_remove = np.where(dataset.iloc[:,1024].values==1024)

rows_to_remove
plt.imshow(X_images[2000], cmap=plt.get_cmap('gray'))

plt.axis('off')

plt.show()
dataset = dataset.drop(dataset.index[rows_to_remove[0]])
X = dataset.iloc[:,:-1]

X_images = X.values.reshape(X.shape[0], img_width, img_height)

Y_d = dataset.iloc[:,-1]
# output in binary format

from sklearn.preprocessing import LabelBinarizer

binencoder = LabelBinarizer()

Y = binencoder.fit_transform(Y_d)
# data normalization

X = X / 255
seed = 123 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
from keras.models import Sequential

from keras.layers import Dense

from keras.models import Model
def baseline_model():

    model = Sequential()

    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model = baseline_model()



epochs = 10

batch_size = 400

history = model.fit(X_train.values, y_train, validation_split=0.20, epochs=epochs, batch_size=batch_size, verbose=2)
scores = model.evaluate(X_test.values, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')



plt.legend(['train', 'validation'], loc='upper left')

plt.show()
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.callbacks import EarlyStopping

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
seed = 123 

X_train, X_test, y_train, y_test = train_test_split(X_images, Y, test_size=0.20, random_state=seed)



X_train = X_train/255

X_test = X_test/255
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, img_depth).astype('float32')

X_test = X_test.reshape(X_test.shape[0], img_width, img_height, img_depth).astype('float32')
def cnn_model():

    model = Sequential()

    model.add(Conv2D(32, (4, 4), input_shape=(img_height, img_width, img_depth), 

                     activation='relu', name="conv_1"))

    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_1"))

    model.add(Conv2D(64, (3, 3), activation='relu', name="conv_2"))

    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_2"))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', name="dense_1"))

    model.add(Dense(50, activation='relu', name="dense_2"))

    model.add(Dense(num_classes, activation='softmax', name="modeloutput"))

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model = cnn_model()



early_stopping_monitor = EarlyStopping(patience=2)



epochs = 10

batch_size = 200

history = model.fit(X_train, y_train, validation_split=0.20, epochs=epochs, batch_size=batch_size, 

                    callbacks=[early_stopping_monitor], verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')



plt.legend(['train', 'validation'], loc='upper left')

plt.show()