import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, BatchNormalization
from keras.layers import Dense, Dropout

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras
train_data = pd.read_csv("../input/train.csv")
train_data.head()
train_data.shape
images = train_data.iloc[:,1:].values
labels = train_data.iloc[:,0].values
def show_image(number):
    image = images[number]
    plt.axis('off')
    plt.imshow(image.reshape(28,28))
    plt.title(labels[number])
show_image(24)
show_image(890)
show_image(32190)
X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.2, random_state=0)
print("Length of X_train:", len(X_train))
print("Length of Y_train:", len(Y_train))
print("Length of X_val:", len(X_val))
print("Length of Y_train:", len(Y_val))
X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)
Y_train_one_hot = np_utils.to_categorical(Y_train, 10)
Y_validation_one_hot = np_utils.to_categorical(Y_val, 10)
classifier = Sequential()

classifier.add(Convolution2D(16,3,3, input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

classifier.add(Convolution2D(32,3,3, input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))

classifier.add(Flatten())

classifier.add(BatchNormalization())
classifier.add(Dense(output_dim = 64, activation='relu'))
classifier.add(Dense(output_dim = 10, activation='softmax'))

classifier.summary()

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False)
train_set = train_datagen.flow(X_train, Y_train_one_hot, batch_size=32)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_set = validation_datagen.flow(X_val, Y_validation_one_hot, batch_size=32)


classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit_generator(train_set,
                    steps_per_epoch=33600,epochs=5,
                    validation_data=(validation_set), validation_steps=8400, shuffle=True)
X_test = pd.read_csv("../input/test.csv")
X_test.head()
X_test = X_test.iloc[:,:].values
X_test = (X_test)*1./255
X_test = X_test.reshape(-1,28,28,1)
predictions = classifier.predict(X_test)
predictions = np.argmax(predictions,axis = 1)
ImageId = np.arange(1,28001)
test_images = X_test
test_labels = predictions
def show_test_image(number):
    test_image = test_images[number]
    plt.axis('off')
    plt.imshow(test_image.reshape(28,28))
    plt.title("Predicted:{}".format(test_labels[number]))
show_test_image(54)
show_test_image(3439)
show_test_image(15439)
Label = predictions
submission = pd.DataFrame()
submission['ImageId'] = ImageId
submission['Label'] = Label
submission.to_csv("MNIST2.csv",index=False)