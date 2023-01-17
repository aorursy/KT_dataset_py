import pandas as pd

import numpy as np



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
mnist_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

mnist_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(mnist_train.shape, mnist_test.shape)
mnist_train.head()
mnist_train.describe()
mnist_train.isna().any().any()

# There is no empty field. Data is clean already.
mnist_train_data = mnist_train.loc[:, "pixel0":]

mnist_train_label = mnist_train.loc[:, "label"]



mnist_train_data = mnist_train_data/255.0

mnist_test = mnist_test/255.0
digit_array = mnist_train.loc[3, "pixel0":]

arr = np.array(digit_array) 



#.reshape(a, (28,28))

image_array = np.reshape(arr, (28,28))



digit_img = plt.imshow(image_array, cmap=plt.cm.binary)

plt.colorbar(digit_img)

print("IMAGE LABEL: {}".format(mnist_train.loc[3, "label"]))
sns.countplot(mnist_train.label)

print(list(mnist_train.label.value_counts().sort_index()))
# Converting dataframe into arrays

mnist_train_data = np.array(mnist_train_data)

mnist_train_label = np.array(mnist_train_label)
mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 28, 28, 1)

print(mnist_train_data.shape, mnist_train_label.shape)
from tensorflow import keras



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D

from tensorflow.keras.optimizers import Adadelta

from keras.utils.np_utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import LearningRateScheduler
nclasses = mnist_train_label.max() - mnist_train_label.min() + 1

mnist_train_label = to_categorical(mnist_train_label, num_classes = nclasses)

print("Shape of ytrain after encoding: ", mnist_train_label.shape)
def build_model(input_shape=(28, 28, 1)):

    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(128, kernel_size = 4, activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))

    return model



    

def compile_model(model, optimizer='adam', loss='categorical_crossentropy'):

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    

    

def train_model(model, train, test, epochs, split):

    history = model.fit(train, test, shuffle=True, epochs=epochs, validation_split=split)

    return history
cnn_model = build_model((28, 28, 1))

compile_model(cnn_model, 'adam', 'categorical_crossentropy')

model_history = train_model(cnn_model, mnist_train_data, mnist_train_label, 100, 0.2)
plt.plot(model_history.history['accuracy'],label = 'ACCURACY')

plt.plot(model_history.history['val_accuracy'],label = 'VALIDATION ACCURACY')

plt.legend()
plt.plot(model_history.history['loss'],label = 'TRAINING LOSS')

plt.plot(model_history.history['val_loss'],label = 'VALIDATION LOSS')

plt.legend()
mnist_test_arr = np.array(mnist_test)

mnist_test_arr = mnist_test_arr.reshape(mnist_test_arr.shape[0], 28, 28, 1)

print(mnist_test_arr.shape)
predictions = cnn_model.predict(mnist_test_arr)
predictions_test = []



for i in predictions:

    predictions_test.append(np.argmax(i))
submission =  pd.DataFrame({

        "ImageId": mnist_test.index+1,

        "Label": predictions_test

    })



submission.to_csv('my_submission.csv', index=False)